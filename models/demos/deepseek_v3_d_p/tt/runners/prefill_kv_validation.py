# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""KV-cache PCC validation for the DeepSeek / Kimi (MLA) prefill model.

The single home for the block-cyclic KV-cache PCC check and its golden-trace
loaders, plus the slot->slot and multi-pair migration validators. There is ONE
PCC entrypoint, ``kv_cache_pcc_check``, used by both paths:

  * the runner's standalone bring-up loop, via ``TtPrefillRuntime.kv_cache_pcc_check``
    (the runtime forwards here) — golden trace dir + per-rank ``first_layer_idx``;
  * the migration validators here (``validate_after_prefill`` and friends) — golden
    ``.pt`` or trace dir + ``real_len``.

``validate_after_prefill`` / ``validate_migration_kv`` / ``validate_migrations_pairwise``
have NO in-repo caller: they are driven by tt-llm-engine (the prefill scheduler /
migration driver) after it issues migrations. Keep their signatures in sync with that
caller. Everything prefixed ``_`` is internal.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import DEFAULT_MODEL, get_adapter
from models.demos.common.prefill.runners.runner_utils import resolve_trace_dir

if TYPE_CHECKING:
    # Type-only: importing the runtime at module load would pull the device/model stack into the
    # host-only callers of the golden loaders (e.g. test_chunked_trace_helpers). The runtime forwards
    # into this module lazily, so by the time these functions run the runtime is already imported.
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

# Model-aware golden trace dir, used only when DEEPSEEK_PREFILL_TRACE_{PT,DIR} are unset.
DEFAULT_PREFILL_TRACE_DIR = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL)).prefill_trace_default
_kv_pt_trace_cache: dict = {}


def _load_kv_pt_trace(pt_path: str) -> dict:
    """Load (and memoize) a `.pt` reference produced by `save_reference_cache`
    (see `utils.transformer_helpers`). Holds `ref_snapshots` + `ref_kvpe_list`;
    we only consume `ref_kvpe_list[i]` shape `[1, 1, seq, kv_lora + qk_rope_head_dim]`
    here. `mmap=True` keeps it lazy on first touch; subsequent layers are zero-copy
    slices into the same backing storage.

    Both `validate_migration_kv` PCC calls (BEFORE/AFTER) reuse one load via the
    module-level cache; the cache lives for the runner's lifetime.
    """
    import torch

    cached = _kv_pt_trace_cache.get(pt_path)
    if cached is not None:
        return cached
    cached = torch.load(pt_path, map_location="cpu", weights_only=True, mmap=True)
    if "ref_kvpe_list" not in cached:
        raise KeyError(
            f"DEEPSEEK_PREFILL_TRACE_PT={pt_path} missing 'ref_kvpe_list'. "
            f"Got keys: {list(cached.keys())}. Expected a save_reference_cache .pt."
        )
    _kv_pt_trace_cache[pt_path] = cached
    return cached


def _load_golden_kv_post(trace_dir, layer_idx: int, total_len: int) -> "torch.Tensor":
    """[total_len, 576] golden kv_post_transform for one layer, format-agnostic:
    - DeepSeek: a single kv_cache/layer_N.safetensors holding the full tensor.
    - Kimi (vllm): kv_cache/layer_N/rows_<start>_<end>.safetensors shards, concatenated by start row.
    """
    import torch
    from safetensors import safe_open

    key = f"kv_post_transform_layer_{layer_idx}"
    single = Path(trace_dir) / "kv_cache" / f"layer_{layer_idx}.safetensors"
    if single.exists():
        with safe_open(single, framework="pt") as f:
            return f.get_slice(key)[:total_len].to(torch.float32)
    layer_dir = Path(trace_dir) / "kv_cache" / f"layer_{layer_idx}"
    shards = sorted(layer_dir.glob("rows_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]))
    rows, have = [], 0
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            t = f.get_tensor(key)
        rows.append(t)
        have += t.shape[0]
        if have >= total_len:
            break
    return torch.cat(rows, dim=0)[:total_len].to(torch.float32)


def kv_cache_pcc_check(
    pipeline: "TtPrefillRuntime",
    kvpe_cache,
    *,
    slot_id: int,
    n_chunks: int,
    trace_dir=None,
    pt_path_override: str | None = None,
    real_len: int | None = None,
    first_layer_idx: int = 0,
) -> float:
    """Gather the engine-owned KV cache for `slot_id`, un-rotate the block-cyclic layout to natural
    order, and PCC-compare each layer against the golden DeepSeek-R1 `kv_post_transform` trace.
    Returns the min per-layer PCC and asserts (unless PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1) when
    any layer is below threshold.

    The single PCC entrypoint for both callers. Golden source, in priority order:
      1. `pt_path_override` / DEEPSEEK_PREFILL_TRACE_PT — a save_reference_cache .pt carrying
         ref_kvpe_list[layer] ([1, 1, seq, kv_lora + qk_rope_head_dim], already Meta-interleaved).
         Used by the migration validators.
      2. `trace_dir` (caller-resolved) — the runner's standalone loop passes the resolved
         PREFILL_TRACE_DIR golden here; descended via resolve_trace_dir.
      3. DEEPSEEK_PREFILL_TRACE_DIR env (default: the longbook_qa 56320 trace) — migration fallback.
    A trace dir holds kv_cache/layer_*.safetensors (or a Kimi row-sharded dir) keyed by
    kv_post_transform_layer_<global_layer>.

    `real_len` bounds the compare to written, non-pad positions (a partial last chunk overshoots the
    prompt; falls back to total_len when unset). `first_layer_idx` offsets the golden layer index for
    a pipeline-parallel rank: the device cache holds this rank's `num_layers` slice at local indices,
    but the golden is indexed by global layer (golden layer = first_layer_idx + local_idx).

    Env:
      PREFILL_STANDALONE_CHUNKED_PCC          min per-layer KV-cache PCC threshold (default 0.88)
      PREFILL_STANDALONE_CHUNKED_RECORD_ONLY  "1" -> log PCC only, do not assert
    """

    import torch

    from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
    from tests.ttnn.utils_for_testing import comp_pcc

    cfg = pipeline.config
    mesh_device = pipeline.mesh_device
    sp = cfg.sp_factor
    chunk_size = cfg.chunk_size
    num_layers = cfg.num_layers
    seq_len_cache = cfg.max_seq_len
    total_len = n_chunks * chunk_size
    # Compare only the REAL tokens. With a partial last chunk (prompt not a multiple of chunk_size),
    # total_len = n_chunks*chunk_size overshoots the prompt by the padding; real_len (the last chunk's
    # actual_end) bounds the compare to written, non-pad positions. Falls back to total_len for the
    # exact-multiple case (real_len unset).
    compare_len = real_len if real_len is not None else total_len

    pt_path = (pt_path_override or os.environ.get("DEEPSEEK_PREFILL_TRACE_PT", "")).strip()
    if pt_path:
        if not Path(pt_path).is_file():
            raise FileNotFoundError(f"DEEPSEEK_PREFILL_TRACE_PT={pt_path} does not exist or is not a file")
        kv_pt = _load_kv_pt_trace(pt_path)["ref_kvpe_list"]
        if len(kv_pt) < first_layer_idx + num_layers:
            raise RuntimeError(
                f"DEEPSEEK_PREFILL_TRACE_PT={pt_path} has {len(kv_pt)} layers in ref_kvpe_list "
                f"but this rank needs global layers [{first_layer_idx}, {first_layer_idx + num_layers}); "
                f"pick a .pt that matches the runner's layer count."
            )
        resolved_dir = None
    elif trace_dir is not None:
        # Standalone path: caller passes the PREFILL_TRACE_DIR golden; descend to the dir holding it.
        resolved_dir = resolve_trace_dir(trace_dir)
        kv_pt = None
    else:
        resolved_dir = Path(os.environ.get("DEEPSEEK_PREFILL_TRACE_DIR", DEFAULT_PREFILL_TRACE_DIR))
        if not resolved_dir.exists():
            raise FileNotFoundError(
                f"golden trace dir not found: {resolved_dir} "
                "(set DEEPSEEK_PREFILL_TRACE_DIR or DEEPSEEK_PREFILL_TRACE_PT, or pass trace_dir)"
            )
        kv_pt = None

    threshold = float(os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88"))
    record_only = os.environ.get("PREFILL_STANDALONE_CHUNKED_RECORD_ONLY", "0") == "1"
    kv_lora = pipeline.hf_config.kv_lora_rank
    kvpe_dim = pipeline.hf_config.qk_rope_head_dim + kv_lora

    # One gather: [num_users*num_layers, tp_replicas, seq_len_cache, kvpe] -> collapse TP via [:, :1].
    cache_full = ttnn.to_torch(
        kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.float32)[
        :, :1
    ]  # [num_users*num_layers, 1, seq_len_cache, kvpe]

    p = blockcyclic_positions(sp, chunk_size, seq_len_cache)
    logger.info(f"[kv-pcc] device KV cache vs golden kv_post_transform (slot={slot_id}, per layer):")
    min_pcc = 1.0
    failures = []
    for i in range(num_layers):
        # user-major slot layout: cache batch index = slot_id * num_layers + local_layer_idx
        batch_idx = slot_id * num_layers + i
        global_layer = first_layer_idx + i  # golden trace is indexed by global layer (PP rank offset)
        nat = torch.empty(seq_len_cache, kvpe_dim, dtype=torch.float32)
        nat[p] = cache_full[batch_idx, 0]  # un-rotate block-cyclic -> natural order
        dev_cache = nat[:compare_len]

        if kv_pt is not None:
            # ref_kvpe_list[global_layer] = ref_cache.key_cache[global_layer] from HF's DynamicCache
            # after the HF model's MLA forward. Per tests/test_prefill_transformer.py (canonical KVPE
            # PCC, lines ~664-671), this tensor is ALREADY in the device's rotary basis — pe is compared
            # directly with no re-interleave. Applying HF->Meta to it produces noise.
            g_post = kv_pt[global_layer][0, 0, :compare_len].to(torch.float32)
        else:
            # The safetensors trace stores `kv_post_transform_layer_<global_layer>` in HF half-split
            # layout (single-file DeepSeek or Kimi's row-sharded dir — _load_golden_kv_post auto-detects);
            # nope (kv_lora) compares directly, the pe slice is re-interleaved to Meta below. Load only
            # the populated [:compare_len] positions.
            g_post = _load_golden_kv_post(resolved_dir, global_layer, compare_len)
        _, pcc_nope = comp_pcc(g_post[:, :kv_lora], dev_cache[:, :kv_lora])
        ref_pe = g_post[:, kv_lora:]
        if kv_pt is not None:
            ref_pe_for_comp = ref_pe  # already Meta-interleaved (HF DynamicCache from save_reference_cache)
            basis_tag = "direct"
        else:
            d = ref_pe.shape[-1]
            ref_pe_for_comp = torch.stack([ref_pe[:, : d // 2], ref_pe[:, d // 2 :]], dim=-1).reshape(
                -1, d
            )  # HF -> Meta
            basis_tag = "interleaved"
        _, pcc_pe = comp_pcc(ref_pe_for_comp, dev_cache[:, kv_lora:])
        layer_pcc = min(pcc_nope, pcc_pe)
        min_pcc = min(min_pcc, layer_pcc)
        logger.info(
            f"  cache layer local={i} global={global_layer} PCC: "
            f"nope={pcc_nope:.6f} pe({basis_tag})={pcc_pe:.6f} -> {layer_pcc:.6f}"
        )
        if layer_pcc < threshold:
            failures.append((global_layer, layer_pcc))

    logger.info(f"[kv-pcc] KV cache min PCC across {num_layers} layers: {min_pcc:.6f} (threshold {threshold})")
    # stdout, not a log line: callers (tests / orchestrators) parse this.
    print(
        f"[standalone-chunked] kv_cache_pcc_complete slot={slot_id} n_chunks={n_chunks} "
        f"total_len={total_len} compare_len={compare_len} min_pcc={min_pcc:.6f}"
    )
    if failures:
        msg = "; ".join(f"layer {layer} PCC {pcc:.6f} < {threshold}" for layer, pcc in failures)
        if record_only:
            logger.warning(f"[kv-pcc] sub-threshold PCC (record-only, not asserted): {msg}")
        else:
            raise AssertionError(f"[kv-pcc] KV cache PCC below {threshold}: {msg}")
    else:
        logger.success(f"[kv-pcc] KV cache PCC PASSED (min {min_pcc:.6f} >= {threshold})")
    return min_pcc


def validate_migration_kv(
    pipeline: TtPrefillRuntime, kvpe_cache, src_slot: int, dst_slot: int, n_chunks: int, real_len: int | None = None
):
    """Validate the KV cache BEFORE and AFTER a slot->slot migration.

    The migration (src_slot -> dst_slot) is driven by tt-llm-engine (the prefill
    scheduler / driver over the migration layer) — the runner never issues migrate
    itself (see kv_chunk_table; the runner only publishes the table via SET_TABLE).
    This reuses `kv_cache_pcc_check` to PCC BOTH endpoints against the SAME golden
    `kv_post_transform` trace:

      * BEFORE: the SRC slot — the model-produced KV that tt-llm-engine migrates.
      * AFTER:  the DST slot — the migrated copy tt-llm-engine wrote.

    A correct migration => the DST slot PCCs to golden exactly as the SRC slot does, so
    a drop (or an empty / 0-PCC dst) flags a broken or absent migration. Emits
    `[kv-migrate-validate] BEFORE/AFTER` lines (orchestrators parse these).
    """
    logger.info(f"[kv-migrate-validate] BEFORE migration: validating SRC slot {src_slot} (real_len={real_len})")
    src_pcc = kv_cache_pcc_check(pipeline, kvpe_cache, slot_id=src_slot, n_chunks=n_chunks, real_len=real_len)
    print(f"[kv-migrate-validate] BEFORE src_slot={src_slot} min_pcc={src_pcc:.6f}")

    logger.info(f"[kv-migrate-validate] AFTER migration: validating DST slot {dst_slot} (real_len={real_len})")
    dst_pcc = kv_cache_pcc_check(pipeline, kvpe_cache, slot_id=dst_slot, n_chunks=n_chunks, real_len=real_len)
    print(f"[kv-migrate-validate] AFTER dst_slot={dst_slot} min_pcc={dst_pcc:.6f}")

    logger.success(
        f"[kv-migrate-validate] slot {src_slot} -> {dst_slot}: "
        f"BEFORE(src) min_pcc={src_pcc:.6f}, AFTER(dst) min_pcc={dst_pcc:.6f}"
    )
    return src_pcc, dst_pcc


def validate_migrations_pairwise(pipeline: TtPrefillRuntime, kvpe_cache, pairs):
    """Validate N concurrent slot->slot migrations of distinct prompts.

    Asserts each dst slot's KV equals its own src slot's (migration fidelity + cross-talk detection),
    via one host-side compare of the raw stored cache. Then golden-anchors the slots configured by
    PREFILL_MIGRATE_GOLDEN_PTS: a positional comma list of .pt paths indexed by slot (same order as
    the driver's --token-json; empty entry skips that slot), confirming each prefill is model-correct.
    Raises AssertionError on any failure.
    """
    import torch

    from tests.ttnn.utils_for_testing import comp_pcc

    cfg = pipeline.config
    mesh_device = pipeline.mesh_device
    num_layers = cfg.num_layers
    thr = float(os.environ.get("PREFILL_MIGRATE_PAIRWISE_PCC", "0.99"))

    # Read each pair sequentially to bound peak host memory. Gathering the whole cache at once is
    # [num_users*num_layers, 1, seq_len_cache, kvpe] in fp32 (~236 GiB at 32 users / 56K seq) and
    # OOMs/thrashes the host. Instead, per pair we slice only the src and dst user blocks off the
    # device ([num_layers, 1, seq_len_cache, kvpe] each, ~8 GiB) and free them before the next pair.
    # No un-rotation is needed: both endpoints carry the same block-cyclic rotation, so comparing
    # them directly is rotation-invariant.
    dev_shape = list(kvpe_cache.shape)  # slice dim 0 (user*layer); keep dims 1..3 full

    def _read_user_block(user: int) -> "torch.Tensor":
        # memory_config=DRAM interleaved is REQUIRED: kvpe_cache is ND-sharded ROUND_ROBIN_1D over
        # the 8 DRAM banks, and slicing it into another ND-shard produces a sub-tensor whose host
        # read-back miscomputes the DRAM core (TT_FATAL "logical DRAM core 8-0 ... num_views=8").
        # Forcing an interleaved output makes slice gather from the banks correctly and stay
        # host-readable (verified bit-exact vs the full-cache gather in test_kv_slice_read_repro).
        sl = ttnn.slice(
            kvpe_cache,
            [user * num_layers, 0, 0, 0],
            [(user + 1) * num_layers, dev_shape[1], dev_shape[2], dev_shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        block = ttnn.to_torch(
            sl, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
        ).to(torch.float32)[
            :, :1
        ]  # [num_layers, 1, seq_len_cache, kvpe]
        ttnn.deallocate(sl)
        return block

    failures = []
    for src, dst in pairs:
        src_block = _read_user_block(src)
        dst_block = _read_user_block(dst)
        min_pcc = 1.0
        for layer in range(num_layers):
            _, pcc = comp_pcc(src_block[layer, 0], dst_block[layer, 0])
            min_pcc = min(min_pcc, pcc)
        del src_block, dst_block
        status = "PASS" if min_pcc >= thr else "FAIL"
        logger.info(f"[kv-migrate-validate] pairwise src_slot={src} dst_slot={dst} min_pcc={min_pcc:.6f} -> {status}")
        print(f"[kv-migrate-validate] AFTER pairwise src={src} dst={dst} min_pcc={min_pcc:.6f}")
        if min_pcc < thr:
            failures.append((src, dst, min_pcc))

    if failures:
        msg = "; ".join(f"{s}->{d} pcc={p:.6f}" for s, d, p in failures)
        raise AssertionError(f"[kv-migrate-validate] {len(failures)} pair(s) dst!=src below {thr}: {msg}")
    logger.success(f"[kv-migrate-validate] ALL {len(pairs)} pair(s) dst==src PASSED (>= {thr})")

    # Golden anchor config. One knob: PREFILL_MIGRATE_GOLDEN_PTS = positional comma list of .pt
    # paths (entry i anchors slot i, same order as --token-json; empty entry skips that slot).
    # Back-compat: PREFILL_MIGRATE_GOLDEN_SLOT + per-slot PREFILL_MIGRATE_GOLDEN_PT_<slot>.
    golden_pt = {}
    pts = os.environ.get("PREFILL_MIGRATE_GOLDEN_PTS", "").strip()
    if pts:
        for slot, path in enumerate(pts.split(",")):
            if path.strip():
                golden_pt[slot] = path.strip()
    else:
        for tok in os.environ.get("PREFILL_MIGRATE_GOLDEN_SLOT", "").split(","):
            if tok.strip().isdigit():
                golden_pt[int(tok)] = os.environ.get(f"PREFILL_MIGRATE_GOLDEN_PT_{int(tok)}", "").strip() or None

    n_pairs = max(1, len(pairs))
    gchunks = max(1, int(os.environ.get("PREFILL_STANDALONE_CHUNKED_NCHUNKS", "1")) // n_pairs)
    for s in sorted(golden_pt):
        d = next((dd for ss, dd in pairs if ss == s), None)
        gpt = golden_pt[s]
        logger.info(f"[kv-migrate-validate] golden anchor: src slot {s} (n_chunks={gchunks}) pt={gpt or 'global'}")
        sp = kv_cache_pcc_check(pipeline, kvpe_cache, slot_id=s, n_chunks=gchunks, pt_path_override=gpt)
        print(f"[kv-migrate-validate] GOLDEN src_slot={s} min_pcc={sp:.6f}")
        if d is not None:
            dp = kv_cache_pcc_check(pipeline, kvpe_cache, slot_id=d, n_chunks=gchunks, pt_path_override=gpt)
            print(f"[kv-migrate-validate] GOLDEN dst_slot={d} min_pcc={dp:.6f}")


def validate_after_prefill(
    pipeline: TtPrefillRuntime,
    kvpe_cache,
    *,
    chunks_per_slot: dict,
    real_end_per_slot: dict,
    num_users: int,
    total_chunks: int,
    first_layer_idx: int = 0,
) -> None:
    """Post-prefill KV-cache validation entrypoint for the runner's PCC mode.

    Drains the device, then validates the populated KV cache against the golden trace:
      * PREFILL_VALIDATE_MIGRATION=1 -> wait for the driver's DONE sentinel, parse the
        migrated (src, dst) pairs, and validate them (PREFILL_MIGRATE_PAIRWISE -> dst==src
        fidelity for distinct prompts; else each pair vs the shared golden).
      * otherwise -> PCC every populated slot over its own [0, real_len) extent against
        its golden (per-slot DEEPSEEK_PREFILL_TRACE_PT list, or a shared trace).

    Raises on sentinel timeout or sub-threshold PCC. ``total_chunks`` is the loop's chunk
    count, used as the per-slot fallback. The runner calls this once after the loop.
    """
    # Drain the device, then validate the KV cache against the golden trace.
    ttnn.synchronize_device(pipeline.mesh_device)

    if os.environ.get("PREFILL_VALIDATE_MIGRATION", "0") == "1":
        # Migration mode: tt-llm-engine (the prefill scheduler/driver) migrates N
        # (src -> dst) pairs over the migration layer and writes a DONE sentinel when
        # prefill + all migrations finish. The sentinel CONTENT is the machine-readable
        # pair list ("src dst" per line) the driver wrote, so we validate exactly the
        # pairs that migrated (BEFORE=src, AFTER=dst) against the same golden. Each src
        # slot is validated with ITS OWN chunk count (not the loop total), since with
        # concurrent migrations the chunks are spread across N slots. Falls back to the
        # single PREFILL_MIGRATE_SRC/DST_SLOT env pair if the sentinel carries no pairs.
        done_file = os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")
        wait_s = int(os.environ.get("PREFILL_MIGRATE_WAIT_S", "1200"))
        logger.info(f"[kv-migrate-validate] waiting for DONE sentinel {done_file} (<= {wait_s}s)")
        deadline = time.time() + wait_s
        while not os.path.exists(done_file):
            if time.time() >= deadline:
                raise TimeoutError(
                    f"[kv-migrate-validate] sentinel {done_file} never appeared after {wait_s}s "
                    "(did the prefill driver finish prefill + migration?)"
                )
            time.sleep(0.5)
        # Parse "src dst" pairs from the sentinel; fall back to the env single pair.
        pairs = []
        try:
            for line in open(done_file).read().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    pairs.append((int(parts[0]) % num_users, int(parts[1]) % num_users))
        except OSError:
            pass
        if not pairs:
            pairs = [
                (
                    int(os.environ.get("PREFILL_MIGRATE_SRC_SLOT", "0")) % num_users,
                    int(os.environ.get("PREFILL_MIGRATE_DST_SLOT", "1")) % num_users,
                )
            ]
        logger.success(f"[kv-migrate-validate] sentinel found — validating {len(pairs)} pair(s): {pairs}")
        ttnn.synchronize_device(pipeline.mesh_device)
        if os.environ.get("PREFILL_MIGRATE_PAIRWISE", "0") == "1":
            # N distinct prompts: dst==src fidelity + optional per-slot golden anchor.
            validate_migrations_pairwise(pipeline, kvpe_cache, pairs)
        else:
            # Same prompt across slots: PCC each (src, dst) against the shared golden.
            for src_slot, dst_slot in pairs:
                n_src = chunks_per_slot.get(src_slot, total_chunks)  # per-slot chunk count (NOT the loop total)
                rl_src = real_end_per_slot.get(src_slot)  # real ISL (excludes pad); dst copies the same range
                validate_migration_kv(pipeline, kvpe_cache, src_slot, dst_slot, n_src, real_len=rl_src)
        logger.success(f"[kv-migrate-validate] ALL {len(pairs)} migrated pair(s) PASSED")
    else:
        # Validate EVERY populated slot, each over its own populated range: chunks_per_slot[s]
        # chunks and real_len = that slot's actual_end (excludes padding). Multi-user prefill
        # fills several slots with different prompts; each is PCC'd against its own golden over
        # only its real (non-pad) positions. kv_cache_pcc_check raises on a sub-threshold slot
        # (unless RECORD_ONLY), so any failure aborts here.
        #
        # Per-slot golden: DEEPSEEK_PREFILL_TRACE_PT may be a COMMA-SEPARATED list, one .pt per
        # slot in slot order (slot s -> golden[s]) — for multi-user runs where each slot holds a
        # different prompt. A single value (no comma) is the shared golden for every slot (the
        # kv_cache_pcc_check default env read handles that case).
        golden_list = [p.strip() for p in os.environ.get("DEEPSEEK_PREFILL_TRACE_PT", "").split(",") if p.strip()]
        multi_golden = len(golden_list) > 1
        slots = sorted(chunks_per_slot)
        logger.info(
            f"[request] running KV-cache PCC check for {len(slots)} slot(s): {slots} "
            f"(per-slot goldens={multi_golden})"
        )
        slot_pccs = {}
        for s in slots:
            real_len = real_end_per_slot.get(s)
            n_chunks_s = chunks_per_slot[s]
            if multi_golden:
                if s >= len(golden_list):
                    raise IndexError(
                        f"slot {s} has no golden: DEEPSEEK_PREFILL_TRACE_PT lists {len(golden_list)} "
                        f"golden(s) but slot {s} is populated. Provide one .pt per slot in slot order."
                    )
                gpt = golden_list[s]
            else:
                gpt = None  # kv_cache_pcc_check reads the single DEEPSEEK_PREFILL_TRACE_PT
            logger.info(f"[request]  -> slot={s} n_chunks={n_chunks_s} real_len={real_len} golden={gpt or '<shared>'}")
            slot_pccs[s] = kv_cache_pcc_check(
                pipeline,
                kvpe_cache,
                slot_id=s,
                n_chunks=n_chunks_s,
                pt_path_override=gpt,
                real_len=real_len,
                first_layer_idx=first_layer_idx,
            )
        logger.success(
            f"[request] all {len(slots)} slot(s) PASSED KV-cache PCC: "
            + ", ".join(f"slot{s}={p:.6f}" for s, p in sorted(slot_pccs.items()))
        )
