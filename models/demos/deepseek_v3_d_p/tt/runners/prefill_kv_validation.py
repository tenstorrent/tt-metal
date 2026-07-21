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
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import DEFAULT_MODEL, get_adapter

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

    # Gather the persistent representation and reconstruct scaled FP8 only on the host. This keeps
    # validation compatible with both cache formats without allocating a BF16 cache on device.
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)

    def _to_host(tensor):
        return ttnn.to_torch(tensor, mesh_composer=composer)[:, :1]

    cache_full = kvpe_cache.unpack_host(_to_host(kvpe_cache.storage)).to(torch.float32)

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
