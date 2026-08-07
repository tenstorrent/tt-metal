# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Model-agnostic post-prefill KV validation (bring-up / migration accuracy — never in serving).

The runner calls ``validate_after_prefill`` once after the request loop when PREFILL_VALIDATE_MIGRATION=1
(single-rank). All the ORCHESTRATION here is model-agnostic — the DONE-sentinel wait, the (src, dst)
pair parsing, the burst/pairwise dispatch, thresholds, and logging. It is driven by exactly two
per-model runtime PRIMITIVES (kept small so a new model implements almost nothing):

  * ``runtime.kv_cache_pcc_check(kv_cache, *, slot_id, n_chunks, real_len=None, pt_path_override=None)``
    — PCC one slot's cache against the golden trace; returns the min per-layer PCC (asserts on failure).
    Also the PREFILL_STANDALONE_PCC hook. This is the single place a model's KV layout / golden format
    lives.
  * ``runtime.read_slot_kv(kv_cache, slot) -> list[torch.Tensor]`` — the raw per-slot cache blocks for
    the golden-free dst==src pairwise compare, one tensor per cache tensor, shaped
    ``[num_layers, heads(or 1), seq_cache, head_dim]`` (replicas collapsed to 1). The blocks carry the
    same block-cyclic rotation on both endpoints, so comparing src vs dst directly is rotation-invariant
    (no un-rotation needed). Only needed for pairwise.

The runner never issues migrate itself — tt-llm-engine (the scheduler/driver) migrates the (src, dst)
pairs over the migration layer and writes the DONE sentinel; we read the runner-owned cache and PCC.
"""

import os
import time

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc


def _pcc_safe(a: torch.Tensor, b: torch.Tensor) -> float:
    """PCC that treats two all-zero tensors as a perfect match (1.0). A byte-copy migration makes dst
    bit-identical to src including the zeroed pad tail and dense-layer (all-zero) index_k, where a plain
    correlation is undefined; short-circuit those so they don't false-fail the dst==src check."""
    if not a.any() and not b.any():
        return 1.0
    return float(comp_pcc(a, b, 0.0)[1])


def validate_migration_kv(runtime, kv_cache, src_slot: int, dst_slot: int, n_chunks: int, real_len=None):
    """Burst: PCC the src slot (BEFORE) and dst slot (AFTER) against the SAME golden. A correct migration
    => the dst slot PCCs to golden exactly as the src slot does. Emits ``[kv-migrate-validate]`` lines."""
    logger.info(f"[kv-migrate-validate] BEFORE migration: validating SRC slot {src_slot}")
    src_pcc = runtime.kv_cache_pcc_check(kv_cache, slot_id=src_slot, n_chunks=n_chunks, real_len=real_len)
    print(f"[kv-migrate-validate] BEFORE src_slot={src_slot} min_pcc={src_pcc:.6f}")

    logger.info(f"[kv-migrate-validate] AFTER migration: validating DST slot {dst_slot}")
    dst_pcc = runtime.kv_cache_pcc_check(kv_cache, slot_id=dst_slot, n_chunks=n_chunks, real_len=real_len)
    print(f"[kv-migrate-validate] AFTER dst_slot={dst_slot} min_pcc={dst_pcc:.6f}")

    logger.success(
        f"[kv-migrate-validate] slot {src_slot} -> {dst_slot}: BEFORE(src)={src_pcc:.6f} AFTER(dst)={dst_pcc:.6f}"
    )
    return src_pcc, dst_pcc


def validate_migrations_pairwise(runtime, kv_cache, pairs):
    """Validate N concurrent src->dst migrations by asserting each dst slot's stored cache equals its src
    slot's, across every cache tensor (via ``runtime.read_slot_kv``) — golden-free fidelity + cross-talk
    detection, length-agnostic. Then optionally golden-anchor the slots named in PREFILL_MIGRATE_GOLDEN_PTS
    (a positional comma list of per-slot .pt paths, same order as the driver's --token-json)."""
    thr = float(os.environ.get("PREFILL_MIGRATE_PAIRWISE_PCC", "0.99"))

    failures = []
    for src, dst in pairs:
        src_blocks = runtime.read_slot_kv(kv_cache, src)
        dst_blocks = runtime.read_slot_kv(kv_cache, dst)
        assert len(src_blocks) == len(dst_blocks), (
            f"[kv-migrate-validate] src_slot={src} dst_slot={dst}: cache tensor count mismatch "
            f"{len(src_blocks)} != {len(dst_blocks)}"
        )
        min_pcc, worst = 1.0, None
        for ti, (sb, db) in enumerate(zip(src_blocks, dst_blocks)):
            assert sb.shape == db.shape, (
                f"[kv-migrate-validate] src_slot={src} dst_slot={dst}: tensor {ti} shape mismatch "
                f"{tuple(sb.shape)} != {tuple(db.shape)}"
            )
            for layer in range(sb.shape[0]):
                for head in range(sb.shape[1]):
                    pcc = _pcc_safe(sb[layer, head], db[layer, head])
                    if pcc < min_pcc:
                        min_pcc, worst = pcc, f"t{ti}[L{layer},h{head}]"
        del src_blocks, dst_blocks
        status = "PASS" if min_pcc >= thr else "FAIL"
        logger.info(
            f"[kv-migrate-validate] pairwise src_slot={src} dst_slot={dst} min_pcc={min_pcc:.6f} "
            f"(worst {worst}) -> {status}"
        )
        print(f"[kv-migrate-validate] AFTER pairwise src={src} dst={dst} min_pcc={min_pcc:.6f}")
        if min_pcc < thr:
            failures.append((src, dst, min_pcc))

    if failures:
        msg = "; ".join(f"{s}->{d} pcc={p:.6f}" for s, d, p in failures)
        raise AssertionError(f"[kv-migrate-validate] {len(failures)} pair(s) dst!=src below {thr}: {msg}")
    logger.success(f"[kv-migrate-validate] ALL {len(pairs)} pair(s) dst==src PASSED (>= {thr})")

    # Golden anchor (optional): PREFILL_MIGRATE_GOLDEN_PTS = positional comma list of .pt paths (entry i
    # anchors slot i, same order as --token-json; empty entry skips). Confirms each prefill is model-correct.
    golden_pt = {}
    pts = os.environ.get("PREFILL_MIGRATE_GOLDEN_PTS", "").strip()
    if pts:
        for slot, path in enumerate(pts.split(",")):
            if path.strip():
                golden_pt[slot] = path.strip()
    if not golden_pt:
        return

    n_pairs = max(1, len(pairs))
    gchunks = max(1, int(os.environ.get("PREFILL_STANDALONE_CHUNKED_NCHUNKS", "1")) // n_pairs)
    for s in sorted(golden_pt):
        d = next((dd for ss, dd in pairs if ss == s), None)
        gpt = golden_pt[s]
        logger.info(f"[kv-migrate-validate] golden anchor: src slot {s} (n_chunks={gchunks}) pt={gpt}")
        sp = runtime.kv_cache_pcc_check(kv_cache, slot_id=s, n_chunks=gchunks, pt_path_override=gpt)
        print(f"[kv-migrate-validate] GOLDEN src_slot={s} min_pcc={sp:.6f}")
        if d is not None:
            dp = runtime.kv_cache_pcc_check(kv_cache, slot_id=d, n_chunks=gchunks, pt_path_override=gpt)
            print(f"[kv-migrate-validate] GOLDEN dst_slot={d} min_pcc={dp:.6f}")


def validate_after_prefill(runtime, kv_cache, *, chunks_per_slot, real_end_per_slot, num_users, total_chunks):
    """Post-prefill KV validation entrypoint the runner calls (single-rank, after the serve loop).

      * PREFILL_VALIDATE_MIGRATION=1 -> wait for the driver's DONE sentinel, parse the migrated (src, dst)
        pairs, and validate them (PREFILL_MIGRATE_PAIRWISE=1 -> dst==src fidelity; else each pair vs golden).
      * otherwise -> PCC every populated slot vs golden.

    Drives ``runtime.kv_cache_pcc_check`` (always) and ``runtime.read_slot_kv`` (pairwise only)."""
    if not hasattr(runtime, "kv_cache_pcc_check"):
        raise RuntimeError(
            f"PREFILL_VALIDATE_MIGRATION/PCC requested but runtime {type(runtime).__name__} implements no "
            "kv_cache_pcc_check (see docs/ADDING_A_PREFILL_MODEL.md §2)."
        )
    ttnn.synchronize_device(runtime.mesh_device)

    if os.environ.get("PREFILL_VALIDATE_MIGRATION", "0") != "1":
        for slot, n in sorted(chunks_per_slot.items()):
            runtime.kv_cache_pcc_check(kv_cache, slot_id=slot, n_chunks=n, real_len=real_end_per_slot.get(slot))
        return

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
    ttnn.synchronize_device(runtime.mesh_device)

    if os.environ.get("PREFILL_MIGRATE_PAIRWISE", "0") == "1":
        if not hasattr(runtime, "read_slot_kv"):
            raise RuntimeError(
                f"PREFILL_MIGRATE_PAIRWISE=1 but runtime {type(runtime).__name__} implements no read_slot_kv "
                "(the per-slot raw-cache primitive; see ADDING_A_PREFILL_MODEL.md §2)."
            )
        validate_migrations_pairwise(runtime, kv_cache, pairs)
    else:
        for src_slot, dst_slot in pairs:
            n_src = chunks_per_slot.get(src_slot, total_chunks)
            validate_migration_kv(
                runtime, kv_cache, src_slot, dst_slot, n_src, real_len=real_end_per_slot.get(src_slot)
            )
    logger.success(f"[kv-migrate-validate] ALL {len(pairs)} migrated pair(s) PASSED")
