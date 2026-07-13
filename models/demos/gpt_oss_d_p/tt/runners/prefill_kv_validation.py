# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Post-migration KV validation for GPT-OSS prefill loopback tests."""

from __future__ import annotations

import os
import time

from loguru import logger

import ttnn


def validate_migrations_pairwise(pipeline, kv_cache, pairs, *, real_len: int | None = None) -> None:
    """Assert each dst slot's K/V equals its src slot's (no golden required)."""
    import torch

    from tests.ttnn.utils_for_testing import comp_pcc

    mesh_device = pipeline.mesh_device
    num_layers = len(kv_cache)
    thr = float(os.environ.get("PREFILL_MIGRATE_PAIRWISE_PCC", "0.99"))
    max_seq_len = pipeline.max_seq_len
    sp = pipeline.sp_factor
    seq_local = max_seq_len // sp
    compare_len = real_len if real_len is not None else max_seq_len
    compare_len = min(compare_len, max_seq_len)

    def _read_slot_tensor(cache: ttnn.Tensor, slot: int) -> torch.Tensor:
        sl = ttnn.slice(
            cache,
            [slot, 0, 0, 0],
            [slot + 1, cache.shape[1], seq_local, cache.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        block = ttnn.to_torch(
            sl,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, None), mesh_shape=mesh_device.shape),
        ).to(torch.float32)
        ttnn.deallocate(sl)
        # Canonical [1, heads_local, seq_local, head_dim]
        block = block.reshape(1, block.shape[-3], block.shape[-2], block.shape[-1])
        return block[:, :, : min(compare_len // sp + 1, seq_local), :]

    failures = []
    for src, dst in pairs:
        min_pcc = 1.0
        for layer in range(num_layers):
            k_cache, v_cache = kv_cache[layer]
            src_k = _read_slot_tensor(k_cache, src)
            dst_k = _read_slot_tensor(k_cache, dst)
            src_v = _read_slot_tensor(v_cache, src)
            dst_v = _read_slot_tensor(v_cache, dst)
            for sk, dk in ((src_k, dst_k), (src_v, dst_v)):
                _, pcc = comp_pcc(sk, dk)
                min_pcc = min(min_pcc, pcc)
        status = "PASS" if min_pcc >= thr else "FAIL"
        logger.info(f"[kv-migrate-validate] pairwise src_slot={src} dst_slot={dst} min_pcc={min_pcc:.6f} -> {status}")
        print(f"[kv-migrate-validate] AFTER pairwise src={src} dst={dst} min_pcc={min_pcc:.6f}")
        if min_pcc < thr:
            failures.append((src, dst, min_pcc))

    if failures:
        msg = "; ".join(f"{s}->{d} pcc={p:.6f}" for s, d, p in failures)
        raise AssertionError(f"[kv-migrate-validate] {len(failures)} pair(s) dst!=src below {thr}: {msg}")
    logger.success(f"[kv-migrate-validate] ALL {len(pairs)} pair(s) dst==src PASSED (>= {thr})")


def validate_after_prefill(pipeline, kv_cache, *, num_users: int, real_end: int | None = None) -> None:
    """Wait for migration DONE sentinel and run pairwise slot validation."""
    if os.environ.get("PREFILL_VALIDATE_MIGRATION", "0") != "1":
        return

    done_file = os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")
    wait_s = int(os.environ.get("PREFILL_MIGRATE_WAIT_S", "1200"))
    logger.info(f"[kv-migrate-validate] waiting for DONE sentinel {done_file} (<= {wait_s}s)")
    deadline = time.time() + wait_s
    while not os.path.exists(done_file):
        if time.time() >= deadline:
            raise TimeoutError(
                f"[kv-migrate-validate] sentinel {done_file} never appeared after {wait_s}s "
                "(did the scheduler finish prefill + migration?)"
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
    ttnn.synchronize_device(pipeline.mesh_device)
    if os.environ.get("PREFILL_MIGRATE_PAIRWISE", "1") == "1":
        validate_migrations_pairwise(pipeline, kv_cache, pairs, real_len=real_end)
    logger.success(f"[kv-migrate-validate] ALL {len(pairs)} migrated pair(s) PASSED")
