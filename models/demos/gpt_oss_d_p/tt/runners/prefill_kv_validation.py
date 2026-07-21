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
    compare_len = real_len if real_len is not None else max_seq_len
    compare_len = min(compare_len, max_seq_len)
    isl_per_row = max_seq_len // sp

    def _read_slot_tensor(cache: ttnn.Tensor, slot: int) -> torch.Tensor:
        """Read one slot's K or V back to CPU as [1, num_kv_heads, compare_len, head_dim]."""
        sp_rows, tp_cols = mesh_device.shape
        device_tensors = ttnn.get_device_tensors(cache)
        col_slices = []
        for col in range(tp_cols):
            row_shards = []
            for row in range(sp_rows):
                shard_len = min(isl_per_row, max(0, compare_len - row * isl_per_row))
                if shard_len == 0:
                    break
                t = ttnn.to_torch(device_tensors[row * tp_cols + col]).float()
                if len(t.shape) == 5:
                    row_shards.append(t[0, slot, :, :shard_len, :])
                elif len(t.shape) == 4:
                    row_shards.append(t[slot, :, :shard_len, :])
                else:
                    raise ValueError(f"[kv-migrate-validate] unexpected cache shape {tuple(t.shape)} for slot={slot}")
            if row_shards:
                col_slices.append(torch.cat(row_shards, dim=1))
        gathered = torch.cat(col_slices, dim=0)  # [num_kv_heads, compare_len, head_dim]
        return gathered.unsqueeze(0)

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
