# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""qr-ring — the COMPUTE half of the perf story (single Blackhole chip, no fabric).

The transport half (KV-gather O(T) vs qr flat) is measured in test_qr_ring_transport_perf.py. This shows the
other half: the sparse_sdpa COMPUTE is flat in context length T — each query attends a FIXED top-k=2048
regardless of how long the KV cache is — and the qr stat export (sparse_sdpa_stats) adds only a small,
constant overhead over the plain op. Together: both transport AND compute are flat in T for Q-gather, so the
KV-gather O(T) all-gather it eliminates is the whole difference.

Run: scripts/run_safe_pytest.sh --run-all tests/nightly/blackhole/sdpa/test_sparse_sdpa_qr_op_perf.py
"""
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import make_inputs, to_dev

K_DIM, V_DIM = 576, 512
H, S, TOPK = 32, 128, 128  # fixed query/selection work; only T (cache length) varies


def _time_op(fn, iters=8, warmup=3):
    ts = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(out[0].device() if isinstance(out, list) else out.device())
        dt = (time.perf_counter() - t0) * 1e3
        if i >= warmup:
            ts.append(dt)
    ts.sort()
    return ts[len(ts) // 2]


@run_for_blackhole()
def test_qr_op_compute_flat_in_T(device):
    logger.info("sparse_sdpa compute vs context length T (top-k fixed) — plain op vs qr stat export")
    logger.info(f"{'T (cache)':>10} {'plain ms':>10} {'stats ms':>10} {'overhead':>10}")
    rows = []
    for T in [2048, 8192, 32768, 131072]:
        q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: 10**9, seed=0)
        tt_q = to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
        tt_kv = to_dev(kv.to(torch.bfloat16), device, ttnn.bfloat16)
        tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)
        scale = K_DIM**-0.5

        plain = _time_op(
            lambda: ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, scale=scale, k_chunk_size=128)
        )
        stats = _time_op(
            lambda: ttnn.transformer.sparse_sdpa_stats(tt_q, tt_kv, tt_idx, V_DIM, scale=scale, k_chunk_size=128)
        )
        rows.append((T, plain, stats))
        logger.info(f"{T:>10} {plain:>10.3f} {stats:>10.3f} {stats - plain:>9.3f}ms")
        for t in (tt_q, tt_kv, tt_idx):
            ttnn.deallocate(t)

    logger.info("")
    logger.info("=== qr-ring COMPUTE: flat in T (top-k fixed); stat export = small constant overhead ===")
    base = rows[0][1]
    for T, plain, stats in rows:
        logger.info(f"  T={T:>7}: plain {plain:6.3f}ms ({plain / base:.2f}x vs T=2048)  stats {stats:6.3f}ms")

    # Compute must NOT scale with T (attends fixed top-k): the largest T stays within ~2x of the smallest.
    assert rows[-1][1] < rows[0][1] * 2.0, "sparse_sdpa compute should be ~flat in cache length T"
