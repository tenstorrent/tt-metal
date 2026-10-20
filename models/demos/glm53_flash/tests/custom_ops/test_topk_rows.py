# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/topk_rows`` (per-row-bounded top-512, a model-local generic op) vs torch.topk over each row's own
prefix: ids inside the row's bound and the score multiset equal (bf16 ties may reorder ids), for W = 1056 (fused
body) and 262 176 (classic body, the 1 M-position pool width) with B = 1 / 4 / 8 / 32 rows; trace capture + replay
with refreshed bounds and scores; timing vs the shared single-bound op.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/custom_ops/test_topk_rows.py -s
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.topk_rows.op import TopkRows

K = 512


def _check(scores: torch.Tensor, valid: torch.Tensor, ids: torch.Tensor) -> tuple[bool, int]:
    """(ok, common ids) of device ids [B,1,1,K] vs the golden sets: in range and equal score multisets."""
    ref = TopkRows.golden(scores, valid, K)
    ok, common = True, 0
    for b in range(scores.shape[0]):
        n = int(valid[b])
        got = ids[b, 0, 0].tolist()
        exp = ref[b, 0, 0].tolist()
        inr = all(0 <= i < n for i in got)
        s_got = sorted(scores[b, 0, 0, got].float().tolist())
        s_exp = sorted(scores[b, 0, 0, exp].float().tolist())
        ok &= inr and s_got == s_exp
        common += len(set(got) & set(exp))
    return ok, common


def _tensors(device, scores, valid):
    s = ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    v = ttnn.from_torch(valid.reshape(-1, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    o = ttnn.zeros((scores.shape[0], 1, 1, K), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return s, v, o


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("W", [1056, 262176])
@pytest.mark.parametrize("B", [1, 4, 8, 32])
def test_topk_rows_vs_golden(mesh_device, W, B):
    op = TopkRows(mesh_device)
    g = torch.Generator().manual_seed(W + B)
    scores = torch.randn(B, 1, 1, W, generator=g).bfloat16()
    valid = torch.randint(K, W + 1, (B,), generator=g)
    valid[0] = W
    valid[-1] = K
    s_t, v_t, o_t = _tensors(mesh_device, scores, valid)
    op(s_t, v_t, o_t)
    ids = ttnn.to_torch(o_t).long()
    ok, common = _check(scores, valid, ids)
    logger.info(
        f"topk_rows W={W} B={B}: per-row bounds respected and score multisets equal: {ok}; common ids {common}/{B * K}"
    )
    assert ok
    # a second call on new scores / bounds reuses the program
    scores2 = torch.randn(B, 1, 1, W, generator=g).bfloat16()
    valid2 = torch.randint(K, W + 1, (B,), generator=g)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(scores2, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), s_t)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(valid2.reshape(-1, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), v_t
    )
    op(s_t, v_t, o_t)
    ok2, _ = _check(scores2, valid2, ttnn.to_torch(o_t).long())
    assert ok2
    for t in (s_t, v_t, o_t):
        ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 16 * 1024 * 1024, "l1_small_size": 32768}], indirect=True
)
def test_topk_rows_trace_replay(mesh_device):
    W, B = 262176, 8
    op = TopkRows(mesh_device)
    g = torch.Generator().manual_seed(77)
    scores = torch.randn(B, 1, 1, W, generator=g).bfloat16()
    valid = torch.randint(K, W + 1, (B,), generator=g)
    s_t, v_t, o_t = _tensors(mesh_device, scores, valid)
    op(s_t, v_t, o_t)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op(s_t, v_t, o_t)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for step in range(3):
        scores = torch.randn(B, 1, 1, W, generator=g).bfloat16()
        valid = torch.randint(K, W + 1, (B,), generator=g)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), s_t)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(valid.reshape(-1, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), v_t
        )
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        ok, common = _check(scores, valid, ttnn.to_torch(o_t).long())
        logger.info(f"trace replay {step}: ok {ok}, common ids {common}/{B * K}")
        assert ok
    ttnn.release_trace(mesh_device, tid)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_topk_rows_timing(mesh_device):
    W = 262176
    op = TopkRows(mesh_device)
    g = torch.Generator().manual_seed(3)
    for B in (1, 4, 8):
        scores = torch.randn(B, 1, 1, W, generator=g).bfloat16()
        valid = torch.full((B,), W, dtype=torch.int64)
        s_t, v_t, o_t = _tensors(mesh_device, scores, valid)
        one = ttnn.from_torch(
            torch.tensor([W], dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
        )

        def bench(fn, n=5):
            fn()
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            ttnn.synchronize_device(mesh_device)
            return (time.perf_counter() - t0) / n * 1e3

        t_rows = bench(lambda: op(s_t, v_t, o_t))
        t_shared = bench(lambda: ttnn.experimental.topk_large_indices(s_t, k=K, valid_length_tensor=one))
        t_per_user = sum(
            bench(
                lambda i=i: ttnn.experimental.topk_large_indices(
                    ttnn.slice(s_t, [i, 0, 0, 0], [i + 1, 1, 1, W]), k=K, valid_length_tensor=one
                )
            )
            for i in range(B)
        )
        logger.info(
            f"topk W={W} B={B}: topk_rows (per-row bounds) {t_rows:.3f} ms; shared op one call {t_shared:.3f} ms; shared op per user {t_per_user:.3f} ms"
        )
        for t in (s_t, v_t, o_t, one):
            ttnn.deallocate(t)
