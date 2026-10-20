# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/index_score_decode`` (batched one-query-per-user indexer scores, a model-local generic op) vs the
torch golden sum_h relu(q_h . k_p) w_h: PCC and the top-512 selected-set overlap per user (the quantity the sparse
decode consumes) for W = 1280 and 262400 (the 1 M-position pool width) with B = 1 / 4 / 8 / 32; trace replay with
refreshed queries; the pooled-key streaming rate.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/custom_ops/test_index_score_decode.py -s
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.index_score_decode.op import IndexScoreDecode
from models.demos.glm53_flash.tests.utils import pcc

Hi, D, K = 32, 128, 512


def _inputs(B, W, seed):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(B, Hi, D, generator=g) * 0.5).bfloat16()
    k = (torch.randn(B, W, D, generator=g) * 0.5).bfloat16()
    w = (torch.randn(B, Hi, generator=g) * 0.2).bfloat16()
    return q, k, w


def _device(mesh_device, q, k, w, slots=None):
    B, W = k.shape[0], k.shape[1]
    slots = slots or B
    kk = torch.zeros(slots, 1, W, D)
    kk[:B, 0] = k.float()
    q_t = ttnn.from_torch(q.reshape(1, B, Hi, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    w_t = ttnn.from_torch(w.reshape(1, 1, B, Hi), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    k_t = ttnn.from_torch(kk.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    o_t = ttnn.zeros((B, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    return q_t, w_t, k_t, o_t


def _compare(ref, got, B):
    worst_pcc, worst_ov = 1.0, 1.0
    for b in range(B):
        worst_pcc = min(worst_pcc, pcc(ref[b], got[b]))
        top_r = set(torch.topk(ref[b], K).indices.tolist())
        top_g = set(torch.topk(got[b], K).indices.tolist())
        worst_ov = min(worst_ov, len(top_r & top_g) / K)
    return worst_pcc, worst_ov


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("W", [1280, 262400])
@pytest.mark.parametrize("B", [1, 4, 8, 32])
def test_index_score_decode_vs_golden(mesh_device, W, B):
    op = IndexScoreDecode(mesh_device)
    q, k, w = _inputs(B, W, W + B)
    ref = op.golden(q, k, w)
    q_t, w_t, k_t, o_t = _device(mesh_device, q, k, w, slots=B + 1)
    op(q_t, w_t, k_t, o_t)
    got = ttnn.to_torch(o_t).float().reshape(B, W)
    p, ov = _compare(ref, got, B)
    logger.info(
        f"index_score_decode W={W} B={B}: worst per-user pcc {p:.6f}, worst top-512 overlap {ov:.4f}, max|diff| {(ref - got).abs().max():.3f} (ref absmax {ref.abs().max():.2f})"
    )
    assert p > 0.999 and ov > 0.97
    for t in (q_t, w_t, k_t, o_t):
        ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("B", [1, 8])
def test_index_score_decode_bfp8_keys(mesh_device, B):
    """bfp8_b pooled keys (GLM53_POOL_DTYPE=bfloat8_b: half the bytes of the budget's floor term) vs the bf16 run on the
    same keys: per-user PCC of the scores and the top-512 selected-set overlap bf16 -> bfp8 (and both vs the golden)."""
    W = 262400
    op = IndexScoreDecode(mesh_device)
    q, k, w = _inputs(B, W, 17)
    ref = op.golden(q, k, w)
    q_t, w_t, k16, o16 = _device(mesh_device, q, k, w)
    op(q_t, w_t, k16, o16)
    got16 = ttnn.to_torch(o16).float().reshape(B, W)
    k8 = ttnn.typecast(k16, ttnn.bfloat8_b)
    o8 = ttnn.zeros((B, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    op(q_t, w_t, k8, o8)
    got8 = ttnn.to_torch(o8).float().reshape(B, W)
    p16, ov16 = _compare(ref, got16, B)
    p8, ov8 = _compare(ref, got8, B)
    p88, ov88 = _compare(got16, got8, B)
    logger.info(
        f"bfp8 keys B={B}: vs golden pcc {p8:.6f} / top-512 overlap {ov8:.4f} (bf16 keys: {p16:.6f} / {ov16:.4f}); "
        f"bfp8 vs bf16 run pcc {p88:.6f}, selected-set overlap {ov88:.4f}"
    )
    assert p8 > 0.995 and ov8 > 0.95
    for t in (q_t, w_t, k16, o16, k8, o8):
        ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 16 * 1024 * 1024, "l1_small_size": 32768}], indirect=True
)
def test_index_score_decode_trace_replay(mesh_device):
    W, B = 262400, 8
    op = IndexScoreDecode(mesh_device)
    q, k, w = _inputs(B, W, 9)
    q_t, w_t, k_t, o_t = _device(mesh_device, q, k, w)
    op(q_t, w_t, k_t, o_t)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op(q_t, w_t, k_t, o_t)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for step in range(3):
        q2, _, w2 = _inputs(B, 1, 100 + step)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(q2.reshape(1, B, Hi, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), q_t
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(w2.reshape(1, 1, B, Hi), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), w_t
        )
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        p, ov = _compare(op.golden(q2, k, w2), ttnn.to_torch(o_t).float().reshape(B, W), B)
        logger.info(f"trace replay {step}: worst pcc {p:.6f}, worst top-512 overlap {ov:.4f}")
        assert p > 0.999 and ov > 0.97
    ttnn.release_trace(mesh_device, tid)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_index_score_decode_timing(mesh_device):
    W = 262400
    op = IndexScoreDecode(mesh_device)
    for B in (1, 4, 8):
        q, k, w = _inputs(B, W, 3)
        q_t, w_t, k_t, o_t = _device(mesh_device, q, k, w)
        op(q_t, w_t, k_t, o_t)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        n = 5
        for _ in range(n):
            op(q_t, w_t, k_t, o_t)
        ttnn.synchronize_device(mesh_device)
        dt = (time.perf_counter() - t0) / n
        mb = B * W * D * 2 / 2**20
        logger.info(
            f"index_score_decode W={W} B={B}: {dt * 1e3:.3f} ms per call = {mb / dt / 1024:.0f} GB/s of pooled keys ({mb:.0f} MiB)"
        )
        for t in (q_t, w_t, k_t, o_t):
            ttnn.deallocate(t)
