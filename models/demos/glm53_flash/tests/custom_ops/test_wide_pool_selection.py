# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The prefill selection chain (``custom_ops/index_score_decode`` scorer -> ``custom_ops/topk_rows`` bounded top-512)
at the pool widths of the 512 K and 1 M contexts, with PLANTED keys: query t is the pooled key of pool p_t (scaled),
so pool p_t must score highest and be selected. Pools on both sides of 65536 (the 256 K / 512 K ladder boundary:
the needle is found at 256 K and missed at 512 K and 1 M, see doc/mesh/README.md).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/custom_ops/test_wide_pool_selection.py -s
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.index_score_decode.op import IndexScoreDecode
from models.demos.glm53_flash.custom_ops.topk_rows.op import TopkRows

Hi, D, K = 32, 128, 512


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("W", [131072, 262144])
def test_wide_pool_selection_planted(mesh_device, W):
    planted = [
        p for p in (100, 4095, 32768, 65535, 65536, 65537, 70000, 100000, 131071, 131072, 200000, 262143) if p < W
    ]
    T = 32
    g = torch.Generator().manual_seed(7)
    keys = (torch.randn(1, 1, W, D, generator=g) * 0.5).bfloat16()
    q = (torch.randn(T, Hi, D, generator=g) * 0.5).bfloat16()
    for t, p in enumerate(planted):
        q[t] = keys[0, 0, p].unsqueeze(0).expand(Hi, D)  # every head = the planted key: the dot product peaks at p
    w = torch.full((T, Hi), 0.2).bfloat16()  # positive head weights: the score is a positive sum over heads
    scorer = IndexScoreDecode(mesh_device)
    topk = TopkRows(mesh_device, k=K)
    pools = ttnn.from_torch(keys, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    w_t = ttnn.from_torch(w.reshape(1, 1, T, Hi), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    q_r = ttnn.from_torch(q.reshape(1, T, Hi, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    bounds = torch.full((T, 1), W, dtype=torch.int32)
    v_t = ttnn.from_torch(bounds, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    scores = ttnn.zeros((T, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    scorer(q_r, w_t, pools, scores, valid_rows=v_t, slot=0, cols=W)
    got = ttnn.to_torch(scores).float().reshape(T, W)
    # 1) the scorer: the planted pool is the row's maximum
    bad_score = []
    for t, p in enumerate(planted):
        am = int(torch.argmax(got[t]))
        if am != p:
            bad_score.append((t, p, am, float(got[t, p]), float(got[t, am])))
    # 2) the top-k: the planted pool is among the 512 selected
    out = ttnn.zeros((T, 1, 1, K), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    topk(scores, v_t, out)
    ids = ttnn.to_torch(out).reshape(T, K).to(torch.int64) & 0xFFFFFFFF
    bad_topk = [(t, p) for t, p in enumerate(planted) if p not in set(ids[t].tolist())]
    # 3) the golden top-512 of the device scores vs the device top-k (id sets)
    overlap = min(len(set(torch.topk(got[t], K).indices.tolist()) & set(ids[t].tolist())) / K for t in range(T))
    logger.info(
        f"W={W}: planted pools {planted}; scorer argmax misses {bad_score}; top-k misses {bad_topk}; "
        f"worst top-512 overlap vs torch over the device scores {overlap:.4f}"
    )
    for t in (pools, w_t, q_r, v_t, scores, out):
        ttnn.deallocate(t)
    assert not bad_score, bad_score
    assert not bad_topk, bad_topk
    # bf16 ties near the 512th score of a random 262 K row move a few ids (0.967 measured at W=262144): the planted
    # maxima are the exact check, the overlap a sanity bound
    assert overlap > 0.95, overlap
