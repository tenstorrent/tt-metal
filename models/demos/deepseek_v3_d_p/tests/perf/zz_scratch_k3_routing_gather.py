# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch probe: which op chain turns the router's ids / weights [S, K] per chip into the planner's single-page
[1, N * S * K] uint16 / bf16 rows (replicated)."""
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params

S, K, E = 320, 16, 448
_PARAMS = [pytest.param((2, 4), fabric2d_device_params(fabric_payload_size=7232, l1_small_size=0), id="2x4")]


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_routing_gather(mesh_device, device_params):
    N = 8
    x = torch.randn(N * S, 7168).to(torch.bfloat16)
    g = torch.randn(7168, E).to(torch.bfloat16)
    sh = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=sh)
    gt = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    bias = ttnn.from_torch(torch.zeros(S, E), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    logits = ttnn.linear(xt, gt, dtype=ttnn.bfloat16)
    wts, ids = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        logits, bias, n_groups=1, summed_experts_per_group=1, topk_groups=1, n_activated_experts=K,
        route_scale=1.0, stable_sort=True, epsilon=1e-20, score_func="sigmoid", padding_config=None,
        weights_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    logger.info(f"PROBE ids {ids.dtype} {ids.layout} {ids.shape} {ids.memory_config().buffer_type}; wts {wts.dtype} {wts.layout} {wts.shape}")
    ref_ids = torch.cat([ttnn.to_torch(t).to(torch.int64).reshape(S, K) for t in ttnn.get_device_tensors(ids)]).reshape(-1) & 0xFFFF

    def chain(name, f):
        try:
            out = f()
            got = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
            ok = tuple(out.shape) == (1, N * S * K) and torch.equal(got.to(torch.int64).reshape(-1) & 0xFFFF, ref_ids)
            logger.info(f"PROBE {name}: {out.dtype} {out.layout} {tuple(out.shape)} pages? equal={ok}")
        except Exception as e:  # noqa: BLE001
            logger.info(f"PROBE {name}: FAILED {str(e).splitlines()[0][:200]}")

    def ag_last(t):
        a = ttnn.all_gather(t, dim=1, cluster_axis=1)
        return ttnn.all_gather(a, dim=1, cluster_axis=0)

    def c1():  # typecast u32 (tile) -> RM -> reshape [1, S K] -> typecast u16 -> gather last dim
        t = ids if ids.dtype == ttnn.uint32 else ttnn.typecast(ids, ttnn.uint32)
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) if t.layout != ttnn.ROW_MAJOR_LAYOUT else t
        t = ttnn.reshape(t, [1, S * K])
        t = ttnn.typecast(t, ttnn.uint16)
        return ag_last(t)

    def c2():  # u16 tile -> reshape in tile -> RM -> gather
        t = ids if ids.dtype == ttnn.uint16 else ttnn.typecast(ids, ttnn.uint16)
        t = ttnn.reshape(t, [1, S * K])
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        return ag_last(t)

    def c3():  # u32 RM reshape, gather in u32, then typecast u16
        t = ids if ids.dtype == ttnn.uint32 else ttnn.typecast(ids, ttnn.uint32)
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) if t.layout != ttnn.ROW_MAJOR_LAYOUT else t
        t = ttnn.reshape(t, [1, S * K])
        return ttnn.typecast(ag_last(t), ttnn.uint16)

    def c4():  # bf16 weights: RM reshape and gather
        t = wts if wts.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(wts, ttnn.ROW_MAJOR_LAYOUT)
        t = ttnn.typecast(t, ttnn.bfloat16) if t.dtype != ttnn.bfloat16 else t
        out = ag_last(ttnn.reshape(t, [1, S * K]))
        logger.info(f"PROBE w: {out.dtype} {out.layout} {tuple(out.shape)}")
        return ids

    for n, f in (("c1", c1), ("c2", c2), ("c3", c3)):
        chain(n, f)
    try:
        c4()
    except Exception as e:  # noqa: BLE001
        logger.info(f"PROBE w FAILED {str(e).splitlines()[0][:200]}")
