# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the K3 block's small ops at 320 rows per chip, device time per variant (layout conversions, top-k)."""
import pytest
import torch
from loguru import logger

import ttnn
from ttnn.bringup.combine_rf.layout_ops import layout_op_prepare

_KEY = "DEVICE KERNEL DURATION [ns]"


def _times(mesh_device):
    ttnn.ReadDeviceProfiler(mesh_device)
    rows = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        rows.append([float(p.program_analyses_results[_KEY].duration) / 1e3 for p in programs if _KEY in (p.program_analyses_results or {})])
    return rows


def run(mesh_device, name, f, n=3):
    _times(mesh_device)
    out = None
    best = None
    for _ in range(n):
        out = f()
        ttnn.synchronize_device(mesh_device)
        rows = _times(mesh_device)
        tot = max(sum(r) for r in rows)
        progs = [round(max(r[i] for r in rows), 1) for i in range(min(len(r) for r in rows))]
        best = (tot, progs) if best is None or tot < best[0] else best
    logger.info(f"SMALLOP {name:44s} {best[0]:8.1f} us  {best[1]}")
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_small_ops(mesh_device):
    S = 320
    dev = lambda t, lay, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
    for W in (3584, 7168):
        x = torch.randn(S, W).bfloat16()
        xt, xr = dev(x, ttnn.TILE_LAYOUT), dev(x, ttnn.ROW_MAJOR_LAYOUT)
        run(mesh_device, f"to_layout tile->RM [{S},{W}]", lambda: ttnn.to_layout(xt, ttnn.ROW_MAJOR_LAYOUT))
        run(mesh_device, f"untilize [{S},{W}]", lambda: ttnn.untilize(xt))
        run(mesh_device, f"to_layout RM->tile [{S},{W}]", lambda: ttnn.to_layout(xr, ttnn.TILE_LAYOUT))
        run(mesh_device, f"tilize [{S},{W}]", lambda: ttnn.tilize(xr))
    til, until = layout_op_prepare(mesh_device, "tilize"), layout_op_prepare(mesh_device, "untilize")
    for W in (3584, 7168):
        x = torch.randn(S, W).bfloat16()
        xt, xr = dev(x, ttnn.TILE_LAYOUT), dev(x, ttnn.ROW_MAJOR_LAYOUT)
        a = run(mesh_device, f"layout_ops untilize [{S},{W}]", lambda: until(xt))
        b = run(mesh_device, f"layout_ops tilize [{S},{W}]", lambda: til(xr))
        logger.info(f"SMALLOP check W {W}: untilize exact {torch.equal(ttnn.to_torch(a), x)}, tilize exact {torch.equal(ttnn.to_torch(b), x)}")
    until2 = layout_op_prepare(mesh_device, "untilize", pages_per_row=2)
    x = torch.randn(S, 7168).bfloat16()
    a = run(mesh_device, "layout_ops untilize P2 [320,7168]", lambda: until2(dev(x, ttnn.TILE_LAYOUT)))
    logger.info(f"SMALLOP check P2: {tuple(a.shape)} exact {torch.equal(ttnn.to_torch(a).reshape(S, 7168), x)}")
    E = 448
    logits = dev(torch.randn(S, E).bfloat16(), ttnn.TILE_LAYOUT)
    bias = dev(torch.randn(S, E).bfloat16() * 0.01, ttnn.TILE_LAYOUT)
    run(mesh_device, "moe_grouped_topk", lambda: ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        logits, bias, n_groups=1, summed_experts_per_group=1, topk_groups=1, n_activated_experts=16, route_scale=1.0,
        stable_sort=True, epsilon=1e-20, score_func="sigmoid", padding_config=None, weights_layout=ttnn.ROW_MAJOR_LAYOUT))
    run(mesh_device, "moe_grouped_topk unstable", lambda: ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        logits, bias, n_groups=1, summed_experts_per_group=1, topk_groups=1, n_activated_experts=16, route_scale=1.0,
        stable_sort=False, epsilon=1e-20, score_func="sigmoid", padding_config=None, weights_layout=ttnn.ROW_MAJOR_LAYOUT))
    run(mesh_device, "sigmoid", lambda: ttnn.sigmoid(logits))
    import os
    from ttnn.bringup.combine_rf.router_topk import router_topk_prepare
    l32 = dev(torch.randn(S, E) * 2, ttnn.TILE_LAYOUT, ttnn.float32)
    s32 = ttnn.sigmoid(l32)
    b32 = ttnn.add(s32, dev(torch.randn(1, E) * 0.01, ttnn.TILE_LAYOUT, ttnn.float32))
    for pr in ("", "noweights"):
        os.environ["TOPK_PROBE"] = pr
        rt = router_topk_prepare(mesh_device, S=S, E=E, K=16)
        run(mesh_device, f"router_topk {pr or 'full'}", lambda: rt(s32, b32))
    run(mesh_device, "topk k=16 (ttnn.topk)", lambda: ttnn.topk(logits, 16, dim=-1, largest=True, sorted=False))
