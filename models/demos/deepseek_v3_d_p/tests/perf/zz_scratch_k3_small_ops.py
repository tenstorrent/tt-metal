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
    # rms_norm [320, 3584]: DRAM interleaved (as the block) vs block-sharded L1 (8 x 10: 1 row tile x 14 tiles a core)
    xr_ = torch.randn(S, 3584).bfloat16()
    g_ = dev(torch.randn(1, 1, 112, 32).bfloat16(), ttnn.ROW_MAJOR_LAYOUT)
    xt_ = dev(xr_, ttnn.TILE_LAYOUT)
    hi = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    ref_ = run(mesh_device, "rms_norm DRAM interleaved", lambda: ttnn.rms_norm(xt_, epsilon=1e-5, weight=g_, compute_kernel_config=hi))
    for gx, gy in ((8, 10), (4, 10), (7, 10)):
        bw = 112 // gx
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]), [32, bw * 32], ttnn.ShardOrientation.ROW_MAJOR))
        xs = run(mesh_device, f"to block-sharded {gx}x{gy}", lambda: ttnn.to_memory_config(xt_, mc))
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), subblock_w=max(d_ for d_ in (1, 2, 4, 7) if bw % d_ == 0 and d_ <= 4), block_h=1, block_w=bw, inplace=False)
        try:
            o_ = run(mesh_device, f"rms_norm block-sharded {gx}x{gy}", lambda: ttnn.rms_norm(xs, epsilon=1e-5, weight=g_, program_config=pc, memory_config=mc, compute_kernel_config=hi))
            a_, b_ = ttnn.to_torch(o_).float(), ttnn.to_torch(ref_).float()
            logger.info(f"SMALLOP rms check {gx}x{gy}: PCC {torch.corrcoef(torch.stack([a_.flatten(), b_.flatten()]))[0, 1].item():.6f}")
            run(mesh_device, f"sharded -> interleaved {gx}x{gy}", lambda: ttnn.sharded_to_interleaved(o_, ttnn.DRAM_MEMORY_CONFIG))
        except Exception as e:  # noqa: BLE001
            logger.info(f"SMALLOP rms_norm block-sharded {gx}x{gy} FAILED {str(e).splitlines()[0][:160]}")
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
