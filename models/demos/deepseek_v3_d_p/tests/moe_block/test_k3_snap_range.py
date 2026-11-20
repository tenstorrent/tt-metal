# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic (not CI): does the combine's int32 snap saturate on Kimi-K3's real activations?

With the snap (q = 2^-14) each routed product is an int32 count of q and the packer adds the counts in L1, saturating
at (2^31 - 1) q ~ 131072: a token whose partial sums cross that would make the sum order-dependent. For every MoE layer
of the L24 model and every chunk of the L24 test's eleven (the golden trace's real MoE inputs), the block runs with the
snap and with it off (fp32 sums): without saturation the two outputs agree to the snap's rounding (K/2 q per element,
plus fp32 order noise); a saturated sum would be off by the overflow. Logs the per-layer output range and the gap."""

import gc
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.moe_block.test_moe_block_models import (
    MODELS,
    BlockCase,
    _mesh_dims,
    _params,
    _to_host,
)

SNAP = 2.0**-14
LAYERS = int(os.environ.get("SNAP_RANGE_LAYERS", "24"))
CHUNKS = int(os.environ.get("SNAP_RANGE_CHUNKS", "11"))
CHUNK = 5120
CASE = BlockCase("kimi_k3", "8x4", CHUNK, weights="weka", layer=1)


@pytest.mark.parametrize("case, mesh_device, device_params", _params([CASE]), indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(7200)
def test_k3_snap_range(case, mesh_device, device_params):
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_trace
    from models.demos.deepseek_v3_d_p.tt.moe_block.moe_block_ffn import MoEBlockFfnOptions, TtMoEBlockFfn

    m = MODELS["kimi_k3"]
    H = m.cfg.EMB_SIZE
    R, C = tuple(mesh_device.shape)
    trace = resolve_trace(TRACE_100K)
    assert trace is not None, "no K3 golden trace"
    K = m.cfg.NUM_EXPERTS_PER_TOKEN
    worst_excess = 0.0
    lines = []
    for L in range(m.cfg.NUM_DENSE_LAYERS, LAYERS):  # (the first layers are dense: no routed experts)
        if not trace.has("moe_io", f"moe_input_layer_{L}"):
            continue
        ys = {}
        for name, q in (("snap", SNAP), ("fp32", 0.0)):
            ffn = TtMoEBlockFfn(mesh_device, m.cfg, layer_idx=L, options=MoEBlockFfnOptions(combine_snap=q))
            outs = []
            for c in range(CHUNKS):
                x = trace.rows("moe_io", f"moe_input_layer_{L}", c * CHUNK, (c + 1) * CHUNK).to(torch.bfloat16)
                x_tt = ttnn.from_torch(
                    x.reshape(1, 1, CHUNK, H),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=_mesh_dims(case)),
                )
                y = ffn(x_tt)
                outs.append(_to_host(y, mesh_device, CHUNK, H, case))
                ttnn.deallocate(y)
                ttnn.deallocate(x_tt)
            ys[name] = torch.cat(outs)
            del ffn
            gc.collect()
        ys_, yf = ys["snap"], ys["fp32"]
        gap = (ys_ - yf).abs()
        g = float(gap.max())
        # allowed per element: the snap's rounding (K terms, q/2 each) plus bf16 rounding of the outputs - 4 ulp of the
        # larger magnitude (L1-L4 measured 2^-11 past K q/2 + 2 ulp, at bf16 binade edges). A saturated sum is off by its overflow (~1e5), orders of magnitude beyond this.
        ulp = torch.exp2(torch.floor(torch.log2((yf.abs() + K * SNAP).clamp_min(2.0**-30))) - 7)
        tol = K * SNAP / 2 + 4 * ulp
        excess = gap - tol
        worst_excess = max(worst_excess, float(excess.max()))
        line = (
            f"L{L}: max|y| {float(yf.abs().max()):.3f} (p99.99 {float(yf.abs().flatten()[::61].quantile(0.9999)):.3f}, "
            f"headroom x{131072 / max(float(yf.abs().max()), 1e-9):.0f} to 2^17), max|snap - fp32| {g:.6f} "
            f"at token {int(gap.max(dim=1).values.argmax())}, elements over the rounding bound: {int((excess > 0).sum())}"
        )
        logger.info(f"SNAP RANGE {line}")
        lines.append(line)
    logger.info("SNAP RANGE summary:\n" + "\n".join(lines))
    assert lines, "no MoE layers in the trace"
    assert worst_excess <= 0, f"snap vs fp32: {worst_excess} over the rounding bound somewhere - saturation?"
