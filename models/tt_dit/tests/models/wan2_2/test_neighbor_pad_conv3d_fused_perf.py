# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device driver for the fused neighbor_pad_conv3d vs standalone (NP + conv3d) comparison.

This test only DISPATCHES both paths per shape — standalone (two ops: NeighborPadAsync + Conv3d,
both on the FULL grid) then fused (one NpConv3d op) — so a profiler can capture their device time.
The clean speedup TABLE is produced by the repo-root ``np_speedup_table.py``, which runs this test
under tracy and reports device-FW MIN: fused vs (NP + conv) with the NP/conv/sum broken out.

The wall-clock logged here is host-dispatch-bound and only indicative — it makes fusion look slower
while the real device win is hidden (wiki/NP_CONV3D_FUSED.md §4e). Use the table for the real number.

Coverage: every LTX VAE production shape on 2x4 (real NP topology) and 4x8mock (4x8 per-device conv
sizes on this 2x4 box — NP topology is still 2x4), plus a real 4x8 set (skipped unless a 32-chip mesh
is present). The standalone conv3d always runs on the full grid; the fused op reserves column-0 for
the NP fabric — that is the real deployed comparison.
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.vae_wan2_1 import WanCausalConv3d, WanConv2d
from ....parallel.config import ParallelFactor, VaeHWParallelConfig
from ....parallel.manager import CCLManager
from ....utils.conv3d import ConvDims, conv_pad_height, conv_pad_in_channels
from ....utils.tensor import typed_tensor_2dshard

NUM_MEASURED_DISPATCHES = 5

# Real 4x8 (32-device) shapes need a 32-chip mesh; skip on the 8-chip BH-LB so the coverage gap is
# visible in the report rather than silently dropped. 4x8mock shapes run here (they use a 2x4 mesh).
_SKIP_4X8 = pytest.mark.skip(reason="real 4x8 (32-device) shape — needs a 32-chip mesh, not 8-chip BH-LB")

# 4x8mock runs 4x8 per-device conv sizes on this 2x4 box. The 2x4 config keys miss the 4x8-tuned
# blockings (config is keyed by mesh) and fall back ~16x slower, so restore the real 4x8 blockings —
# the conv is the part the overlap scheme acts on, so this is a faithful proxy (NP topology still 2x4).
# Keyed by (C_in, C_out, H, W) of the 4x8mock shape.
_MOCK_4X8_BLK = {
    (512, 512, 34, 60): (64, 256, 1, 4, 8),  # ltx_s1_res 4x8 per-dev 17x15
    (512, 512, 68, 120): (64, 256, 1, 8, 4),  # ltx_s2_res 4x8 per-dev 34x30
    (256, 256, 68, 120): (64, 256, 1, 8, 4),  # ltx_s3_res 4x8 per-dev 34x30
    (256, 512, 68, 120): (64, 256, 1, 8, 4),  # ltx_s3_chg 4x8 per-dev 34x30
    (128, 128, 136, 240): (128, 64, 6, 4, 4),  # ltx_s4_res 4x8 per-dev 68x60 — force_spatial 6,4,4
    (128, 48, 136, 240): (128, 64, 6, 2, 16),  # ltx_s4_out 4x8 per-dev 68x60 — standalone-optimal
    (1024, 1024, 20, 32): (128, 64, 5, 4, 8),  # ltx ups_post_res 4x8 per-dev 10x8
    (1024, 128, 20, 32): (128, 64, 7, 8, 4),  # ltx ups_final 4x8 per-dev 10x8
    (128, 1024, 18, 32): (64, 128, 7, 8, 4),  # ltx_s0_conv_in 4x8 per-dev 9x8
    (1024, 4096, 18, 32): (128, 64, 5, 4, 8),  # ltx_s0_up 4x8 per-dev 9x8
    (512, 4096, 34, 60): (128, 64, 5, 4, 8),  # ltx_s1_up 4x8 per-dev 17x15
}


def _set_blk(model, ci, co, t, h, w):
    """Override the conv3d blocking (C_in, C_out, T, H, W block sizes)."""
    model.conv_config.C_in_block = ci
    model.conv_config.C_out_block = co
    model.conv_config.T_out_block = t
    model.conv_config.H_out_block = h
    model.conv_config.W_out_block = w


def _env_blk(name):
    """Parse a 'Cin,Cout,T,H,W' env override into a 5-int tuple, or None."""
    v = os.environ.get(name)
    return tuple(int(x) for x in v.split(",")) if v else None


def _build_model(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype, *, use_fused
):
    h_factor = tuple(mesh_device.shape)[h_axis]
    w_factor = tuple(mesh_device.shape)[w_axis]
    H_dev, W_dev = H // h_factor, W // w_factor
    kernel_tuple = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=h_factor, mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=w_factor, mesh_axis=w_axis),
    )

    if kernel_tuple[0] == 1:
        model = WanConv2d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=kernel_tuple,
            mesh_device=mesh_device,
            stride=1,
            padding=padding,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=ConvDims(T=T, H=H_dev, W=W_dev),
        )
    else:
        model = WanCausalConv3d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=kernel_size,
            mesh_device=mesh_device,
            stride=1,
            padding=padding,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=ConvDims(T=T, H=H_dev, W=W_dev),
            use_fused=use_fused,
        )

    # Random weights — this test checks device time, not correctness (see the PCC test for that).
    torch.manual_seed(0)
    weight = torch.randn(C_out, C_in, *kernel_tuple, dtype=torch.float32) * 0.01
    state = {"weight": weight, "bias": torch.zeros(C_out, dtype=torch.float32)}
    if kernel_tuple[0] == 1 and state["weight"].ndim == 5:
        state["weight"] = state["weight"].squeeze(2)
    model.load_torch_state_dict(state)

    if isinstance(model, WanCausalConv3d):
        # 4x8mock blocking restore; then optional dev sweep overrides (CONV3D_BLOCKING_SWEEP_RUNBOOK):
        #   NP_S4_BLK / NP_S4OUT_BLK pin a per-shape blocking on BOTH models; NP_BLK pins a fused-only
        #   blocking so the standalone baseline in the same run keeps its tuned entry.
        if (C_in, C_out, H, W) in _MOCK_4X8_BLK:
            _set_blk(model, *_MOCK_4X8_BLK[(C_in, C_out, H, W)])
        if (blk := _env_blk("NP_S4_BLK")) and (C_in, C_out, T) == (128, 128, 147):
            _set_blk(model, *blk)
        if (blk := _env_blk("NP_S4OUT_BLK")) and (C_in, C_out, T) == (128, 48, 147):
            _set_blk(model, *blk)
        if (blk := _env_blk("NP_BLK")) and use_fused:
            _set_blk(model, *blk)

    if use_fused and isinstance(model, WanCausalConv3d):
        # Bypass the MIN_T_FOR_FUSED hybrid-dispatch threshold so perf covers all shapes, incl. small T.
        if not model._use_fused and model._needs_halo:
            model._use_fused = True
        if os.environ.get("NP_FORCE_SPATIAL"):
            model.conv_config.force_spatial_parallel = True
        if os.environ.get("NP_HALO_LAST"):
            model.conv_config.halo_last = True

    # Standalone conv3d always runs on the FULL grid — the real deployed comparison against the fused op,
    # which reserves column-0 for the NP fabric (conv on ~102 of 110 cores). No grid reduction here.
    return model, h_factor, w_factor, parallel_config


def _build_input(mesh_device, B, C_in, T, H, W, h_axis, w_axis):
    torch.manual_seed(42)
    x = torch.randn(B, C_in, T, H, W, dtype=torch.float32).permute(0, 2, 3, 4, 1)
    x = conv_pad_in_channels(x)
    x, logical_h = conv_pad_height(x, tuple(mesh_device.shape)[h_axis])
    x = typed_tensor_2dshard(
        x, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}, dtype=ttnn.bfloat16
    )
    return x, logical_h


def _measure(model, input_tensor, logical_h, mesh_device, *, n):
    """Wall-clock ms/dispatch (host-dispatch-bound; indicative only — the device truth is the table)."""
    _ = model(input_tensor, logical_h=logical_h)  # warmup: pay JIT/cache once
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter_ns()
    for _ in range(n):
        out = model(input_tensor, logical_h=logical_h)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter_ns() - t0) / n / 1e6
    try:
        ttnn.deallocate(out)
    except Exception:
        pass
    return elapsed_ms


def _param(*row):
    """Build a parametrize entry; id is the trailing shape_id, real 4x8 (mesh (4,8)) is auto-skipped."""
    return pytest.param(*row, id=row[-1], marks=(_SKIP_4X8,) if row[8] == (4, 8) else ())


# (B, C_in, C_out, T, H, W, kernel_size, padding, mesh_device, h_axis, w_axis, num_links, shape_id)
# num_links=2 everywhere: 8 fabric cores (num_links*2 H + pad2_num_links*2 W) balance the NP+border
# work against the ~102 bulk conv cores. 2x4 H,W are chosen so the per-device key hits the tuned
# blocking; real 4x8 H,W = 4x8mock per-device sizes * (4,8) so they shard evenly on a 32-chip mesh.
_PERF_PARAMS = [
    # --- LTX VAE 2x4 production (real NP topology) ---
    _param(1, 128, 1024, 21, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_conv_in_2x4"),  # per-dev 17x15
    _param(1, 1024, 4096, 21, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_up_2x4"),  # per-dev 17x15
    _param(1, 512, 4096, 39, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s1_up_2x4"),  # per-dev 34x30
    _param(1, 1024, 1024, 21, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_res_2x4"),  # per-dev 17x15
    _param(1, 512, 512, 39, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s1_res_2x4"),  # per-dev 34x30
    _param(1, 512, 512, 75, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s2_res_2x4"),  # per-dev 68x60
    _param(1, 256, 256, 147, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s3_res_2x4"),  # per-dev 68x60
    _param(1, 256, 512, 147, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s3_chg_2x4"),  # per-dev 68x60
    _param(1, 128, 128, 147, 272, 480, 3, 1, (2, 4), 0, 1, 2, "ltx_s4_res_2x4"),  # per-dev 136x120
    _param(1, 128, 48, 147, 272, 480, 3, 1, (2, 4), 0, 1, 2, "ltx_s4_out_2x4"),  # per-dev 136x120
    # --- 4x8mock: 4x8 per-device conv sizes on this 2x4 box (NP topology still 2x4) ---
    _param(1, 128, 1024, 21, 18, 32, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_conv_in_4x8mock"),  # per-dev 9x8
    _param(1, 1024, 4096, 21, 18, 32, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_up_4x8mock"),  # per-dev 9x8
    _param(1, 512, 4096, 39, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s1_up_4x8mock"),  # per-dev 17x15
    _param(1, 512, 512, 39, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s1_res_4x8mock"),  # per-dev 17x15
    _param(1, 512, 512, 75, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s2_res_4x8mock"),  # per-dev 34x30
    _param(1, 256, 256, 147, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s3_res_4x8mock"),  # per-dev 34x30
    _param(1, 256, 512, 147, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s3_chg_4x8mock"),  # per-dev 34x30
    _param(1, 128, 128, 147, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s4_res_4x8mock"),  # per-dev 68x60
    _param(1, 128, 48, 147, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s4_out_4x8mock"),  # per-dev 68x60
    _param(1, 1024, 1024, 21, 20, 32, 3, 1, (2, 4), 0, 1, 2, "ltx_ups_post_res_4x8mock"),  # per-dev 10x8
    _param(1, 1024, 128, 21, 20, 32, 3, 1, (2, 4), 0, 1, 2, "ltx_ups_final_4x8mock"),  # per-dev 10x8
    # --- Real 4x8 (32-chip mesh): same per-device sizes on a real (4,8) mesh → real 4x8 NP topology ---
    _param(1, 128, 1024, 21, 36, 64, 3, 1, (4, 8), 0, 1, 2, "ltx_s0_conv_in_4x8"),  # per-dev 9x8
    _param(1, 1024, 4096, 21, 36, 64, 3, 1, (4, 8), 0, 1, 2, "ltx_s0_up_4x8"),  # per-dev 9x8
    _param(1, 512, 4096, 39, 68, 120, 3, 1, (4, 8), 0, 1, 2, "ltx_s1_up_4x8"),  # per-dev 17x15
    _param(1, 512, 512, 39, 68, 120, 3, 1, (4, 8), 0, 1, 2, "ltx_s1_res_4x8"),  # per-dev 17x15
    _param(1, 512, 512, 75, 136, 240, 3, 1, (4, 8), 0, 1, 2, "ltx_s2_res_4x8"),  # per-dev 34x30
    _param(1, 256, 256, 147, 136, 240, 3, 1, (4, 8), 0, 1, 2, "ltx_s3_res_4x8"),  # per-dev 34x30
    _param(1, 256, 512, 147, 136, 240, 3, 1, (4, 8), 0, 1, 2, "ltx_s3_chg_4x8"),  # per-dev 34x30
    _param(1, 128, 128, 147, 272, 480, 3, 1, (4, 8), 0, 1, 2, "ltx_s4_res_4x8"),  # per-dev 68x60
    _param(1, 128, 48, 147, 272, 480, 3, 1, (4, 8), 0, 1, 2, "ltx_s4_out_4x8"),  # per-dev 68x60
    _param(1, 1024, 1024, 21, 40, 64, 3, 1, (4, 8), 0, 1, 2, "ltx_ups_post_res_4x8"),  # per-dev 10x8
    _param(1, 1024, 128, 21, 40, 64, 3, 1, (4, 8), 0, 1, 2, "ltx_ups_final_4x8"),  # per-dev 10x8
]


@pytest.mark.parametrize(
    "B, C_in, C_out, T, H, W, kernel_size, padding, mesh_device, h_axis, w_axis, num_links, shape_id",
    _PERF_PARAMS,
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.timeout(600)
def test_fused_vs_standalone_perf(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype, shape_id
):
    """Dispatch standalone (full-grid NP + conv3d) then fused, back-to-back; log wall-clock (indicative).

    Log-only — no assertions. Read the real device-FW speedup from ``np_speedup_table.py``, which runs
    this under tracy. Wall-clock here is host-dispatch-bound and makes fusion look slower (§4e).
    """
    sa_model, *_ = _build_model(
        mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype, use_fused=False
    )
    sa_input, sa_logical_h = _build_input(mesh_device, B, C_in, T, H, W, h_axis, w_axis)
    sa_ms = _measure(sa_model, sa_input, sa_logical_h, mesh_device, n=NUM_MEASURED_DISPATCHES)
    ttnn.deallocate(sa_input)

    f_model, *_ = _build_model(
        mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype, use_fused=True
    )
    f_input, f_logical_h = _build_input(mesh_device, B, C_in, T, H, W, h_axis, w_axis)
    f_ms = _measure(f_model, f_input, f_logical_h, mesh_device, n=NUM_MEASURED_DISPATCHES)
    ttnn.deallocate(f_input)

    ratio = f_ms / sa_ms if sa_ms > 0 else float("inf")
    logger.info(
        f"PERF shape={shape_id}  fused={f_ms:.3f}ms  standalone={sa_ms:.3f}ms  ratio={ratio:.3f}  "
        f"(WALL — host-bound; device truth in np_speedup_table.py; n={NUM_MEASURED_DISPATCHES})"
    )
