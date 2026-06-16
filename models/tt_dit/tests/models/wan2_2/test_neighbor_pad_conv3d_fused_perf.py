# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Log-only perf comparison: fused neighbor_pad_conv3d vs standalone NP + conv3d.

For each production shape, runs 1 warmup + N measured dispatches of the fused
path back-to-back, then N of the standalone path, and logs the ratio:

    PERF shape=<id>  fused=<f.ff>ms  standalone=<s.ss>ms  ratio=<f/s>  (lower is better)

No hard assertion — visible regressions are surfaced in CI logs but the test
PASSes either way. Promote to a hard assert once the numbers stabilize.
"""

from __future__ import annotations

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


def _build_model(
    mesh_device,
    B,
    C_in,
    C_out,
    T,
    H,
    W,
    kernel_size,
    padding,
    h_axis,
    w_axis,
    num_links,
    dtype,
    *,
    use_fused,
):
    h_factor = tuple(mesh_device.shape)[h_axis]
    w_factor = tuple(mesh_device.shape)[w_axis]
    H_dev = H // h_factor
    W_dev = W // w_factor
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

    # Random weights — we're not checking correctness here.
    torch.manual_seed(0)
    weight = torch.randn(C_out, C_in, *kernel_tuple, dtype=torch.float32) * 0.01
    bias = torch.zeros(C_out, dtype=torch.float32)
    state = {"weight": weight, "bias": bias}
    if kernel_tuple[0] == 1 and "weight" in state and state["weight"].ndim == 5:
        state["weight"] = state["weight"].squeeze(2)
    model.load_torch_state_dict(state)
    # 4x8 mock: these (C_in, H, W) globals reproduce the 4x8 per-device sizes on this 2x4 mesh, but the
    # 2x4 config keys miss the 4x8-tuned blockings (config keys by mesh) and fall back ~16x slower.
    # Restore the real 4x8 blockings so the conv is faithfully mocked. NP topology still differs (2x4
    # vs 4x8) — the conv is the part the overlap scheme acts on, so this is the meaningful proxy.
    _MOCK_4X8_BLK = {
        (512, 512, 34, 60): (64, 256, 1, 4, 8),  # ltx_s1_res 4x8 per-dev 17x15
        (512, 512, 68, 120): (64, 256, 1, 8, 4),  # ltx_s2_res 4x8 per-dev 34x30
        (256, 256, 68, 120): (64, 256, 1, 8, 4),  # ltx_s3_res 4x8 per-dev 34x30
        (256, 512, 68, 120): (64, 256, 1, 8, 4),  # ltx_s3_chg 4x8 per-dev 34x30
        (128, 128, 136, 240): (
            128,
            64,
            6,
            4,
            4,
        ),  # ltx_s4_res 4x8 per-dev 68x60 — force_spatial 6,4,4 (deployed fused-only)
        (128, 48, 136, 240): (128, 64, 6, 2, 16),  # ltx_s4_out 4x8 per-dev 68x60 — standalone-optimal blocking
        (1024, 1024, 20, 32): (128, 64, 5, 4, 8),  # ltx ups_post_res 4x8 per-dev 10x8
        (1024, 128, 20, 32): (128, 64, 7, 8, 4),  # ltx ups_final 4x8 per-dev 10x8
        (128, 1024, 18, 32): (64, 128, 7, 8, 4),  # ltx_s0_conv_in 4x8 per-dev 9x8
        (1024, 4096, 18, 32): (128, 64, 5, 4, 8),  # ltx_s0_up 4x8 per-dev 9x8
        (512, 4096, 34, 60): (128, 64, 5, 4, 8),  # ltx_s1_up 4x8 per-dev 17x15
    }
    if isinstance(model, WanCausalConv3d) and (C_in, C_out, H, W) in _MOCK_4X8_BLK:
        ci, co, t, h, w = _MOCK_4X8_BLK[(C_in, C_out, H, W)]
        model.conv_config.C_in_block = ci
        model.conv_config.C_out_block = co
        model.conv_config.T_out_block = t
        model.conv_config.H_out_block = h
        model.conv_config.W_out_block = w
    # NP_S4_BLK="Cin,Cout,T,H,W": override the s4_res blocking to sweep halo_last's boundary-ring
    # fraction (finer H/W blocks → smaller ring → interior pass hides NP). Applies to every s4_res
    # variant (2x4 and 4x8-mock; both are C_in=C_out=128, T=147).
    import os as _os

    _s4blk = _os.environ.get("NP_S4_BLK")
    if _s4blk and isinstance(model, WanCausalConv3d) and C_in == 128 and C_out == 128 and T == 147:
        ci, co, t, h, w = (int(x) for x in _s4blk.split(","))
        model.conv_config.C_in_block = ci
        model.conv_config.C_out_block = co
        model.conv_config.T_out_block = t
        model.conv_config.H_out_block = h
        model.conv_config.W_out_block = w
    # NP_S4OUT_BLK="Cin,Cout,T,H,W": same blocking-override knob for the s4_out shape (C_out=48).
    # Sweeps halo_last's boundary fraction for the lighter-matmul out conv.
    _s4outblk = _os.environ.get("NP_S4OUT_BLK")
    if _s4outblk and isinstance(model, WanCausalConv3d) and C_in == 128 and C_out == 48 and T == 147:
        ci, co, t, h, w = (int(x) for x in _s4outblk.split(","))
        model.conv_config.C_in_block = ci
        model.conv_config.C_out_block = co
        model.conv_config.T_out_block = t
        model.conv_config.H_out_block = h
        model.conv_config.W_out_block = w
    # NP_BLK="Cin,Cout,T,H,W": generic blocking override applied ONLY to the fused model (use_fused),
    # so the standalone baseline measured in the same run keeps its tuned _BLOCKINGS entry. Lets us
    # sweep a fused-only finer blocking on any shape without regressing the standalone comparison.
    _npblk = _os.environ.get("NP_BLK")
    if _npblk and use_fused and isinstance(model, WanCausalConv3d):
        ci, co, t, h, w = (int(x) for x in _npblk.split(","))
        model.conv_config.C_in_block = ci
        model.conv_config.C_out_block = co
        model.conv_config.T_out_block = t
        model.conv_config.H_out_block = h
        model.conv_config.W_out_block = w
    if use_fused and isinstance(model, WanCausalConv3d):
        # Bypass the MIN_T_FOR_FUSED hybrid-dispatch threshold — the perf test exists
        # specifically to measure fused kernel perf across all shapes, including below
        # the production threshold.
        if not model._use_fused and model._needs_halo:
            model._use_fused = True
        import os

        if os.environ.get("NP_FORCE_SPATIAL"):
            model.conv_config.force_spatial_parallel = True
        if os.environ.get("NP_HALO_LAST"):
            model.conv_config.halo_last = True
    # NP_REDUCE_GRID_COLS: shrink the standalone conv3d's core grid to approximate the fused
    # op's conv3d grid (which loses the column-0 fabric cores), isolating conv3d-on-reduced-grid
    # time from the fused op's halo-overlap stall.
    import os

    if os.environ.get("NP_REDUCE_GRID_COLS") and not use_fused:
        full = model.conv_config.compute_with_storage_grid_size
        model.conv_config.compute_with_storage_grid_size = ttnn.CoreCoord(
            full.x - int(os.environ["NP_REDUCE_GRID_COLS"]), full.y
        )
        logger.info(f"REDUCED conv3d grid {full} -> {model.conv_config.compute_with_storage_grid_size}")
    return model, h_factor, w_factor, parallel_config


def _build_input(mesh_device, B, C_in, T, H, W, h_axis, w_axis):
    torch.manual_seed(42)
    x = torch.randn(B, C_in, T, H, W, dtype=torch.float32)
    x = x.permute(0, 2, 3, 4, 1)
    x = conv_pad_in_channels(x)
    x, logical_h = conv_pad_height(x, tuple(mesh_device.shape)[h_axis])
    x = typed_tensor_2dshard(
        x,
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={h_axis: 2, w_axis: 3},
        dtype=ttnn.bfloat16,
    )
    return x, logical_h


def _measure(model, input_tensor, logical_h, mesh_device, *, n):
    # Warmup: pay JIT/cache costs once.
    _ = model(input_tensor, logical_h=logical_h)
    ttnn.synchronize_device(mesh_device)

    t0 = time.perf_counter_ns()
    for _ in range(n):
        out = model(input_tensor, logical_h=logical_h)
    ttnn.synchronize_device(mesh_device)
    t1 = time.perf_counter_ns()

    try:
        ttnn.deallocate(out)
    except Exception:
        pass

    return (t1 - t0) / n / 1e6  # ms per dispatch


@pytest.mark.parametrize(
    "B, C_in, C_out, T, H, W, kernel_size, padding, mesh_device, h_axis, w_axis, num_links, shape_id",
    [
        # LTX VAE 2x4 production k3 residual convs. T,H,W chosen so the per-device key
        # (h=2,w=4,C_in,C_out,k3,T,H/2,W/4) hits the tuned _BLOCKINGS entry (else conv3d
        # falls back to a generic blocking, ~2x slower, and the wrong T_out_block skews the
        # progress-sem pipelining granularity). Some LoudBox wiring forms only a 2x4 mesh.
        # Always num_links=2: 8 fabric cores (num_links*2 H + pad2_num_links*2 W) parallelize the
        # NP+border work enough to balance against the interior on the ~102 bulk cores.
        (1, 128, 1024, 21, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_conv_in_2x4"),  # per-dev 17x15
        (1, 1024, 4096, 21, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_up_2x4"),  # per-dev 17x15
        (1, 512, 4096, 39, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s1_up_2x4"),  # per-dev 34x30
        (1, 1024, 1024, 21, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_res_2x4"),
        (1, 512, 512, 39, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s1_res_2x4"),
        (1, 512, 512, 75, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s2_res_2x4"),
        (1, 256, 256, 147, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s3_res_2x4"),
        (1, 256, 512, 147, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s3_chg_2x4"),  # per-dev 68x60
        (1, 128, 128, 147, 272, 480, 3, 1, (2, 4), 0, 1, 2, "ltx_s4_res_2x4"),
        (1, 128, 48, 147, 272, 480, 3, 1, (2, 4), 0, 1, 2, "ltx_s4_out_2x4"),
        # 4x8 mock: the LTX VAE on a 4x8 mesh splits H/4,W/8 → quarter the per-device spatial of 2x4.
        # We run those 4x8 per-device sizes on this 2x4 box (global = 4x8-per-device * 2 * 4) so the
        # conv — where the overlap scheme matters — is exercised at the 4x8 scale. NP topology differs.
        (1, 128, 1024, 21, 18, 32, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_conv_in_4x8mock"),  # per-dev 9x8
        (1, 1024, 4096, 21, 18, 32, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_up_4x8mock"),  # per-dev 9x8
        (1, 512, 4096, 39, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s1_up_4x8mock"),  # per-dev 17x15
        (1, 512, 512, 39, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s1_res_4x8mock"),  # per-dev 17x15
        (1, 512, 512, 75, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s2_res_4x8mock"),  # per-dev 34x30
        (1, 256, 256, 147, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s3_res_4x8mock"),  # per-dev 34x30
        (1, 256, 512, 147, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s3_chg_4x8mock"),  # per-dev 34x30
        (1, 128, 128, 147, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s4_res_4x8mock"),  # per-dev 68x60
        (1, 128, 48, 147, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s4_out_4x8mock"),  # per-dev 68x60
        # 4x8 latent-upsampler k3 convs (per-dev 10x8, huge channels) — expected NP-light.
        (1, 1024, 1024, 21, 20, 32, 3, 1, (2, 4), 0, 1, 2, "ltx_ups_post_res_4x8mock"),  # per-dev 10x8
        (1, 1024, 128, 21, 20, 32, 3, 1, (2, 4), 0, 1, 2, "ltx_ups_final_4x8mock"),  # per-dev 10x8
    ],
    ids=[
        "ltx_s0_conv_in_2x4",
        "ltx_s0_up_2x4",
        "ltx_s1_up_2x4",
        "ltx_s0_res_2x4",
        "ltx_s1_res_2x4",
        "ltx_s2_res_2x4",
        "ltx_s3_res_2x4",
        "ltx_s3_chg_2x4",
        "ltx_s4_res_2x4",
        "ltx_s4_out_2x4",
        "ltx_s0_conv_in_4x8mock",
        "ltx_s0_up_4x8mock",
        "ltx_s1_up_4x8mock",
        "ltx_s1_res_4x8mock",
        "ltx_s2_res_4x8mock",
        "ltx_s3_res_4x8mock",
        "ltx_s3_chg_4x8mock",
        "ltx_s4_res_4x8mock",
        "ltx_s4_out_4x8mock",
        "ltx_ups_post_res_4x8mock",
        "ltx_ups_final_4x8mock",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.timeout(600)
def test_fused_vs_standalone_perf(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype, shape_id
):
    """Back-to-back fused vs standalone wall-clock. Log-only — no assertions.

    Promote to a hard assert once numbers are stable and a target ratio is
    agreed on.
    """
    # Build standalone model + input first
    sa_model, _, _, _ = _build_model(
        mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype, use_fused=False
    )
    sa_input, sa_logical_h = _build_input(mesh_device, B, C_in, T, H, W, h_axis, w_axis)
    sa_ms = _measure(sa_model, sa_input, sa_logical_h, mesh_device, n=NUM_MEASURED_DISPATCHES)
    ttnn.deallocate(sa_input)

    # Build fused model + input
    f_model, _, _, _ = _build_model(
        mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype, use_fused=True
    )
    f_input, f_logical_h = _build_input(mesh_device, B, C_in, T, H, W, h_axis, w_axis)
    f_ms = _measure(f_model, f_input, f_logical_h, mesh_device, n=NUM_MEASURED_DISPATCHES)
    ttnn.deallocate(f_input)

    ratio = f_ms / sa_ms if sa_ms > 0 else float("inf")
    logger.info(
        f"PERF shape={shape_id}  fused={f_ms:.3f}ms  standalone={sa_ms:.3f}ms  ratio={ratio:.3f}  "
        f"(lower is better; n={NUM_MEASURED_DISPATCHES})"
    )
    # Intentionally no assertion — log-only until numbers stabilize.
