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
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype, *, use_fused
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
    if use_fused and isinstance(model, WanCausalConv3d) and model.conv_config.T_out_block > 0:
        model.conv_config.input_progress_t_batch_size = model.conv_config.T_out_block
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
        # 2x2 mid_block (C_in=C_out=384 3x3x3)
        (1, 384, 384, 7, 60, 104, 3, 1, (2, 2), 0, 1, 1, "mid_res_2x2_480p"),
        # 2x2 conv_in (C_in=32 → C_out=384 3x3x3)
        (1, 32, 384, 7, 60, 104, 3, 1, (2, 2), 0, 1, 1, "conv_in_2x2_480p"),
        # 2x2 up1 residual (C_in=C_out=384 3x3x3, T=14)
        (1, 384, 384, 14, 120, 208, 3, 1, (2, 2), 0, 1, 1, "up1_res_2x2_480p"),
        # 2x2 up2 residual (C_in=C_out=192 3x3x3, T=28)
        (1, 192, 192, 28, 240, 416, 3, 1, (2, 2), 0, 1, 1, "up2_res_2x2_480p"),
    ],
    ids=["mid_res_2x2_480p", "conv_in_2x2_480p", "up1_res_2x2_480p", "up2_res_2x2_480p"],
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
