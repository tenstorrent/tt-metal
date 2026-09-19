# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The in-kernel fp32 operand split (`split_mode="kernel"`, `Conv3dConfig.operand_split`) against the three-conv
host split (`"full"`) it replaces, at H3 audio shapes, one chip.

Same operands reach the matrix engine in both forms; only the summation differs (one fp32 DST accumulation of
three K passes versus three K accumulations plus two fp32 SFPU adds), so the two must agree to a few fp32 ulp and
both must sit far below the unsplit conv's error against a float64 reference. Also times the three forms.
"""

import copy
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.audio_ops import Conv1dViaConv3d
from ....models.audio_vae.minimax_h3.blockings_minimax_h3_audio import register_h3_audio_blockings

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

# (in_channels, out_channels, kernel, frames): the widest AMP band, a narrow band, and conv_pre.
SHAPES = [
    pytest.param(512, 512, 11, 400, id="band0_k11"),
    pytest.param(512, 512, 3, 400, id="band0_k3"),
    pytest.param(128, 128, 3, 1900, id="band2_k3"),
    pytest.param(2048, 1024, 7, 80, id="conv_pre_k7"),
]
MODES = ("off", "full", "kernel")


def _best(fn, mesh_device, n=10):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - t0)
    return best, out


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("in_channels", "out_channels", "kernel", "frames"), SHAPES)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_kernel_split_matches_host_split(mesh_device, in_channels, out_channels, kernel, frames):
    register_h3_audio_blockings()
    torch.manual_seed(0)
    reference = torch.nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2).float().eval()
    x = torch.randn(2, in_channels, frames) * 0.1
    with torch.no_grad():
        golden = copy.deepcopy(reference).double()(x.double()).float()
    x_device = ttnn.from_torch(
        x.transpose(1, 2).contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
    )

    outputs, errors, times = {}, {}, {}
    for mode in MODES:
        layer = Conv1dViaConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            padding_mode="zeros",
            bias=True,
            mesh_device=mesh_device,
            dtype=ttnn.float32,
            split_mode=mode,
        )
        assert layer.split_mode == mode
        layer.load_torch_state_dict(
            {"weight": reference.weight.detach().contiguous(), "bias": reference.bias.detach().contiguous()},
            strict=False,
        )
        best, out = _best(lambda: layer(x_device), mesh_device)
        actual = ttnn.to_torch(out).float().transpose(1, 2)
        outputs[mode] = actual
        errors[mode] = float((actual.double() - golden.double()).pow(2).mean().sqrt() / golden.double().std())
        times[mode] = best * 1e3

    logger.info(
        f"({in_channels}->{out_channels}, k{kernel}, T{frames}) rel_rmse "
        + ", ".join(f"{m}={errors[m]:.3e}" for m in MODES)
        + " | min-of-10 ms "
        + ", ".join(f"{m}={times[m]:.3f}" for m in MODES)
    )
    kernel_vs_full = (outputs["kernel"].double() - outputs["full"].double()).abs()
    scale = outputs["full"].double().abs().max().item()
    logger.info(
        f"kernel vs full: max |diff| {kernel_vs_full.max().item():.3e} ({kernel_vs_full.max().item() / scale:.3e} of "
        f"max |out|), mean |diff| {kernel_vs_full.mean().item():.3e}"
    )

    # The two splits see the same operands; only the fp32 summation order differs.
    assert errors["kernel"] <= 1.05 * errors["full"], f"kernel split is less accurate than the host split: {errors}"
    assert errors["kernel"] <= 0.70 * errors["off"], f"kernel split did not beat the unsplit conv: {errors}"
    assert kernel_vs_full.max().item() <= 1e-5 * scale, f"kernel and host splits disagree by {kernel_vs_full.max():.3e}"
