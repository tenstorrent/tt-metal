# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Polyphase vs zero-stuffed ConvTranspose1dViaConv3d at H3."""

import copy
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.audio_ops import ConvTranspose1dViaConv3d
from ....models.audio_vae.minimax_h3.blockings_minimax_h3_audio import register_h3_audio_blockings

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

SHAPES = [
    pytest.param(1024, 512, 9, 5, 80, id="ups0_k9_s5"),
    pytest.param(256, 128, 4, 2, 400, id="ups2_k4_s2"),
    pytest.param(16, 8, 4, 2, 1500, id="ups6_k4_s2"),
]


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
@pytest.mark.parametrize(("in_channels", "out_channels", "kernel", "stride", "frames"), SHAPES)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_polyphase_matches_zero_stuffed(mesh_device, in_channels, out_channels, kernel, stride, frames):
    register_h3_audio_blockings()
    torch.manual_seed(0)
    reference = torch.nn.ConvTranspose1d(
        in_channels, out_channels, kernel, stride=stride, padding=(kernel - stride) // 2
    ).float()
    x = torch.randn(2, in_channels, frames) * 0.1
    with torch.no_grad():
        golden = copy.deepcopy(reference).double()(x.double()).float()
    x_device = ttnn.from_torch(
        x.transpose(1, 2).contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
    )
    state = {"weight": reference.weight.detach().contiguous(), "bias": reference.bias.detach().contiguous()}

    outputs, errors, times = {}, {}, {}
    for name, polyphase in (("stuffed", False), ("polyphase", True)):
        layer = ConvTranspose1dViaConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            bias=True,
            mesh_device=mesh_device,
            dtype=ttnn.float32,
            split_mode="full",
            polyphase=polyphase,
        )
        layer.load_torch_state_dict(dict(state), strict=False)
        best, out = _best(lambda: layer(x_device), mesh_device)
        actual = ttnn.to_torch(out).float().transpose(1, 2)
        assert actual.shape == golden.shape, f"{name}: {tuple(actual.shape)} != {tuple(golden.shape)}"
        outputs[name] = actual
        errors[name] = float((actual.double() - golden.double()).pow(2).mean().sqrt() / golden.double().std())
        times[name] = best * 1e3

    ends = max(
        (outputs["polyphase"][..., : 2 * kernel].double() - golden[..., : 2 * kernel].double()).abs().max().item(),
        (outputs["polyphase"][..., -2 * kernel :].double() - golden[..., -2 * kernel :].double()).abs().max().item(),
    )
    logger.info(
        f"({in_channels}->{out_channels}, k{kernel}, s{stride}, T{frames}) rel_rmse stuffed={errors['stuffed']:.3e} "
        f"polyphase={errors['polyphase']:.3e} | ends max|err| {ends:.3e} | min-of-10 ms stuffed={times['stuffed']:.3f} "
        f"polyphase={times['polyphase']:.3f}"
    )
    dd = outputs["polyphase"].double() - outputs["stuffed"].double()
    rms_dd = dd.pow(2).mean().sqrt().item()
    ends_dd = max(dd[..., : 2 * kernel].abs().max().item(), dd[..., -2 * kernel :].abs().max().item())
    scale = golden.abs().max().item()
    logger.info(f"  polyphase vs stuffed on device: rms {rms_dd:.3e}, ends max {ends_dd:.3e}, max |out| {scale:.3e}")
    assert errors["polyphase"] <= 1.2 * errors["stuffed"], f"polyphase form is less accurate: {errors}"
    assert ends <= 1e-2 * scale, f"sequence ends differ from torch by {ends:.3e} (max |out| {scale:.3e}): crop bug"
    assert ends_dd <= max(
        10 * rms_dd, 1e-6 * scale
    ), f"the device forms disagree at the ends: {ends_dd:.3e} vs rms {rms_dd:.3e}"
