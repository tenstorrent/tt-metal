# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

WT = [128, 192, 256, 512, 2048, 8192]


def _config(fp32_dest_acc_en):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )


def _first(tt_output):
    return ttnn.to_torch(tt_output).double().reshape(-1)[0].item()


@pytest.mark.parametrize("wt", WT)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_sum_w_ones_is_exact(device, wt, fp32_dest_acc_en):
    w = wt * 32
    tt_input = ttnn.from_torch(torch.ones(1, 1, 32, w, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    got = _first(ttnn.sum(tt_input, dim=-1, compute_kernel_config=_config(fp32_dest_acc_en)))
    assert got == pytest.approx(float(w), rel=1e-6), f"sum of {w} ones returned {got}"


@pytest.mark.parametrize("wt", WT)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_sum_w_signed_matches_reference(device, wt, fp32_dest_acc_en):
    torch.manual_seed(0)
    w = wt * 32
    torch_input = torch.randn(1, 1, 32, w).to(torch.bfloat16)
    expected = torch_input[0, 0, 0].double().sum().item()
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    got = _first(ttnn.sum(tt_input, dim=-1, compute_kernel_config=_config(fp32_dest_acc_en)))
    assert abs(got - expected) <= 0.01 * abs(expected), f"sum over {w} columns returned {got}, expected {expected}"


@pytest.mark.parametrize("wt", [192, 512, 2048])
def test_mean_w_matches_reference(device, wt):
    torch.manual_seed(0)
    w = wt * 32
    torch_input = torch.randn(1, 1, 32, w).to(torch.bfloat16)
    expected = torch_input[0, 0, 0].double().mean().item()
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    got = _first(ttnn.mean(tt_input, dim=-1, compute_kernel_config=_config(False)))
    assert abs(got - expected) <= 0.01 * abs(expected), f"mean over {w} columns returned {got}, expected {expected}"


@pytest.mark.parametrize("h, w", [(32, 4096), (32, 65536), (1024, 8192)])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_sum_hw_ones_is_exact(device, h, w, fp32_dest_acc_en):
    tt_input = ttnn.from_torch(torch.ones(1, 1, h, w, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    got = _first(ttnn.sum(tt_input, dim=(-2, -1), compute_kernel_config=_config(fp32_dest_acc_en)))
    assert got == pytest.approx(float(h * w), rel=1e-6), f"sum of {h}x{w} ones returned {got}"
