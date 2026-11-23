# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Layer norm and RMS norm at model widths, which the small Sanity shapes do not reach.

The widest rows in the other fused Sanity tests are W=4096. Model hidden sizes of 5120 and up pick a
different program: the large-tensor factory, when a whole row no longer fits in L1. That is where
#53419 regressed (BH DeepSeek RMS norm, PCC 0.984), and where #52081 doubled the L1 use of the
pre-all-gather stage by following fp32_dest_acc_en. The distributed norm is covered end to end on one
device: pre-all-gather statistics, then post-all-gather with the weight.

The row-width test here uses a few tile rows, so the simulators run it too: the factory is chosen from the
width, not the height. The full-grid size, with 4 tile rows on every core, is in
tests/ttnn/nightly/unit_tests/operations/fused/test_norm_model_widths_full_grid.py.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics

EPS = 1e-5
# PCC ignores a constant scale, so the relative Frobenius error is checked too. HiFi2 with bf16 gives 0.007 to 0.009
# on BH for W = 4096 to 8192; a doubled output gives about 1.0.
FROBENIUS = 0.02


def norm_reference(op, x, weight, bias):
    x = x.float()
    if op == "layer_norm":
        return F.layer_norm(x, (x.shape[-1],), weight.float(), bias.float(), EPS)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()


# The factory is chosen from the width (the L1 footprint of a row), not the height, so a few tile rows suffice.
TILE_ROWS = 2


def run_norm_model_width(device, op, w, h):
    torch.manual_seed(0)
    torch_input = torch.randn((1, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    tt_bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)
    if op == "layer_norm":
        tt_output = ttnn.layer_norm(
            tt_input, epsilon=EPS, weight=tt_weight, bias=tt_bias, compute_kernel_config=compute_config
        )
    else:
        tt_output = ttnn.rms_norm(tt_input, epsilon=EPS, weight=tt_weight, compute_kernel_config=compute_config)
    output = ttnn.to_torch(tt_output).float()

    # Small heights compare every row; large heights sample rows to keep the reference cheap.
    rows = torch.arange(h) if h <= 64 else torch.arange(0, h, 17)
    golden = norm_reference(op, torch_input[:, rows], torch_weight, torch_bias)
    assert_numeric_metrics(
        golden, output[:, rows], pcc_threshold=0.999, frobenius_threshold=FROBENIUS, check_allclose=False
    )


@pytest.mark.parametrize("op", ["layer_norm", "rms_norm"])
# Where the large-tensor factory starts depends on L1, so all three widths run.
# W >= 5120 uses the large-tensor kernels today (#53419); recheck if the L1 rule in
# layernorm_op_multi_core.cpp changes.
@pytest.mark.parametrize("w", [4096, 5120, 8192])
def test_norm_model_width_hifi2_fp32_acc(device, op, w):
    run_norm_model_width(device, op, w, h=32 * TILE_ROWS)


@pytest.mark.parametrize("op", ["layer_norm", "rms_norm"])
# The widest rows that fit L1 with fp32 intermediates. Every buffer holds a whole row, so rows wider than
# these throw at allocation: no residual at W=8192 (x^2 and input are ~1 MB each) and a residual at W=4096
# (four row buffers, ~2 MB). Model hidden sizes reach this stage divided by the number of devices.
@pytest.mark.parametrize("w, with_residual", [(2048, False), (4096, False), (5120, False), (1024, True), (2048, True)])
def test_pre_all_gather_fp32_acc_fits_l1(device, op, w, with_residual):
    """bf16 input with fp32_dest_acc_en must still fit L1 at the widest widths that fit (#54697: after #52081
    stored the x^2 and pre-add buffers as fp32, a W=4096 row threw at allocation), with the right statistics."""
    torch.manual_seed(0)
    h = 64
    torch_input = torch.randn((1, 1, h, w), dtype=torch.bfloat16)
    torch_residual = torch.randn((1, 1, h, w), dtype=torch.bfloat16)
    fused = (torch_input + torch_residual if with_residual else torch_input).float()
    sum_x2 = fused.pow(2).sum(-1)[0, 0]
    sum_x = fused.sum(-1)[0, 0]

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    extra = {}
    if with_residual:
        extra["residual_input_tensor"] = ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device)
    pre_all_gather = ttnn.layer_norm_pre_all_gather if op == "layer_norm" else ttnn.rms_norm_pre_all_gather
    stats = ttnn.to_torch(
        pre_all_gather(tt_input, compute_kernel_config=compute_config, dtype=ttnn.bfloat16, **extra)
    ).float()

    # Column 0 of the stats tile holds sum(x^2) per row; layer_norm adds sum(x) in column 32. The bound is
    # about twice the bf16 rounding of the sum, so a wrong buffer format or a dropped block still fails.
    torch.testing.assert_close(stats[0, 0, :, 0], sum_x2, rtol=0.02, atol=1.0)
    if op == "layer_norm":
        torch.testing.assert_close(stats[0, 0, :, 32], sum_x, rtol=0.02, atol=2.0)


# Whole-row buffers again: the input, weight, bias and fp32 intermediates are each a row long, so these are the
# widest widths that fit L1. layer_norm (which also holds the bias) throws at allocation from W=4096 and
# rms_norm from W=5120.
@pytest.mark.parametrize("op, w", [("layer_norm", 2048), ("rms_norm", 2048), ("rms_norm", 4096)])
def test_pre_then_post_all_gather_with_weight(device, op, w):
    """The whole distributed norm on one device: with a single device the pre-all-gather statistics already
    are the gathered ones, so they go straight into the post-all-gather op with the weight (and the bias for
    layer_norm) on that device. Weight and bias are row-major [1, 1, W / 32, 32] as the post stage expects."""
    torch.manual_seed(0)
    h = 64
    torch_input = torch.randn((1, 1, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_weight = ttnn.from_torch(torch_weight.reshape(1, 1, w // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_bias = ttnn.from_torch(torch_bias.reshape(1, 1, w // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    if op == "layer_norm":
        stats = ttnn.layer_norm_pre_all_gather(tt_input, compute_kernel_config=compute_config, dtype=ttnn.bfloat16)
        tt_output = ttnn.layer_norm_post_all_gather(
            tt_input,
            stats,
            epsilon=EPS,
            weight=tt_weight,
            bias=tt_bias,
            compute_kernel_config=compute_config,
            dtype=ttnn.bfloat16,
        )
    else:
        stats = ttnn.rms_norm_pre_all_gather(tt_input, compute_kernel_config=compute_config, dtype=ttnn.bfloat16)
        tt_output = ttnn.rms_norm_post_all_gather(
            tt_input,
            stats,
            epsilon=EPS,
            weight=tt_weight,
            compute_kernel_config=compute_config,
            dtype=ttnn.bfloat16,
        )
    output = ttnn.to_torch(tt_output).float()

    golden = norm_reference(op, torch_input, torch_weight, torch_bias)
    assert_numeric_metrics(golden, output, pcc_threshold=0.999, frobenius_threshold=FROBENIUS, check_allclose=False)
