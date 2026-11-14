# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout

MAX_ERROR_VS_ROUNDED = 1.5
MAX_ERROR_VS_ROUNDED_FEW_K_BLOCKS = 1.08


def _conv(
    device, x, w, b, packer_l1_acc, shard_layout, dtype=ttnn.bfloat16, output_layout=ttnn.TILE_LAYOUT, padding=(1, 1)
):
    cout, cin, kh, kw = w.shape
    hw = x.shape[2]
    out_h, out_w = hw + 2 * padding[0] - kh + 1, hw + 2 * padding[1] - kw + 1
    tx = ttnn.from_torch(
        x.permute(0, 2, 3, 1).reshape(1, 1, hw * hw, cin),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias = {}
    if b is not None:
        bias["bias_tensor"] = ttnn.from_torch(
            b.reshape(1, 1, 1, cout), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4 if is_blackhole() else ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )
    out = ttnn.conv2d(
        input_tensor=tx,
        weight_tensor=tw,
        in_channels=cin,
        out_channels=cout,
        device=device,
        kernel_size=(kh, kw),
        stride=(1, 1),
        padding=padding,
        batch_size=1,
        input_height=hw,
        input_width=hw,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16, shard_layout=shard_layout, output_layout=output_layout
        ),
        compute_config=compute_config,
        groups=1,
        dtype=dtype,
        **bias,
    )
    return ttnn.to_torch(out).float().reshape(-1, cout)[: out_h * out_w]


def _inputs(cin, cout, hw, kernel, has_bias):
    torch.manual_seed(0)
    x = torch.randn(1, cin, hw, hw).bfloat16().float()
    w = (torch.randn(cout, cin, *kernel) / (cin * kernel[0] * kernel[1]) ** 0.5).bfloat16().float()
    b = (torch.randn(cout) * 0.5).bfloat16().float() if has_bias else None
    return x, w, b


def _reference(x, w, b, padding):
    cout = w.shape[0]
    return (
        torch.nn.functional.conv2d(x.double(), w.double(), bias=None if b is None else b.double(), padding=padding)
        .permute(0, 2, 3, 1)
        .reshape(-1, cout)
    )


def _max_error(out, ref):
    return (out.double() - ref).abs().max().item() / ref.abs().max().item()


def _check(out, ref, dtype, max_error_vs_rounded=MAX_ERROR_VS_ROUNDED):
    passing, pcc = check_with_pcc_without_tensor_printout(ref.float(), out, pcc=0.999)
    assert passing, pcc
    if dtype == ttnn.bfloat16:
        rounded = _max_error(ref.float().bfloat16().float(), ref)
        err = _max_error(out, ref)
        assert err <= max_error_vs_rounded * rounded, (
            f"max relative error {err:.5f} is {err / rounded:.2f}x that of the exact result rounded to bfloat16 "
            f"({rounded:.5f}); allowed {max_error_vs_rounded}x"
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize(
    "cin, hw, shard_layout",
    [
        (256, 8, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (256, 8, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (256, 8, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        (320, 64, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (320, 64, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ],
)
def test_conv2d_fp32_accumulation_without_packer_l1_acc(device, cin, hw, shard_layout, has_bias):
    x, w, b = _inputs(cin, cin, hw, (3, 3), has_bias)
    ref = _reference(x, w, b, (1, 1))
    out_l1 = _conv(device, x, w, b, True, shard_layout)
    out_no_l1 = _conv(device, x, w, b, False, shard_layout)
    _check(out_no_l1, ref, ttnn.bfloat16)
    err_l1, err_no_l1 = _max_error(out_l1, ref), _max_error(out_no_l1, ref)
    assert err_no_l1 <= 1.1 * err_l1 + 1e-5, (
        f"fp32_dest_acc_en with packer_l1_acc=False: max relative error {err_no_l1:.5f}, "
        f"{err_no_l1 / max(err_l1, 1e-12):.2f}x the packer_l1_acc=True result {err_l1:.5f}"
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize(
    "dtype, output_layout",
    [
        (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        (ttnn.float32, ttnn.TILE_LAYOUT),
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_conv2d_fp32_accumulation_output_formats(device, dtype, output_layout, has_bias):
    x, w, b = _inputs(256, 256, 8, (3, 3), has_bias)
    out = _conv(device, x, w, b, False, ttnn.TensorMemoryLayout.BLOCK_SHARDED, dtype=dtype, output_layout=output_layout)
    _check(out, _reference(x, w, b, (1, 1)), dtype)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("packer_l1_acc", [False, True])
@pytest.mark.parametrize(
    "shard_layout", [ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED]
)
@pytest.mark.parametrize(
    "kernel, has_bias, cout, output_layout, dtype",
    [
        ((2, 2), False, 64, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((2, 2), False, 64, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        ((2, 2), False, 320, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        ((2, 2), False, 64, ttnn.TILE_LAYOUT, ttnn.float32),
        ((2, 2), False, 64, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        ((1, 1), True, 64, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((1, 1), False, 64, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
def test_conv2d_fp32_accumulation_few_k_blocks(
    device, kernel, has_bias, cout, output_layout, dtype, shard_layout, packer_l1_acc
):
    padding = (0, 0)
    x, w, b = _inputs(64, cout, 16, kernel, has_bias)
    out = _conv(device, x, w, b, packer_l1_acc, shard_layout, dtype=dtype, output_layout=output_layout, padding=padding)
    _check(out, _reference(x, w, b, padding), dtype, MAX_ERROR_VS_ROUNDED_FEW_K_BLOCKS)
