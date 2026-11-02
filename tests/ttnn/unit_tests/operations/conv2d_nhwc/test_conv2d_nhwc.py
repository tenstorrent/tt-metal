# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for conv2d_nhwc (Phase 0 contract) — immutable spec.

Phase 0 SUPPORTED: float32 activations and weights, ROW_MAJOR, DRAM interleaved,
tile-aligned C_in / C_out, stride 1, dilation 1, groups 1, no fused activation,
with and without bias. Shapes cover a single tile, multi-tile, non-square,
multi-batch, multiple K-blocks (large K), a C_out split across cores (small M),
multiple tile-rows per core (large M), padding, and the flattened
`input_spatial` input form.
"""

import pytest
import torch
import ttnn

from ttnn.operations.conv2d_nhwc import (
    conv2d_nhwc,
    prepare_conv2d_bias,
    prepare_conv2d_weights,
)

PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def torch_reference(x_nhwc, w, b, *, padding, stride=1, groups=1, dilation=1):
    x = x_nhwc.float().permute(0, 3, 1, 2)
    y = torch.nn.functional.conv2d(
        x,
        w.float(),
        b.float() if b is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return y.permute(0, 2, 3, 1).contiguous()


# (input_shape NHWC, C_out, kernel_size, padding)
CASES = [
    pytest.param((1, 1, 1, 32), 32, 1, 0, id="single_tile_1x1"),
    pytest.param((1, 8, 8, 32), 64, 3, 1, id="multi_tile_k3_pad1"),
    pytest.param((1, 16, 32, 64), 64, 3, 0, id="non_square_k3"),
    pytest.param((2, 14, 14, 128), 128, 3, 1, id="multi_batch_k3_pad1"),
    pytest.param((1, 28, 28, 32), 64, 5, 2, id="k5_pad2"),
    pytest.param((1, 14, 14, 256), 512, 3, 0, id="large_K_multi_kblock_nsplit"),
    pytest.param((1, 112, 112, 32), 64, 3, 0, id="large_M_multi_row_per_core"),
    pytest.param((4, 14, 14, 128), 128, 3, 0, id="batch4_k3"),
    pytest.param((1, 28, 28, 32), 64, 7, 3, id="k7_pad3"),
]


def _run(device, input_shape, C_out, k, padding, with_bias, flattened):
    torch.manual_seed(42)
    dtype = ttnn.float32
    N, H, W, C_in = input_shape
    x = torch.randn(N, H, W, C_in, dtype=torch.float32)
    w = torch.randn(C_out, C_in, k, k, dtype=torch.float32)
    b = torch.randn(C_out, dtype=torch.float32) if with_bias else None

    expected = torch_reference(x, w, b, padding=padding)
    H_out = H + 2 * padding - k + 1
    W_out = W + 2 * padding - k + 1

    x_dev_src = x.reshape(1, 1, N * H * W, C_in) if flattened else x
    tt_x = ttnn.from_torch(
        x_dev_src,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    common = dict(
        kernel_size=k,
        stride=1,
        padding=padding,
        groups=1,
        dilation=1,
        input_dtype=dtype,
        weight_dtype=dtype,
        input_memory_config=tt_x.memory_config(),
        device=device,
    )
    tt_w = prepare_conv2d_weights(w, **common)
    tt_b = prepare_conv2d_bias(b, **common) if with_bias else None

    kwargs = {"input_spatial": (N, H, W)} if flattened else {}
    tt_out = conv2d_nhwc(tt_x, tt_w, k, bias=tt_b, padding=padding, **kwargs)

    expected_shape = [1, 1, N * H_out * W_out, C_out] if flattened else [N, H_out, W_out, C_out]
    assert list(tt_out.shape) == expected_shape
    assert tt_out.dtype == dtype
    assert tt_out.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tt_out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED

    out = ttnn.to_torch(tt_out).float().reshape(N, H_out, W_out, C_out)
    p = pcc(out, expected)
    assert p >= PCC_BY_DTYPE[dtype], f"PCC {p:.6f} < {PCC_BY_DTYPE[dtype]}"


@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("input_shape, C_out, k, padding", CASES)
def test_conv2d_nhwc(device, input_shape, C_out, k, padding, with_bias):
    _run(device, input_shape, C_out, k, padding, with_bias, flattened=False)


@pytest.mark.parametrize(
    "input_shape, C_out, k, padding",
    [
        pytest.param((2, 14, 14, 64), 64, 3, 1, id="flat_batch2_k3"),
        pytest.param((1, 32, 32, 64), 128, 1, 0, id="flat_1x1"),
    ],
)
def test_conv2d_nhwc_flattened_input_spatial(device, input_shape, C_out, k, padding):
    _run(device, input_shape, C_out, k, padding, with_bias=True, flattened=True)


def test_conv2d_nhwc_rejects_unsupported_stride(device, expect_error):
    """Outside Phase 0 SUPPORTED -> NotImplementedError naming the stride axis (validate() gate)."""
    torch.manual_seed(42)
    x = torch.randn(1, 8, 8, 32, dtype=torch.float32)
    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w = torch.randn(32, 32, 3, 3, dtype=torch.float32)
    with expect_error(NotImplementedError, "stride"):
        tt_w = prepare_conv2d_weights(
            w,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            dilation=1,
            input_dtype=ttnn.float32,
            weight_dtype=ttnn.float32,
            input_memory_config=tt_x.memory_config(),
            device=device,
        )
        conv2d_nhwc(tt_x, tt_w, 3, padding=1, stride=2)
