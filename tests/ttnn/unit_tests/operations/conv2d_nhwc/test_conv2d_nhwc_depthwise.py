# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4: depthwise convolution (regime R11, groups == C_in == C_out).

Per-tap eltwise multiply with row-broadcast tap weights, accumulated in DEST. Covers
interleaved (incl. n-split across cores, n-blocking inside a core, strided, 5x5, padding
edges, N > 1) and BLOCK / WIDTH sharded (no K exchange), bias / no bias, relu, and the
bf16 x bf8b mixed-precision case.
"""

import pytest
import torch
import ttnn

from ttnn.operations.conv2d_nhwc import conv2d_nhwc, prepare_conv2d_bias, prepare_conv2d_weights

from .test_conv2d_nhwc_sharded import _sharded_config, pcc

ML = ttnn.TensorMemoryLayout
_TORCH = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
_TOL = {ttnn.float32: (0.999, 0.02), ttnn.bfloat16: (0.995, 0.05)}


def run_dw(
    device,
    input_shape,
    k,
    padding,
    stride,
    memory_layout=ML.INTERLEAVED,
    with_bias=True,
    dtype=ttnn.float32,
    weight_dtype=ttnn.float32,
    activation=None,
):
    torch.manual_seed(0)
    N, H, W, C = input_shape
    x = torch.randn(input_shape, dtype=torch.float32).to(_TORCH[dtype]).float()
    w = torch.randn(C, 1, k, k, dtype=torch.float32).to(_TORCH[weight_dtype]).float()
    b = torch.randn(C, dtype=torch.float32).to(_TORCH[weight_dtype]).float() if with_bias else None
    ref = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, b, stride=stride, padding=padding, groups=C)
    ref = ref.permute(0, 2, 3, 1).contiguous()
    if activation == "relu":
        ref = torch.relu(ref)

    mc = _sharded_config(memory_layout, input_shape, device)
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    kw = dict(
        kernel_size=k,
        stride=stride,
        padding=padding,
        groups=C,
        input_dtype=dtype,
        weight_dtype=weight_dtype,
        input_memory_config=tt_x.memory_config(),
        device=device,
    )
    tt_w = prepare_conv2d_weights(w, **kw)
    tt_b = prepare_conv2d_bias(b, **kw) if with_bias else None
    out = conv2d_nhwc(tt_x, tt_w, k, bias=tt_b, padding=padding, stride=stride, groups=C, activation=activation)
    assert out.memory_config().memory_layout == memory_layout
    got = ttnn.to_torch(out).float()
    assert list(got.shape) == list(ref.shape)
    p = pcc(got, ref)
    rel_rms = ((got - ref).pow(2).mean().sqrt() / ref.std()).item()
    min_pcc, max_rms = _TOL[dtype]
    assert p >= min_pcc and rel_rms <= max_rms, f"pcc={p} rel_rms={rel_rms}"


CASES = [
    # (input_shape, k, padding, stride, memory_layout)
    pytest.param((1, 14, 14, 32), 3, 1, 1, ML.INTERLEAVED, id="il_14x14x32"),
    pytest.param((2, 9, 11, 64), 3, 1, 1, ML.INTERLEAVED, id="il_n2_odd_hw"),
    pytest.param((1, 28, 28, 96), 5, 2, 1, ML.INTERLEAVED, id="il_5x5_96"),
    pytest.param((1, 56, 56, 128), 3, 1, 2, ML.INTERLEAVED, id="il_strided_128"),
    pytest.param((1, 7, 7, 1024), 3, 1, 1, ML.INTERLEAVED, id="il_1024_nsplit"),
    pytest.param((1, 112, 112, 32), 3, 1, 1, ML.INTERLEAVED, id="il_112x112x32"),
    pytest.param((1, 16, 16, 64), 3, 0, 1, ML.INTERLEAVED, id="il_no_pad"),
    pytest.param((1, 7, 7, 1024), 3, 1, 1, ML.WIDTH_SHARDED, id="width_1024"),
    pytest.param((1, 14, 14, 256), 3, 1, 2, ML.WIDTH_SHARDED, id="width_strided"),
    pytest.param((1, 28, 28, 256), 3, 1, 1, ML.BLOCK_SHARDED, id="block_256"),
    pytest.param((2, 14, 14, 128), 5, 2, 1, ML.BLOCK_SHARDED, id="block_n2_5x5"),
]


@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("input_shape, k, padding, stride, memory_layout", CASES)
def test_conv2d_nhwc_depthwise(device, input_shape, k, padding, stride, memory_layout, with_bias):
    run_dw(device, input_shape, k, padding, stride, memory_layout, with_bias)


@pytest.mark.parametrize(
    "memory_layout, weight_dtype, activation",
    [
        (ML.INTERLEAVED, ttnn.bfloat8_b, None),
        (ML.INTERLEAVED, ttnn.bfloat16, "relu"),
        (ML.BLOCK_SHARDED, ttnn.bfloat8_b, "relu"),
    ],
)
def test_conv2d_nhwc_depthwise_bf16(device, memory_layout, weight_dtype, activation):
    run_dw(
        device,
        (1, 28, 28, 64),
        3,
        1,
        1,
        memory_layout,
        with_bias=activation is not None,
        dtype=ttnn.bfloat16,
        weight_dtype=weight_dtype,
        activation=activation,
    )


def test_conv2d_nhwc_depthwise_rejects_multiplier(device, expect_error):
    from ttnn.operations._op_contract import UnsupportedAxisValue

    x = ttnn.from_torch(torch.randn(1, 8, 8, 32), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    w = prepare_conv2d_weights(
        torch.randn(64, 1, 3, 3),
        kernel_size=3,
        padding=1,
        groups=32,
        input_dtype=ttnn.float32,
        weight_dtype=ttnn.float32,
        input_memory_config=x.memory_config(),
        device=device,
    )
    with expect_error(UnsupportedAxisValue, "multiplier"):
        conv2d_nhwc(x, w, 3, padding=1, groups=32)
