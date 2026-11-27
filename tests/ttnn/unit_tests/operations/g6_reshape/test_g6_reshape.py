# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_reshape, Phase 0 (bfloat16, TILE, DRAM interleaved).

The cases are the bfloat16 TILE production cases of the golden INPUTS plus the cases the spec's rules
and the design require (BB-22): two pins of the `face_row_assemble` regime (FLUX geometry in bfloat16,
since no bfloat16 production shape reaches that regime), special values, NaN-poisoned input padding,
the -1 entry, a tuple shape, repeated calls with different targets, and the host validation errors.
Production has no single-tile bfloat16 case, so none is added.

The op moves bytes only, so every comparison is bit for bit against torch.reshape of the input as it
sits on the device (read back through the same ttnn.to_torch path as the output).
"""

import pytest
import torch

import ttnn
from ttnn.operations.g6_reshape import g6_reshape


# (input shape, target shape): Qwen production cases, bfloat16 TILE, all `face_row_direct`.
PRODUCTION_BF16_TILE = [
    ((1, 16, 6, 256), (1, 1, 16, 1536)),  # Qwen #349
    ((16, 1, 64), (1, 1, 16, 64)),  # Qwen #353
    ((16, 1, 1536), (1, 1, 16, 1536)),  # Qwen #357
    ((1, 1, 16, 4608), (16, 1, 4608)),  # Qwen #364
    ((1, 16, 64), (16, 1, 64)),  # Qwen #367
    ((16, 1, 5120), (1, 1, 16, 5120)),  # Qwen #370
    ((2, 1, 5120), (1, 1, 2, 5120)),  # Qwen #382
    ((1, 8, 6, 256), (1, 1, 8, 1536)),  # Qwen #490
    ((8, 1, 64), (1, 1, 8, 64)),  # Qwen #494
    ((8, 1, 1536), (1, 1, 8, 1536)),  # Qwen #498
    ((1, 1, 8, 4608), (8, 1, 4608)),  # Qwen #505
    ((1, 8, 64), (8, 1, 64)),  # Qwen #508
    ((8, 1, 5120), (1, 1, 8, 5120)),  # Qwen #511
    ((1, 1, 6, 256), (1, 1, 1536)),  # Qwen #553
]

# W_out = 1 and W_in not equal to W_out: every output face row is assembled from element segments.
ASSEMBLE_REGIME_PINS = [
    ((1, 185), (1, 1, 185, 1)),  # FLUX #168 geometry
    ((1, 512), (1, 1, 512, 1)),  # FLUX #78 geometry
]

# Every host validation error of the op names the op first.
HOST_ERROR_PREFIX = "g6_reshape:"

BF16_SPECIALS = [float("nan"), float("inf"), float("-inf"), -0.0, 9.18e-41, 3.0e38]


def _case_id(case):
    input_shape, shape = case
    return f"{'x'.join(map(str, input_shape))}-to-{'x'.join(map(str, shape))}"


def _bits(t):
    return t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def _randn_bf16(shape):
    torch.manual_seed(42)
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def _special_bf16(shape):
    """randn with the last mantissa bit set, and NaN / inf / -0.0 / a denormal spread over the tensor."""
    flat = _randn_bf16(shape).reshape(-1)
    flat = (flat.view(torch.int16) | 1).view(torch.bfloat16).clone()
    n = flat.numel()
    stride = max(n // len(BF16_SPECIALS), 1)
    for i, v in enumerate(BF16_SPECIALS):
        flat[min(i * stride, n - 1)] = v
    return flat.reshape(shape)


def _to_device(torch_input, device, *, poison=None):
    tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )
    if poison is not None:
        tensor = ttnn.fill_implicit_tile_padding(tensor, poison)
    return tensor


def _resolved(shape, numel):
    shape = [int(s) for s in shape]
    if -1 in shape:
        known = 1
        for s in shape:
            if s != -1:
                known *= s
        shape[shape.index(-1)] = numel // known
    return shape


def _check(ttnn_input, ttnn_output, shape):
    device_input = ttnn.to_torch(ttnn_input)
    expected = torch.reshape(device_input, tuple(int(s) for s in shape))
    assert list(ttnn_output.shape) == _resolved(shape, device_input.numel())
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.TILE_LAYOUT
    assert ttnn_output.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert ttnn_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    got = ttnn.to_torch(ttnn_output)
    assert list(got.shape) == list(expected.shape)
    mismatch = (_bits(got) != _bits(expected)).reshape(-1)
    assert not mismatch.any(), (
        f"{int(mismatch.sum())} of {mismatch.numel()} values differ, first at linear index "
        f"{int(mismatch.nonzero()[0])}"
    )
    return got


@pytest.mark.parametrize("case", PRODUCTION_BF16_TILE + ASSEMBLE_REGIME_PINS, ids=_case_id)
def test_g6_reshape(device, case):
    input_shape, shape = case
    torch_input = _randn_bf16(input_shape)
    ttnn_input = _to_device(torch_input, device)

    ttnn_output = g6_reshape(ttnn_input, list(shape))

    got = _check(ttnn_input, ttnn_output, shape)
    assert torch.equal(_bits(got), _bits(torch.reshape(torch_input, shape)))


@pytest.mark.parametrize(
    "case",
    [((1, 16, 6, 256), (1, 1, 16, 1536)), ((1, 185), (1, 1, 185, 1))],
    ids=_case_id,
)
def test_g6_reshape_special_values(device, case):
    input_shape, shape = case
    ttnn_input = _to_device(_special_bf16(input_shape), device)

    _check(ttnn_input, g6_reshape(ttnn_input, list(shape)), shape)


@pytest.mark.parametrize(
    "case",
    [
        ((16, 1, 64), (1, 1, 16, 64)),
        ((1, 16, 6, 256), (1, 1, 16, 1536)),
        ((1, 1, 16, 4608), (16, 1, 4608)),
        ((1, 185), (1, 1, 185, 1)),
    ],
    ids=_case_id,
)
def test_g6_reshape_poisoned_padding(device, case):
    input_shape, shape = case
    ttnn_input = _to_device(_randn_bf16(input_shape), device, poison=float("nan"))

    got = _check(ttnn_input, g6_reshape(ttnn_input, list(shape)), shape)
    assert not torch.isnan(got).any()


def test_g6_reshape_inferred_entry(device):
    ttnn_input = _to_device(_randn_bf16((16, 1, 1536)), device)

    ttnn_output = g6_reshape(ttnn_input, [1, 1, -1, 1536])

    assert list(ttnn_output.shape) == [1, 1, 16, 1536]
    _check(ttnn_input, ttnn_output, [1, 1, -1, 1536])


def test_g6_reshape_tuple_shape(device):
    ttnn_input = _to_device(_randn_bf16((1, 1, 16, 4608)), device)

    _check(ttnn_input, g6_reshape(ttnn_input, (16, 1, 4608)), (16, 1, 4608))


def test_g6_reshape_repeated_calls(device):
    # Two targets with different page maps on one input, then the first again: a cache key that
    # missed the target shape would replay the wrong program, and a repeat must be bit-identical.
    ttnn_input = _to_device(_randn_bf16((16, 1, 64)), device)

    first = _check(ttnn_input, g6_reshape(ttnn_input, [1, 1, 16, 64]), [1, 1, 16, 64])
    _check(ttnn_input, g6_reshape(ttnn_input, [1, 1, 1, 1024]), [1, 1, 1, 1024])
    again = _check(ttnn_input, g6_reshape(ttnn_input, [1, 1, 16, 64]), [1, 1, 16, 64])

    assert torch.equal(_bits(first), _bits(again))


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 16, 100],  # product differs from the element count
        [16, 0, 1536],  # entry of 0
        [-2, 12288],  # entry below -1
        [-1, -1, 1536],  # more than one -1
        [1, 1, 1, 16, 1536],  # output rank 5
        [],  # output rank 0
    ],
    ids=["bad-product", "zero-entry", "below-minus-one", "two-minus-one", "rank-5", "rank-0"],
)
def test_g6_reshape_invalid_shape_raises(device, expect_error, shape):
    ttnn_input = _to_device(_randn_bf16((16, 1, 1536)), device)

    with expect_error((ValueError, RuntimeError), HOST_ERROR_PREFIX):
        g6_reshape(ttnn_input, shape)


def test_g6_reshape_input_rank_5_raises(device, expect_error):
    ttnn_input = _to_device(_randn_bf16((1, 16, 1, 1, 64)), device)

    with expect_error((ValueError, RuntimeError), HOST_ERROR_PREFIX):
        g6_reshape(ttnn_input, [16, 64])
