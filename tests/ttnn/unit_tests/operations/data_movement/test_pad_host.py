# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# ttnn.pad on host tensors, checked against a torch reference and against the same call on device.

import math

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.data_movement.pad_test_utils import (
    assert_matches,
    assert_same_spec,
    make_ramp_input as make_input,
    reference_pad,
)

# (input shape, padding as ((before, after), ...) per dim)
RM_PAD_CASES = [
    pytest.param((8,), ((0, 3),), id="rank1_after"),
    pytest.param((8,), ((2, 0),), id="rank1_before"),
    pytest.param((8,), ((2, 3),), id="rank1_both"),
    pytest.param((3, 5), ((0, 0), (0, 2)), id="rank2_last_after"),
    pytest.param((3, 5), ((0, 0), (2, 0)), id="rank2_last_before"),
    pytest.param((3, 5), ((1, 2), (0, 0)), id="rank2_second_last_both"),
    pytest.param((2, 3, 5), ((0, 0), (0, 0), (0, 1)), id="rank3_last_after"),
    pytest.param((2, 3, 5), ((0, 0), (0, 0), (1, 0)), id="rank3_last_before"),
    pytest.param((2, 3, 5), ((0, 0), (1, 1), (0, 0)), id="rank3_second_last_both"),
    pytest.param((2, 3, 5), ((1, 1), (0, 0), (0, 0)), id="rank3_upper_both"),
    pytest.param((1, 1, 2, 4), ((0, 0), (0, 0), (0, 0), (0, 1)), id="rank4_last_after"),
    pytest.param((1, 1, 2, 4), ((0, 0), (0, 0), (0, 0), (1, 0)), id="rank4_last_before"),
    pytest.param((1, 2, 3, 5), ((0, 0), (0, 0), (1, 2), (0, 0)), id="rank4_second_last_both"),
    pytest.param((1, 2, 3, 5), ((0, 0), (1, 1), (0, 0), (0, 0)), id="rank4_upper_both"),
    pytest.param((1, 2, 3, 5), ((1, 0), (0, 1), (2, 1), (1, 2)), id="rank4_all_dims_both"),
    pytest.param((1, 1, 2, 3, 5), ((0, 0),) * 4 + ((0, 1),), id="rank5_last_after"),
    pytest.param((1, 1, 2, 3, 5), ((0, 0),) * 4 + ((1, 0),), id="rank5_last_before"),
    pytest.param((1, 1, 2, 3, 5), ((0, 0),) * 3 + ((1, 1), (0, 0)), id="rank5_second_last_both"),
    pytest.param((1, 1, 2, 3, 5), ((0, 0), (0, 0), (1, 1), (0, 0), (0, 0)), id="rank5_dim2_both"),
]

# Host pad rejects padding below the last three dims of a rank > 4 tensor; device supports it.
RANK5_LEADING_DIM_CASES = [
    pytest.param((2, 1, 2, 3, 4), ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0)), id="rank5_dim0_after"),
    pytest.param((2, 1, 2, 3, 4), ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)), id="rank5_dim1_both"),
]

# (ttnn dtype, pad value)
DTYPE_VALUE_CASES = [
    pytest.param(ttnn.bfloat16, -1.0, id="bfloat16-minus1"),
    pytest.param(ttnn.bfloat16, math.inf, id="bfloat16-inf"),
    pytest.param(ttnn.float32, 0.0, id="float32-0"),
    pytest.param(ttnn.int32, -1, id="int32-minus1"),
    pytest.param(ttnn.uint16, 0, id="uint16-0"),
]


def legacy_args(input_shape, padding):
    output_padded_shape = [before + size + after for size, (before, after) in zip(input_shape, padding)]
    input_tensor_start = [before for before, _ in padding]
    return output_padded_shape, input_tensor_start


# ---------------------------------------------------------------------------------------------
# Host only (no device needed)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("input_shape, padding", RM_PAD_CASES)
@pytest.mark.parametrize("dtype, value", DTYPE_VALUE_CASES)
def test_host_pad_matches_torch(input_shape, padding, dtype, value):
    torch_input = make_input(input_shape, dtype)
    host_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.pad(host_input, padding=padding, value=value)

    assert_matches("host ttnn.pad", output, reference_pad(torch_input, padding, value), torch_input)


@pytest.mark.parametrize("input_shape, padding", RANK5_LEADING_DIM_CASES)
def test_host_pad_rank5_leading_dims_matches_torch(input_shape, padding):
    torch_input = make_input(input_shape, ttnn.bfloat16)
    host_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.pad(host_input, padding=padding, value=-1.0)

    assert_matches("host ttnn.pad rank 5 leading dims", output, reference_pad(torch_input, padding, -1.0), torch_input)


@pytest.mark.parametrize(
    "input_shape, padding",
    [
        pytest.param((1, 1, 2, 4), ((0, 0), (0, 0), (0, 0), (0, 1)), id="rank4_last_after"),
        pytest.param((1, 1, 2, 4), ((0, 0), (0, 0), (1, 1), (1, 1)), id="rank4_both"),
    ],
)
def test_host_pad_legacy_api_matches_torch(input_shape, padding):
    torch_input = make_input(input_shape, ttnn.bfloat16)
    host_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_padded_shape, input_tensor_start = legacy_args(input_shape, padding)

    output = ttnn.pad(host_input, output_padded_shape, input_tensor_start, -1.0)

    assert_matches("host ttnn.pad legacy API", output, reference_pad(torch_input, padding, -1.0), torch_input)


@pytest.mark.parametrize(
    "input_shape, padding",
    [
        pytest.param((1, 1, 2, 4), ((0, 0), (0, 0), (0, 0), (0, 1)), id="rank4_after"),
        pytest.param((1, 1, 2, 4), ((0, 0), (0, 0), (1, 0), (1, 0)), id="rank4_before"),
        pytest.param((2, 3, 5), ((0, 0), (0, 1), (1, 1)), id="rank3_both"),
    ],
)
def test_host_pad_chained_matches_torch(input_shape, padding):
    torch_input = make_input(input_shape, ttnn.bfloat16)
    host_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    first = ttnn.pad(host_input, padding=padding, value=-1.0)
    second = ttnn.pad(first, padding=padding, value=-2.0)

    expected_first = reference_pad(torch_input, padding, -1.0)
    assert_matches("host ttnn.pad, first pad", first, expected_first, torch_input)
    assert_matches("host ttnn.pad, second pad", second, reference_pad(expected_first, padding, -2.0), torch_input)


@pytest.mark.parametrize(
    "input_shape, padding",
    [
        pytest.param((32, 32), ((0, 32), (0, 0)), id="rank2_height"),
        pytest.param((1, 1, 32, 32), ((0, 0), (0, 0), (0, 0), (0, 32)), id="rank4_width"),
        pytest.param((1, 1, 30, 20), ((0, 1), (0, 0), (0, 2), (0, 12)), id="rank4_partial_tiles"),
    ],
)
def test_host_pad_tile_layout_matches_torch(input_shape, padding):
    torch_input = make_input(input_shape, ttnn.bfloat16)
    host_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output = ttnn.pad(host_input, padding=padding, value=-1.0)

    assert_matches("host ttnn.pad TILE", output, reference_pad(torch_input, padding, -1.0), torch_input)


# The whole padded input, placed at input_tensor_start, must fit in the output shape.
def test_host_tensor_pad_input_does_not_fit_raises(expect_error):
    host_input = ttnn.from_torch(make_input((1, 1, 3, 5), ttnn.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT)

    with expect_error(RuntimeError, "does not fit in output padded shape"):
        host_input.pad([1, 1, 3, 5], [0, 0, 1, 1], -1.0)


# ---------------------------------------------------------------------------------------------
# Host vs device
# ---------------------------------------------------------------------------------------------


def run_host_and_device(device, torch_input, dtype, layout, pad_fn):
    host_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout)
    device_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device)
    return pad_fn(host_input), pad_fn(device_input)


def check_host_and_device(host_output, device_output, expected, torch_input, op="ttnn.pad"):
    failures = []
    for label, tensor in (("host", host_output), ("device", device_output)):
        try:
            assert_matches(f"{label} {op} vs torch", tensor, expected, torch_input)
        except AssertionError as error:
            failures.append(str(error))
    try:
        assert_same_spec(host_output, device_output)
    except AssertionError as error:
        failures.append(f"[host vs device spec]\n{error}")
    assert not failures, "\n\n".join(failures)


@pytest.mark.parametrize("input_shape, padding", RM_PAD_CASES)
@pytest.mark.parametrize("dtype, value", DTYPE_VALUE_CASES)
def test_host_pad_matches_device(device, input_shape, padding, dtype, value):
    torch_input = make_input(input_shape, dtype)

    host_output, device_output = run_host_and_device(
        device, torch_input, dtype, ttnn.ROW_MAJOR_LAYOUT, lambda t: ttnn.pad(t, padding=padding, value=value)
    )

    check_host_and_device(host_output, device_output, reference_pad(torch_input, padding, value), torch_input)


@pytest.mark.parametrize("input_shape, padding", RANK5_LEADING_DIM_CASES)
def test_host_pad_rank5_leading_dims_matches_device(device, input_shape, padding):
    torch_input = make_input(input_shape, ttnn.bfloat16)
    expected = reference_pad(torch_input, padding, -1.0)

    device_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    device_output = ttnn.pad(device_input, padding=padding, value=-1.0)
    assert_matches("device ttnn.pad rank 5 leading dims", device_output, expected, torch_input)

    host_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    host_output = ttnn.pad(host_input, padding=padding, value=-1.0)
    check_host_and_device(host_output, device_output, expected, torch_input)


@pytest.mark.parametrize(
    "input_shape, padding",
    [
        pytest.param((1, 1, 2, 4), ((0, 0), (0, 0), (0, 0), (0, 1)), id="rank4_last_after"),
        pytest.param((1, 1, 2, 4), ((0, 0), (0, 0), (1, 1), (1, 1)), id="rank4_both"),
        pytest.param((1, 3, 4, 6), ((1, 0), (0, 1), (2, 0), (0, 2)), id="rank4_all_dims"),
    ],
)
def test_host_pad_legacy_api_matches_device(device, input_shape, padding):
    torch_input = make_input(input_shape, ttnn.bfloat16)
    output_padded_shape, input_tensor_start = legacy_args(input_shape, padding)

    host_output, device_output = run_host_and_device(
        device,
        torch_input,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        lambda t: ttnn.pad(t, output_padded_shape, input_tensor_start, -1.0),
    )

    check_host_and_device(host_output, device_output, reference_pad(torch_input, padding, -1.0), torch_input)


@pytest.mark.parametrize(
    "input_shape, padding",
    [
        pytest.param((1, 1, 2, 4), ((0, 0), (0, 0), (0, 0), (0, 1)), id="rank4_after"),
        pytest.param((1, 1, 2, 4), ((0, 0), (0, 0), (1, 0), (1, 0)), id="rank4_before"),
        pytest.param((2, 3, 5), ((0, 0), (0, 1), (1, 1)), id="rank3_both"),
    ],
)
def test_host_pad_chained_matches_device(device, input_shape, padding):
    torch_input = make_input(input_shape, ttnn.bfloat16)

    def pad_twice(tensor):
        return ttnn.pad(ttnn.pad(tensor, padding=padding, value=-1.0), padding=padding, value=-2.0)

    host_output, device_output = run_host_and_device(
        device, torch_input, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, pad_twice
    )

    expected = reference_pad(reference_pad(torch_input, padding, -1.0), padding, -2.0)
    check_host_and_device(host_output, device_output, expected, torch_input)


@pytest.mark.parametrize(
    "input_shape, padding",
    [
        pytest.param((32, 32), ((0, 32), (0, 0)), id="rank2_height"),
        pytest.param((1, 1, 32, 32), ((0, 0), (0, 0), (0, 0), (0, 32)), id="rank4_width"),
        pytest.param((1, 1, 30, 20), ((0, 1), (0, 0), (0, 2), (0, 12)), id="rank4_partial_tiles"),
        pytest.param((2, 3, 32, 64), ((0, 0), (0, 1), (0, 32), (0, 0)), id="rank4_upper_and_height"),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_host_pad_tile_layout_matches_device(device, input_shape, padding, dtype):
    torch_input = make_input(input_shape, dtype)

    host_output, device_output = run_host_and_device(
        device, torch_input, dtype, ttnn.TILE_LAYOUT, lambda t: ttnn.pad(t, padding=padding, value=-1.0)
    )

    check_host_and_device(host_output, device_output, reference_pad(torch_input, padding, -1.0), torch_input)


@pytest.mark.parametrize(
    "input_shape, start, end",
    [
        pytest.param((1, 1, 5, 5), (0, 0, 1, 1), (1, 1, 4, 4), id="rank4_inner"),
        pytest.param((2, 3, 4, 6), (1, 0, 1, 2), (2, 2, 4, 5), id="rank4_all_dims"),
        pytest.param((3, 4, 6), (1, 1, 0), (3, 3, 4), id="rank3"),
    ],
)
def test_host_unpad_matches_device_slice(device, input_shape, start, end):
    torch_input = make_input(input_shape, ttnn.bfloat16)
    expected = torch_input[tuple(slice(s, e) for s, e in zip(start, end))].to(torch.float64)

    host_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    host_output = host_input.unpad(list(start), list(end))
    device_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    device_output = ttnn.slice(device_input, list(start), list(end))

    check_host_and_device(
        host_output, device_output, expected, torch_input, op="Tensor.unpad (host) / ttnn.slice (device)"
    )
