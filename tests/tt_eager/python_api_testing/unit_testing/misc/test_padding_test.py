# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest


import torch

import ttnn
from models.common.utility_functions import nearest_32
from tests.ttnn.unit_tests.operations.data_movement.pad_test_utils import (
    assert_matches,
    assert_physical_matches,
)


def reference_pad_at(inp, output_tensor_shape, input_tensor_start, pad_value):
    a_ref = torch.ones(*output_tensor_shape, dtype=inp.dtype) * pad_value
    a_ref[tuple(slice(start, start + size) for start, size in zip(input_tensor_start, inp.shape))] = inp
    return a_ref


def reference_unpad(inp, output_tensor_start, output_tensor_end):
    return inp[tuple(slice(start, end) for start, end in zip(output_tensor_start, output_tensor_end))]


@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_shape, input_tensor_start, pad_value",
    (
        ((1, 1, 3, 3), (1, 1, 5, 5), (0, 0, 1, 1), 0),
        ((1, 1, 3, 3), (2, 2, 5, 5), (0, 0, 0, 0), -1),
        ((1, 3, 30, 30), (1, 3, 32, 32), (0, 0, 0, 0), 1),
        ((1, 3, 30, 30), (3, 5, 32, 32), (1, 2, 0, 0), torch.inf),
        ((1, 3, 30, 30), (3, 3, 64, 64), (0, 0, 31, 31), -torch.inf),
    ),
)
def test_run_padding_test(input_tensor_shape, output_tensor_shape, input_tensor_start, pad_value):
    inp = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)

    # Create tensor on host
    a = ttnn.Tensor(inp, ttnn.bfloat16)

    # Pad inputs on host
    a_pad = a.pad(output_tensor_shape, input_tensor_start, pad_value)

    # Tensor.pad pads the physical data only: padded shape grows, logical shape stays the input's.
    assert_physical_matches(
        "Tensor.pad",
        a_pad,
        reference_pad_at(inp, output_tensor_shape, input_tensor_start, pad_value),
        expected_logical_shape=input_tensor_shape,
    )


@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_shape, input_tensor_start",
    (
        pytest.param((8,), (11,), (2,), id="rank1"),
        pytest.param((3, 5), (4, 7), (1, 1), id="rank2"),
        pytest.param((2, 3, 5), (3, 4, 6), (1, 0, 1), id="rank3"),
        pytest.param((1, 1, 2, 3, 5), (1, 2, 3, 4, 6), (0, 1, 1, 0, 1), id="rank5"),
    ),
)
def test_run_padding_test_ranks(input_tensor_shape, output_tensor_shape, input_tensor_start):
    inp = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)
    a = ttnn.from_torch(inp, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    a_pad = a.pad(output_tensor_shape, input_tensor_start, -1.0)

    assert_physical_matches(
        "Tensor.pad",
        a_pad,
        reference_pad_at(inp, output_tensor_shape, input_tensor_start, -1.0),
        expected_logical_shape=input_tensor_shape,
    )


@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_start, output_tensor_end",
    (
        ((1, 1, 5, 5), (0, 0, 1, 1), (1, 1, 4, 4)),
        ((2, 2, 5, 5), (0, 0, 0, 0), (1, 1, 3, 3)),
        ((1, 3, 32, 32), (0, 0, 0, 0), (1, 3, 30, 30)),
        ((3, 5, 32, 32), (1, 2, 0, 0), (2, 5, 30, 30)),
        ((3, 3, 64, 64), (0, 0, 32, 32), (1, 3, 62, 62)),
    ),
)
def test_run_unpadding_test(input_tensor_shape, output_tensor_start, output_tensor_end):
    inp = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)

    # Create tensor on host
    a = ttnn.from_torch(inp.reshape(input_tensor_shape), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Unpad inputs on host
    a_unpad = a.unpad(output_tensor_start, output_tensor_end)

    a_ref = reference_unpad(inp, output_tensor_start, output_tensor_end)
    assert list(a_unpad.padded_shape) == list(a_ref.shape), f"padded_shape {list(a_unpad.padded_shape)}"
    assert_matches("Tensor.unpad", a_unpad, a_ref, inp)


@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_start, output_tensor_end",
    (
        pytest.param((7,), (2,), (5,), id="rank1"),
        pytest.param((5, 6), (1, 2), (4, 5), id="rank2"),
        pytest.param((3, 4, 5), (1, 0, 2), (3, 4, 4), id="rank3"),
        pytest.param((2, 2, 3, 4, 5), (0, 1, 1, 0, 2), (2, 2, 3, 3, 5), id="rank5"),
        pytest.param((1, 1, 5, 5), (0, 0, 2, 1), (1, 1, 2, 4), id="zero_size"),
        pytest.param((1, 2, 3, 4), (0, 0, 0, 0), (1, 2, 3, 4), id="full_range"),
    ),
)
def test_run_unpadding_test_ranks(input_tensor_shape, output_tensor_start, output_tensor_end):
    inp = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)
    a = ttnn.from_torch(inp, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    a_unpad = a.unpad(output_tensor_start, output_tensor_end)

    assert_matches("Tensor.unpad", a_unpad, reference_unpad(inp, output_tensor_start, output_tensor_end), inp)


def docstring_unpad_input():
    return torch.tensor(
        [[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0], [0, 0, 0, 0, 0]],
        dtype=torch.bfloat16,
    ).reshape(1, 1, 5, 5)


def test_unpad_docstring_example():
    """The Tensor.unpad docstring example: start [0,0,1,1], exclusive end [1,1,4,4]."""
    inp = docstring_unpad_input()
    a = ttnn.from_torch(inp, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    a_unpad = a.unpad([0, 0, 1, 1], [1, 1, 4, 4])

    documented_output = torch.arange(1, 10, dtype=torch.bfloat16).reshape(1, 1, 3, 3)
    assert_matches("Tensor.unpad docstring example", a_unpad, documented_output, inp)


@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_shape, input_tensor_start",
    (
        pytest.param((1, 2, 3, 4), (2, 4, 6, 8), (1, 1, 2, 3), id="rank4_before_and_after"),
        pytest.param((2, 3, 5), (3, 4, 6), (0, 1, 1), id="rank3"),
    ),
)
def test_run_padding_then_unpadding_round_trip(input_tensor_shape, output_tensor_shape, input_tensor_start):
    inp = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)
    a = ttnn.from_torch(inp, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor_end = tuple(start + size for start, size in zip(input_tensor_start, input_tensor_shape))

    a_round_trip = a.pad(output_tensor_shape, input_tensor_start, -1.0).unpad(input_tensor_start, output_tensor_end)

    assert_matches("Tensor.pad then Tensor.unpad", a_round_trip, inp, inp)


# Pad, run op, unpad
@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_shape, input_tensor_start, pad_value",
    (((1, 1, 3, 4), (1, 1, 32, 32), (0, 0, 1, 1), 0),),
)
def test_run_padding_and_add_test(input_tensor_shape, output_tensor_shape, input_tensor_start, pad_value, device):
    # Args for unpad
    output_tensor_start = input_tensor_start
    output_tensor_end = tuple(input_tensor_start[i] + input_tensor_shape[i] for i in range(len(input_tensor_shape)))

    inp = torch.rand(*input_tensor_shape)
    ones = torch.ones(*input_tensor_shape)

    # Tensor.pad() preserves the original logical shape while only setting
    # padded_shape, which causes data at non-zero offsets to be lost during
    # tile-layout conversion. Build the padded tensors in PyTorch so the
    # logical shape equals the full output shape.
    inp_padded = torch.full(output_tensor_shape, pad_value, dtype=torch.float32)
    inp_padded[
        input_tensor_start[0] : output_tensor_end[0],
        input_tensor_start[1] : output_tensor_end[1],
        input_tensor_start[2] : output_tensor_end[2],
        input_tensor_start[3] : output_tensor_end[3],
    ] = inp

    ones_padded = torch.full(output_tensor_shape, pad_value, dtype=torch.float32)
    ones_padded[
        input_tensor_start[0] : output_tensor_end[0],
        input_tensor_start[1] : output_tensor_end[1],
        input_tensor_start[2] : output_tensor_end[2],
        input_tensor_start[3] : output_tensor_end[3],
    ] = ones

    a_pad = ttnn.from_torch(inp_padded.reshape(output_tensor_shape), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_pad = ttnn.from_torch(ones_padded.reshape(output_tensor_shape), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Run add op on device with padded tensors
    a_dev = a_pad.to(ttnn.TILE_LAYOUT).to(device)
    b_dev = b_pad.to(ttnn.TILE_LAYOUT).to(device)
    out_dev = ttnn.add(a_dev, b_dev)
    out_pad = out_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    # Unpad out to get result
    out = out_pad.unpad(output_tensor_start, output_tensor_end)
    out_pt = out.to_torch().to(torch.float32)

    out_ref = inp + ones

    assert torch.allclose(out_pt, out_ref, rtol=1e-2), f"got\n{out_pt}\nexpected\n{out_ref}"


@pytest.mark.parametrize(
    "input_tensor_shape,  pad_value",
    (
        ((1, 1, 3, 3), 0),
        ((2, 2, 5, 5), -1),
        ((1, 3, 30, 30), 1),
        ((3, 5, 32, 32), torch.inf),
        ((3, 3, 66, 66), -torch.inf),
    ),
)
def test_run_tile_padding_test(input_tensor_shape, pad_value):
    output_tensor_shape = (
        *input_tensor_shape[:-2],
        nearest_32(input_tensor_shape[-2]),
        nearest_32(input_tensor_shape[-1]),
    )
    inp = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)

    # Create tensor on host
    a = ttnn.from_torch(inp.reshape(input_tensor_shape), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Pad inputs on host
    a_pad = a.pad_to_tile(pad_value)

    assert_physical_matches(
        "Tensor.pad_to_tile",
        a_pad,
        reference_pad_at(inp, output_tensor_shape, (0,) * len(input_tensor_shape), pad_value),
        expected_logical_shape=input_tensor_shape,
    )


@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_shape",
    (
        ((1, 1, 32, 32), (1, 1, 4, 4)),
        ((2, 2, 32, 32), (2, 2, 32, 32)),
        ((1, 3, 64, 64), (1, 3, 33, 35)),
        ((3, 5, 32, 64), (3, 5, 31, 64)),
        ((3, 3, 64, 128), (3, 3, 64, 121)),
    ),
)
def test_run_tile_unpadding_test(input_tensor_shape, output_tensor_shape):
    inp = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)

    # Create tensor on host
    a = ttnn.from_torch(inp.reshape(input_tensor_shape), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Unpad inputs on host
    a_unpad = a.unpad_from_tile(output_tensor_shape)

    a_ref = reference_unpad(inp, (0,) * len(output_tensor_shape), output_tensor_shape)
    assert_matches("Tensor.unpad_from_tile", a_unpad, a_ref, inp)


# Pad, run op, unpad
@pytest.mark.parametrize(
    "input_tensor_shape, pad_value",
    (((1, 1, 3, 4), 0),),
)
def test_run_tile_padding_and_add_test(input_tensor_shape, pad_value, device):
    inp = torch.rand(*input_tensor_shape)
    ones = torch.ones(*input_tensor_shape)

    # Create tensor on host
    a = ttnn.from_torch(inp.reshape(input_tensor_shape), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    b = ttnn.from_torch(ones.reshape(input_tensor_shape), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Pad inputs on host
    a_pad = a.pad_to_tile(pad_value)
    b_pad = b.pad_to_tile(pad_value)

    a_dev = a_pad.to(ttnn.TILE_LAYOUT).to(device)
    b_dev = b_pad.to(ttnn.TILE_LAYOUT).to(device)
    out_dev = ttnn.add(a_dev, b_dev)
    out_pad = out_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    # to(ROW_MAJOR_LAYOUT) already unpadded; use directly.
    out_pt = out_pad.to_torch().to(torch.float32)

    out_ref = inp + ones

    assert torch.allclose(out_pt, out_ref, rtol=1e-2), f"got\n{out_pt}\nexpected\n{out_ref}"
