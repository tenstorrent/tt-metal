# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest


import torch

import ttnn
from models.utility_functions import nearest_32


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
    a_pt = a_pad.to_torch()

    # Pytorch reference
    input_tensor_end = tuple(input_tensor_start[i] + input_tensor_shape[i] for i in range(len(input_tensor_shape)))
    a_ref = torch.ones(*output_tensor_shape, dtype=torch.bfloat16) * pad_value
    a_ref[
        input_tensor_start[0] : input_tensor_end[0],
        input_tensor_start[1] : input_tensor_end[1],
        input_tensor_start[2] : input_tensor_end[2],
        input_tensor_start[3] : input_tensor_end[3],
    ] = inp

    # print("\n", a_pt.shape)
    # print("\n", a_pt)
    # print("\n", a_ref)

    assert a_pt.shape == output_tensor_shape
    eq = torch.equal(a_pt, a_ref)
    assert eq


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
    a = ttnn.Tensor(
        inp.reshape(-1).tolist(),
        input_tensor_shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )

    # Unpad inputs on host
    output_tensor_shape = tuple(output_tensor_end[i] - output_tensor_start[i] for i in range(len(input_tensor_shape)))
    a_unpad = a.unpad(output_tensor_start, output_tensor_end)
    a_pt = a_unpad.to_torch()

    # Pytorch reference
    a_ref = inp[
        output_tensor_start[0] : output_tensor_end[0],
        output_tensor_start[1] : output_tensor_end[1],
        output_tensor_start[2] : output_tensor_end[2],
        output_tensor_start[3] : output_tensor_end[3],
    ]

    # print("\n", a_pt.shape)
    # print("\n", a_pt)
    # print("\n", a_ref)

    assert a_pt.shape == output_tensor_shape
    eq = torch.equal(a_pt, a_ref)
    assert eq


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

    # Create tensor on host
    a = ttnn.Tensor(
        inp.reshape(-1).tolist(),
        input_tensor_shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    b = ttnn.Tensor(
        ones.reshape(-1).tolist(),
        input_tensor_shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )

    # Pad inputs on host
    a_pad = a.pad(output_tensor_shape, input_tensor_start, pad_value)
    b_pad = b.pad(output_tensor_shape, input_tensor_start, pad_value)

    # Run add op on device with padded tensors

    a_dev = a_pad.to(ttnn.TILE_LAYOUT).to(device)
    b_dev = b_pad.to(ttnn.TILE_LAYOUT).to(device)
    out_dev = ttnn.add(a_dev, b_dev)
    out_pad = out_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    # Unpad out to get result
    out = out_pad.unpad(output_tensor_start, output_tensor_end)
    out_pt = out.to_torch().to(torch.float32)

    out_ref = inp + ones

    # print("\n", out_pt)
    # print("\n", out_ref)

    passing = torch.allclose(out_pt, out_ref, rtol=1e-2)
    assert passing


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
    a = ttnn.Tensor(
        inp.reshape(-1).tolist(),
        input_tensor_shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )

    # Pad inputs on host
    a_pad = a.pad_to_tile(pad_value)
    a_pt = a_pad.to_torch()

    # Pytorch reference
    input_tensor_end = tuple(input_tensor_shape[i] for i in range(len(input_tensor_shape)))
    a_ref = torch.ones(*output_tensor_shape, dtype=torch.bfloat16) * pad_value
    a_ref[
        0 : input_tensor_end[0],
        0 : input_tensor_end[1],
        0 : input_tensor_end[2],
        0 : input_tensor_end[3],
    ] = inp

    # print("\n", a_pt.shape)
    # print("\n", a_pt)
    # print("\n", a_ref)

    assert a_pt.shape == output_tensor_shape
    eq = torch.equal(a_pt, a_ref)
    assert eq


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
    a = ttnn.Tensor(
        inp.reshape(-1).tolist(),
        input_tensor_shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )

    # Unpad inputs on host
    a_unpad = a.unpad_from_tile(output_tensor_shape)
    a_pt = a_unpad.to_torch()

    # Pytorch reference
    a_ref = inp[
        0 : output_tensor_shape[0],
        0 : output_tensor_shape[1],
        0 : output_tensor_shape[2],
        0 : output_tensor_shape[3],
    ]

    # print("\n", a_pt.shape)
    # print("\n", a_pt)
    # print("\n", a_ref)

    assert a_pt.shape == output_tensor_shape
    eq = torch.equal(a_pt, a_ref)
    assert eq


# Pad, run op, unpad
@pytest.mark.parametrize(
    "input_tensor_shape, pad_value",
    (((1, 1, 3, 4), 0),),
)
def test_run_tile_padding_and_add_test(input_tensor_shape, pad_value, device):
    inp = torch.rand(*input_tensor_shape)
    ones = torch.ones(*input_tensor_shape)

    # Create tensor on host
    a = ttnn.Tensor(
        inp.reshape(-1).tolist(),
        input_tensor_shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    b = ttnn.Tensor(
        ones.reshape(-1).tolist(),
        input_tensor_shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )

    # Pad inputs on host
    a_pad = a.pad_to_tile(pad_value)
    b_pad = b.pad_to_tile(pad_value)

    a_dev = a_pad.to(ttnn.TILE_LAYOUT).to(device)
    b_dev = b_pad.to(ttnn.TILE_LAYOUT).to(device)
    out_dev = ttnn.add(a_dev, b_dev)
    out_pad = out_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    # Unpad out to get result
    out = out_pad.unpad_from_tile(input_tensor_shape)
    out_pt = out.to_torch().to(torch.float32)

    out_ref = inp + ones

    # print("\n", out_pt)
    # print("\n", out_ref)

    passing = torch.allclose(out_pt, out_ref, rtol=1e-2)
    assert passing
