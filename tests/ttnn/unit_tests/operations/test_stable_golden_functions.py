# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


OWNED_OPERATIONS = [
    "assign",
    "divide",
    "conv1d",
    "conv_transpose2d",
    "copy",
    "reshape_on_device",
    "unsqueeze_to_4D",
    "view",
    "broadcast",
    "chunk",
    "expand",
    "index_fill",
    "indexed_fill",
    "narrow",
    "roll",
    "scatter",
    "scatter_add",
    "slice",
    "split",
    "squeeze",
    "stack",
    "tilize_with_val_padding",
    "tilize_with_zero_padding",
    "tosa_gather",
    "tosa_scatter",
    "transpose",
    "unsqueeze",
    "untilize",
    "untilize_with_unpadding",
    "fold",
    "cumprod",
    "cumsum",
    "nonzero",
    "prod",
    "sort",
    "std_hw",
    "var_hw",
    "ema",
    "i1",
    "embedding_bw",
    "prod_bw",
    "complex_tensor",
    "snake_beta",
]


@pytest.mark.parametrize("operation_name", OWNED_OPERATIONS)
def test_owned_operation_has_golden_function(operation_name):
    assert callable(ttnn.get_golden_function(getattr(ttnn, operation_name)))


def test_binary_and_core_golden_contracts():
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    denominator = torch.full_like(source, 2)
    torch.testing.assert_close(ttnn.get_golden_function(ttnn.divide)(source, denominator), source / denominator)

    destination = torch.zeros_like(source)
    assigned = ttnn.get_golden_function(ttnn.assign)(input_a=source, input_b=destination)
    assert assigned is destination
    torch.testing.assert_close(assigned, source)
    assert (
        ttnn.get_golden_function(ttnn.assign)(input_tensor=source.to(torch.bfloat16), dtype=ttnn.float32).dtype
        == torch.float32
    )

    destination.zero_()
    copied = ttnn.get_golden_function(ttnn.copy)(source, destination)
    assert copied is destination
    torch.testing.assert_close(copied, source)

    reshaped = ttnn.get_golden_function(ttnn.reshape_on_device)(source, 1, 1, 2, 3)
    assert reshaped.shape == (1, 1, 2, 3)
    assert ttnn.get_golden_function(ttnn.view)(source, (3, 2)).shape == (3, 2)
    assert ttnn.get_golden_function(ttnn.unsqueeze_to_4D)(source).shape == (1, 1, 2, 3)


def test_standard_data_movement_golden_contracts():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    index = torch.tensor([0, 2])

    torch.testing.assert_close(ttnn.get_golden_function(ttnn.broadcast)(x, None), x)
    assert [tensor.shape for tensor in ttnn.get_golden_function(ttnn.chunk)(x, 2, 1)] == [
        torch.Size([2, 2, 4]),
        torch.Size([2, 1, 4]),
    ]
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.expand)(torch.ones(1, 3, 1), (2, 3, 4)),
        torch.ones(2, 3, 4),
    )
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.index_fill)(x, 1, index, -1),
        x.index_fill(1, index, -1),
    )
    torch.testing.assert_close(ttnn.get_golden_function(ttnn.narrow)(x, 1, 1, 2), x[:, 1:3])
    torch.testing.assert_close(ttnn.get_golden_function(ttnn.roll)(x, (1, -1), (0, 2)), torch.roll(x, (1, -1), (0, 2)))
    torch.testing.assert_close(ttnn.get_golden_function(ttnn.slice)(x, [0, 1, 0], [2, 3, 4], [1, 1, 2]), x[:, 1:3, ::2])
    assert [tensor.shape for tensor in ttnn.get_golden_function(ttnn.split)(x, [1, 2], dim=1)] == [
        torch.Size([2, 1, 4]),
        torch.Size([2, 2, 4]),
    ]
    assert ttnn.get_golden_function(ttnn.squeeze)(torch.ones(1, 2, 1), [0, 2]).shape == (2,)
    assert ttnn.get_golden_function(ttnn.stack)([x, x], 1).shape == (2, 2, 3, 4)
    torch.testing.assert_close(ttnn.get_golden_function(ttnn.transpose)(x, 0, 2), x.transpose(0, 2))
    assert ttnn.get_golden_function(ttnn.unsqueeze)(x, -1).shape == (2, 3, 4, 1)


def test_scatter_and_indexed_fill_golden_contracts():
    base = torch.zeros(2, 3)
    index = torch.tensor([[0, 1, 1], [2, 0, 2]])
    src = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.scatter)(base, 1, index, src),
        torch.scatter(base, 1, index, src),
    )
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.scatter_add)(base, 1, index, src),
        torch.scatter_add(base, 1, index, src),
    )

    input_a = torch.zeros(4, 2)
    input_b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batch_ids = torch.tensor([[[[2, 1, 2]]]])
    expected = input_a.clone()
    expected[2] = input_b[2]
    expected[1] = input_b[1]
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.indexed_fill)(batch_ids, input_a, input_b),
        expected,
    )


def test_layout_and_tosa_data_movement_golden_contracts():
    x = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    padded = ttnn.get_golden_function(ttnn.tilize_with_val_padding)(x, (1, 4, 4), -1)
    expected = torch.full((1, 4, 4), -1.0)
    expected[:, :2, :3] = x
    torch.testing.assert_close(padded, expected)
    torch.testing.assert_close(ttnn.get_golden_function(ttnn.tilize_with_zero_padding)(x), x)
    torch.testing.assert_close(ttnn.get_golden_function(ttnn.untilize)(x), x)
    torch.testing.assert_close(ttnn.get_golden_function(ttnn.untilize_with_unpadding)(x, [0, 1, 1]), x[..., :2])

    values = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
    indices = torch.tensor([[0, 3], [4, 1]])
    expanded_indices = indices.unsqueeze(-1).expand(2, 2, 3)
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.tosa_gather)(values, indices),
        torch.gather(values, 1, expanded_indices),
    )
    updates = torch.full((2, 2, 3), -1.0)
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.tosa_scatter)(values, indices, updates),
        torch.scatter(values, 1, expanded_indices, updates),
    )


def test_conv_golden_contracts():
    input_1d = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    weight_1d = torch.ones(3, 2, 3)
    output_1d = ttnn.get_golden_function(ttnn.conv1d)(
        input_tensor=input_1d,
        weight_tensor=weight_1d,
        in_channels=2,
        out_channels=3,
        batch_size=1,
        input_length=5,
        kernel_size=3,
        padding=1,
    )
    expected_1d = torch.nn.functional.conv1d(input_1d.permute(0, 2, 1), weight_1d, padding=1)
    torch.testing.assert_close(output_1d.reshape(1, 5, 3).permute(0, 2, 1), expected_1d)

    input_2d = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    weight_2d = torch.ones(2, 3, 2, 2)
    output_2d = ttnn.get_golden_function(ttnn.conv_transpose2d)(
        input_tensor=input_2d,
        weight_tensor=weight_2d,
        in_channels=2,
        out_channels=3,
        batch_size=1,
        input_height=2,
        input_width=2,
        kernel_size=(2, 2),
    )
    expected_2d = torch.nn.functional.conv_transpose2d(input_2d.permute(0, 3, 1, 2), weight_2d)
    torch.testing.assert_close(output_2d.reshape(1, 3, 3, 3).permute(0, 3, 1, 2), expected_2d)


def test_fold_golden_contract():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)
    expected = x.reshape(1, 1, 2, 1, 2, 4).permute(0, 1, 3, 2, 4, 5).reshape(1, 1, 1, 16)
    torch.testing.assert_close(ttnn.get_golden_function(ttnn.fold)(x, 2, 2, False), expected)


def test_reduction_golden_contracts():
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    torch.testing.assert_close(ttnn.get_golden_function(ttnn.cumsum)(x, 3), torch.cumsum(x, 3))
    integer_x = x.to(torch.int32)
    assert ttnn.get_golden_function(ttnn.cumprod)(integer_x, 3).dtype == torch.int32
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.cumprod)(x, 3, reverse_order=True),
        torch.cumprod(x.flip((3,)), 3).flip((3,)),
    )
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.prod)(x, dim=3, keepdim=True), torch.prod(x, 3, keepdim=True)
    )

    values, indices = ttnn.get_golden_function(ttnn.sort)(x, descending=True)
    expected_values, expected_indices = torch.sort(x, descending=True)
    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.var_hw)(x),
        torch.var(x, (-2, -1), correction=0, keepdim=True),
    )
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.std_hw)(x),
        torch.std(x, (-2, -1), correction=0, keepdim=True),
    )

    count, coordinates = ttnn.get_golden_function(ttnn.nonzero)(x - 2)
    assert count[..., 0].item() == 3
    expected_coordinates = torch.nonzero(x - 2).to(torch.uint32).reshape(-1)
    torch.testing.assert_close(coordinates[..., : expected_coordinates.numel()].reshape(-1), expected_coordinates)


def test_preallocated_output_golden_contracts():
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

    accumulation_out = torch.empty_like(x)
    accumulation_result = ttnn.get_golden_function(ttnn.cumsum)(x, 2, out=accumulation_out)
    assert accumulation_result is accumulation_out
    torch.testing.assert_close(accumulation_result, torch.cumsum(x, 2))

    slice_out = torch.empty(1, 2, 2)
    slice_result = ttnn.get_golden_function(ttnn.slice)(
        x,
        [0, 0, 0],
        [1, 2, 4],
        [1, 1, 2],
        output_tensor=slice_out,
    )
    assert slice_result is slice_out
    torch.testing.assert_close(slice_result, x[:, :2, ::2])

    sort_values_out = torch.empty_like(x)
    sort_indices_out = torch.empty_like(x, dtype=torch.int64)
    sort_result = ttnn.get_golden_function(ttnn.sort)(x, out=(sort_values_out, sort_indices_out))
    assert sort_result[0] is sort_values_out
    assert sort_result[1] is sort_indices_out
    expected_values, expected_indices = torch.sort(x)
    torch.testing.assert_close(sort_result[0], expected_values)
    torch.testing.assert_close(sort_result[1], expected_indices)

    prod_out = torch.empty(1, 1, 4)
    prod_result = ttnn.get_golden_function(ttnn.prod)(x, output_tensor=prod_out, dims=[0, 1])
    assert prod_result is prod_out
    torch.testing.assert_close(prod_result, torch.prod(torch.prod(x, 1, keepdim=True), 0, keepdim=True))


def test_unary_backward_complex_and_ternary_golden_contracts():
    x = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    alpha = 0.25
    expected_ema = torch.tensor([[[[1.0, 1.75, 2.6875]]]])
    torch.testing.assert_close(ttnn.get_golden_function(ttnn.ema)(x, alpha), expected_ema)
    torch.testing.assert_close(ttnn.get_golden_function(ttnn.i1)(x), torch.special.i1(x))

    indices = torch.tensor([[0, 2, 0]])
    weight = torch.zeros(4, 2)
    output_gradient = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    expected_embedding_gradient = torch.zeros_like(weight)
    expected_embedding_gradient.index_add_(0, indices.reshape(-1), output_gradient.reshape(-1, 2))
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.embedding_bw)(indices, weight, output_gradient),
        expected_embedding_gradient,
    )

    product_input = torch.tensor([[2.0, 3.0, 4.0]])
    product_gradient = ttnn.get_golden_function(ttnn.prod_bw)(torch.ones(1, 1), product_input, dim=1)[0]
    torch.testing.assert_close(product_gradient, torch.tensor([[12.0, 8.0, 6.0]]))

    real = torch.tensor([1.0, 2.0])
    imag = torch.tensor([3.0, 4.0])
    torch.testing.assert_close(
        ttnn.get_golden_function(ttnn.complex_tensor)(real, imag),
        torch.complex(real, imag),
    )

    beta = torch.full_like(x, 2)
    expected_snake = x + torch.sin(x).square() / beta
    snake_out = torch.empty_like(x)
    snake_result = ttnn.get_golden_function(ttnn.snake_beta)(x, torch.ones_like(x), beta, output_tensor=snake_out)
    assert snake_result is snake_out
    torch.testing.assert_close(snake_result, expected_snake)
