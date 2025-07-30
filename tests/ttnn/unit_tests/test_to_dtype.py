# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc, tt_dtype_to_torch_dtype

bfloat4_pcc = 0.960
torch.manual_seed(0)


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


ALL_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if dtype != ttnn.DataType.INVALID]
FLOAT_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if is_ttnn_float_type(dtype)]
TORCH_FLOAT_TYPES = [torch.float16, torch.float32, torch.float64]


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("to_dtype", ALL_TYPES)
@pytest.mark.parametrize("from_dtype", ALL_TYPES)
def test_to_dtype(height, width, from_dtype, to_dtype):
    torch_input_tensor = torch.randint(0, 10, (height, width), dtype=tt_dtype_to_torch_dtype[from_dtype])

    input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)

    assert output_tensor.dtype == to_dtype
    assert tuple(output_tensor.shape) == (height, width)
    if to_dtype == ttnn.bfloat8_b or to_dtype == ttnn.bfloat4_b:
        assert output_tensor.layout == ttnn.TILE_LAYOUT
    else:
        assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor, bfloat4_pcc if to_dtype == ttnn.bfloat4_b else 0.9999)


def run_dtype_conversion_on_device(
    device,
    shape,
    ttnn_dtype,
    torch_dtype,
    ttnn_layout,
    convert_with_device,
    min_range=0,
    max_range=100,
    pcc_override=None,
):
    ttnn_dtype_requires_tile = ttnn_dtype in [ttnn.bfloat8_b, ttnn.bfloat4_b]
    ttnn_dtype_has_random = ttnn_dtype not in [ttnn.uint8, ttnn.int32]
    ttnn_is_float = ttnn_dtype in [ttnn.float32, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b]
    torch_is_float = torch_dtype in TORCH_FLOAT_TYPES

    conversion_pcc = None

    if pcc_override:
        conversion_pcc = pcc_override

    else:
        if (ttnn_dtype == ttnn.float32 and torch_dtype == torch.float32) or (
            ttnn_dtype == ttnn.int32 and torch_dtype == torch.int32
        ):
            conversion_pcc = 1

        elif ttnn_is_float != torch_is_float:
            conversion_pcc = 0.98

        elif torch_dtype == torch.bfloat16:
            if ttnn_dtype == ttnn.bfloat16 or ttnn_dtype == ttnn.bfloat8_b:
                conversion_pcc = 0.9999
            elif ttnn_dtype == ttnn.bfloat4_b:
                conversion_pcc = 0.989
            else:
                conversion_pcc = 0.999

        else:
            conversion_pcc = 0.999

    if ttnn_dtype_has_random:
        for store_input_on_device in [True, False]:
            ttnn_input_tensor = ttnn.rand(
                shape,
                dtype=ttnn_dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT if ttnn_dtype_requires_tile else ttnn_layout,
            )

            if not store_input_on_device:
                ttnn_input_tensor = ttnn.from_device(ttnn_input_tensor)

            torch_result_tensor = ttnn.to_torch(
                ttnn_input_tensor, dtype=torch_dtype, device=device if convert_with_device else None
            )
            assert (
                torch_result_tensor.dtype == torch_dtype
            ), f"Expected result {torch_dtype}, got result tensor {torch_result_tensor.dtype} when converting TTNN tensor {ttnn_input_tensor.dtype}"

        assert_with_pcc(
            expected_pytorch_result=torch_result_tensor,
            actual_pytorch_result=ttnn_input_tensor.cpu().to_torch(),
            pcc=conversion_pcc,
        )

    if torch_is_float:
        torch_input_tensor = torch.rand(shape, dtype=torch_dtype) * max_range

    else:
        torch_input_tensor = torch.randint(min_range, max_range, shape, dtype=torch_dtype)

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device if convert_with_device else None,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT if ttnn_dtype_requires_tile else ttnn_layout,
    )

    assert (
        ttnn_result_tensor.dtype == ttnn_dtype
    ), f"Expected result {ttnn_dtype}, got result tensor {ttnn_result_tensor.dtype} when converting torch tensor {torch_input_tensor.dtype}"

    assert_with_pcc(
        expected_pytorch_result=torch_input_tensor,
        actual_pytorch_result=ttnn_result_tensor.cpu().to_torch(),
        pcc=conversion_pcc,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (32, 32),
        (32, 32, 64, 64),
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat16,
        ttnn.uint8,
        ttnn.int32,
        ttnn.uint32,
    ],
)
@pytest.mark.parametrize(
    "torch_dtype",
    [
        torch.float64,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.int32,
        torch.uint8,
    ],
)
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("convert_with_device", [True, False])
def test_dtype_conversion_on_device(device, shape, ttnn_dtype, torch_dtype, ttnn_layout, convert_with_device):
    run_dtype_conversion_on_device(
        device,
        shape,
        ttnn_dtype,
        torch_dtype,
        ttnn_layout,
        convert_with_device,
        min_range=0,
        max_range=10 if torch_dtype in TORCH_FLOAT_TYPES else 100,
    )


@pytest.mark.parametrize(
    "shape,ttnn_dtype,torch_dtype,ttnn_layout,convert_with_device,value_ranges,pcc_override",
    [
        ((4, 4), ttnn.float32, torch.float64, ttnn.TILE_LAYOUT, True, (0, 100), None),
        ((32, 32), ttnn.bfloat16, torch.int64, ttnn.ROW_MAJOR_LAYOUT, False, (0, 255), None),
        ((32, 32, 64, 64), ttnn.bfloat8_b, torch.float16, ttnn.TILE_LAYOUT, True, (0, 127), None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.TILE_LAYOUT, True, (0, 123123), 1),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.TILE_LAYOUT, True, (-2147482786, 2147482213), 1),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.ROW_MAJOR_LAYOUT, True, (-2147482786, 2147482213), 1),
        ((1, 1, 32, 1024), ttnn.float32, torch.float32, ttnn.TILE_LAYOUT, True, (0, 123123), 1),
    ],
)
def test_dtype_conversion_on_device_extra(
    device, shape, ttnn_dtype, torch_dtype, ttnn_layout, convert_with_device, value_ranges, pcc_override
):
    run_dtype_conversion_on_device(
        device,
        shape,
        ttnn_dtype,
        torch_dtype,
        ttnn_layout,
        convert_with_device,
        min_range=value_ranges[0],
        max_range=value_ranges[1],
        pcc_override=pcc_override,
    )


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        return torch.rand(shape, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "shape", [[1, 1, 32, 32], [2, 2, 32, 32], [32, 32, 32, 32], [1, 1, 64, 64], [2, 2, 64, 64], [32, 32, 64, 64]]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_dtype_conversion_pcc(device, shape, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    assert_with_pcc(torch_tensor, torch.Tensor(input_tensor.to_list()), 1)


def format_tensor_as_string(tensor_list: list, precision: int = 4, max_width: int = 120) -> str:
    def get_tensor_shape(nested_list):
        if not isinstance(nested_list, list):
            return []
        if len(nested_list) == 0:
            return [0]

        shape = [len(nested_list)]
        if isinstance(nested_list[0], list):
            shape.extend(get_tensor_shape(nested_list[0]))
        return shape

    def format_number(num, width):
        if isinstance(num, (int, float)):
            if abs(num) < 1e-10:
                formatted = "0.0"
            else:
                formatted = f"{num:.{precision}f}"
        else:
            formatted = str(num)
        return formatted.rjust(width)

    def get_all_values(nested_list):
        values = []
        if isinstance(nested_list, list):
            for item in nested_list:
                values.extend(get_all_values(item))
        else:
            values.append(nested_list)
        return values

    def calculate_col_width(nested_list):
        all_values = get_all_values(nested_list)
        formatted_values = []
        for val in all_values:
            if isinstance(val, (int, float)):
                formatted_values.append(f"{val:.{precision}f}")
            else:
                formatted_values.append(str(val))

        if not formatted_values:
            return precision + 4

        max_len = max(len(str(val)) for val in formatted_values)
        return max(max_len + 2, precision + 4)

    def format_recursive(nested_list, depth, col_width, is_last_at_level):
        if not isinstance(nested_list, list):
            return format_number(nested_list, col_width)

        if len(nested_list) == 0:
            return "[]"

        if not isinstance(nested_list[0], list):
            formatted_items = [format_number(item, col_width) for item in nested_list]
            return "[ " + "   ".join(formatted_items) + " ]"

        lines = []
        indent = " " * depth

        for i, item in enumerate(nested_list):
            is_last = i == len(nested_list) - 1
            formatted_item = format_recursive(item, depth + 1, col_width, is_last)

            if i == 0:
                lines.append("[" + formatted_item)
            else:
                lines.append(indent + " " + formatted_item)

        if depth > 0:
            lines[-1] += "]"
        else:
            lines[-1] += "]"

        return "\n".join(lines)

    if not isinstance(tensor_list, list) or len(tensor_list) == 0:
        return "[]"

    col_width = calculate_col_width(tensor_list)
    return format_recursive(tensor_list, 0, col_width, True)


@pytest.mark.parametrize(
    "shape",
    [
        (8, 1, 8, 8),
        (8, 2, 1024),
    ],
)
@pytest.mark.parametrize("ttnn_dtype_from", ALL_TYPES)
@pytest.mark.parametrize("ttnn_dtype_to", ALL_TYPES)
def test_typecast_accuracy(shape, device, ttnn_dtype_from, ttnn_dtype_to):
    if ttnn_dtype_from == ttnn.uint8 or ttnn_dtype_to == ttnn.uint8:
        pytest.skip("uint8 is not supported by the typecast directly")

    conversion_pcc = None

    if ttnn_dtype_from == ttnn_dtype_to:
        conversion_pcc = 1

    elif ttnn_dtype_to == ttnn.bfloat4_b:
        conversion_pcc = 0.960

    elif ttnn_dtype_to == ttnn.bfloat8_b:
        conversion_pcc = 0.999

    else:
        conversion_pcc = 0.9999

    if is_ttnn_float_type(ttnn_dtype_from):
        input_tensor = ttnn.rand(shape, device=device, dtype=ttnn_dtype_from) * 100

    else:
        input_tensor = ttnn.rand(shape, device=device, dtype=ttnn_dtype_from, low=0, high=100)

    input_torch_tensor = torch.Tensor(input_tensor.to_list())
    output_tensor = ttnn.typecast(input_tensor, dtype=ttnn_dtype_to)
    output_torch_tensor = torch.Tensor(output_tensor.to_list())

    pcc_passed, pcc_message = comp_pcc(input_torch_tensor, output_torch_tensor, conversion_pcc)
    format_message = f"""
{pcc_message}
    """

    print(format_message)

    assert pcc_passed, format_message


@pytest.mark.parametrize(
    "shape",
    [
        (8, 1, 8, 8),
        (4, 4),
        (32, 32),
        (8, 8, 1024),
    ],
)
@pytest.mark.parametrize("ttnn_dtype", ALL_TYPES)
@pytest.mark.parametrize("to_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("from_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_layout_conversion_accuracy(shape, device, ttnn_dtype, to_layout, from_layout):
    type_requires_tile = ttnn_dtype in [ttnn.bfloat8_b, ttnn.bfloat4_b]
    type_requires_row_major = ttnn_dtype not in [ttnn.bfloat16, ttnn.uint32, ttnn.float32, ttnn.int32]
    if type_requires_tile and (to_layout == ttnn.ROW_MAJOR_LAYOUT or from_layout == ttnn.ROW_MAJOR_LAYOUT):
        pytest.skip(f"{ttnn_dtype} requires tile layout")

    elif ttnn_dtype == ttnn.uint8:
        pytest.skip("uint8 is not supported for rand")

    elif type_requires_row_major and (to_layout != ttnn.ROW_MAJOR_LAYOUT or from_layout != ttnn.ROW_MAJOR_LAYOUT):
        pytest.skip(f"{ttnn_dtype} requires row-major layout")

    elif ttnn_dtype == ttnn.float32 and from_layout == ttnn.TILE_LAYOUT and to_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.xfail("float32 loses precision when converting from tile to row major layout")

    if is_ttnn_float_type(ttnn_dtype):
        input_tensor = ttnn.rand(shape, device=device, dtype=ttnn_dtype, layout=from_layout) * 100

    else:
        input_tensor = ttnn.rand(shape, device=device, dtype=ttnn_dtype, low=0, high=100, layout=from_layout)

    input_torch_tensor = torch.Tensor(input_tensor.to_list())
    output_tensor = ttnn.to_layout(input_tensor, layout=to_layout)
    output_torch_tensor = torch.Tensor(output_tensor.to_list())

    pcc_passed, pcc_message = comp_pcc(input_torch_tensor, output_torch_tensor, 1)
    format_message = f"""
{pcc_message}
    """

    assert pcc_passed, format_message


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("to_dtype", FLOAT_TYPES)
@pytest.mark.parametrize("from_dtype", FLOAT_TYPES)
def test_to_float_dtype(height, width, from_dtype, to_dtype):
    torch_input_tensor = torch.rand((height, width), dtype=tt_dtype_to_torch_dtype[from_dtype])

    input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)

    assert output_tensor.dtype == to_dtype
    assert tuple(output_tensor.shape) == (height, width)
    if to_dtype == ttnn.bfloat8_b or to_dtype == ttnn.bfloat4_b:
        assert output_tensor.layout == ttnn.TILE_LAYOUT
    else:
        assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor, bfloat4_pcc if to_dtype == ttnn.bfloat4_b else 0.9999)


@pytest.mark.parametrize("height", [36])
@pytest.mark.parametrize("width", [36])
@pytest.mark.parametrize("to_dtype", ALL_TYPES)
@pytest.mark.parametrize("from_dtype", ALL_TYPES)
def test_to_dtype_unaligned_shape(height, width, from_dtype, to_dtype):
    if (
        from_dtype == ttnn.bfloat4_b
        or from_dtype == ttnn.bfloat8_b
        or to_dtype == ttnn.bfloat4_b
        or to_dtype == ttnn.bfloat8_b
    ):
        pytest.skip("bfloat4_b and bfloat8_b require align shape divisible by tile")

    torch_input_tensor = torch.randint(0, 10, (height, width), dtype=tt_dtype_to_torch_dtype[from_dtype])

    input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)

    assert output_tensor.dtype == to_dtype
    assert tuple(output_tensor.shape) == (height, width)
    assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor, bfloat4_pcc if to_dtype == ttnn.bfloat4_b else 0.9999)


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("to_dtype", ALL_TYPES)
@pytest.mark.parametrize("from_dtype", ALL_TYPES)
def test_to_dtype_with_tile_layout(height, width, from_dtype, to_dtype):
    torch_input_tensor = torch.randint(0, 10, (height, width), dtype=tt_dtype_to_torch_dtype[from_dtype])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    assert input_tensor.layout == ttnn.TILE_LAYOUT

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)

    assert output_tensor.dtype == to_dtype
    assert tuple(output_tensor.shape) == (height, width)
    assert output_tensor.layout == ttnn.TILE_LAYOUT

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor, bfloat4_pcc if to_dtype == ttnn.bfloat4_b else 0.9999)
