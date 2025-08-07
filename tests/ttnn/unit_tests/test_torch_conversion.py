from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc, tt_dtype_to_torch_dtype
import torch
import ttnn
import pytest


torch.manual_seed(0)


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


TORCH_FLOAT_TYPES = [torch.float16, torch.float32, torch.float64]
ALL_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if dtype != ttnn.DataType.INVALID]
FLOAT_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if is_ttnn_float_type(dtype)]


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

        elif ttnn_dtype == ttnn.bfloat4_b:
            conversion_pcc = 0.960

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
        # (32, 32),
        # (32, 32, 64, 64),
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.uint8,
        ttnn.uint16,
        ttnn.uint32,
        ttnn.int32,
    ],
)
@pytest.mark.parametrize(
    "torch_dtype",
    [
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
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
    assert_with_pcc(torch_tensor, torch.Tensor(input_tensor.to_list()), 0.999999)


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
