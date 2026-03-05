# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from tests.ttnn.utils_for_testing import assert_with_pcc
import torch
import ttnn
import pytest
import numpy as np


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


TORCH_FLOAT_TYPES = [torch.float16, torch.float32, torch.float64]


def get_types_from_binding_framework():
    if hasattr(ttnn.DataType, "__entries"):
        # pybind
        ALL_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if dtype != ttnn.DataType.INVALID]
    elif hasattr(ttnn.DataType, "_member_map_"):
        # nanobind
        ALL_TYPES = [dtype for _, dtype in ttnn.DataType._member_map_.items() if dtype != ttnn.DataType.INVALID]
    else:
        raise Exception("ttnn.DataType has unexpected way of holding values. Not matching pybind/nanobind.")

    return ALL_TYPES


ALL_TYPES = get_types_from_binding_framework()
FLOAT_TYPES = [dtype for dtype in ALL_TYPES if is_ttnn_float_type(dtype)]
TTNN_MUST_TILE_TYPES = [ttnn.bfloat8_b, ttnn.bfloat4_b]
NUMPY_FLOAT_TYPES = [np.float16, np.float32, np.float64]


def get_expected_conversion_pcc(ttnn_dtype, other_dtype):
    ttnn_is_float = ttnn_dtype in [ttnn.float32, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b]
    other_is_float = other_dtype in TORCH_FLOAT_TYPES or other_dtype in NUMPY_FLOAT_TYPES

    if (ttnn_dtype == ttnn.float32 and other_dtype == torch.float32) or (
        ttnn_dtype == ttnn.int32 and other_dtype == torch.int32
    ):
        return 1

    elif ttnn_dtype == ttnn.bfloat4_b:
        return 0.960

    elif ttnn_is_float != other_is_float:
        return 0.98

    elif other_dtype == torch.bfloat16:
        if ttnn_dtype == ttnn.bfloat16 or ttnn_dtype == ttnn.bfloat8_b:
            return 0.9999
        elif ttnn_dtype == ttnn.bfloat4_b:
            return 0.989
        else:
            return 0.999

    else:
        return 0.999


def create_from_torch_test_tensors(
    device,
    shape,
    ttnn_dtype,
    torch_dtype,
    ttnn_layout,
    convert_with_device,
    min_range=0,
    max_range=100,
    memory_config=None,
):
    torch.manual_seed(0)
    if torch_dtype in TORCH_FLOAT_TYPES:
        torch_input_tensor = torch.rand(shape, dtype=torch_dtype) * max_range

    else:
        torch_input_tensor = torch.randint(min_range, max_range, shape, dtype=torch_dtype)

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device if convert_with_device else None,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT if (ttnn_dtype in TTNN_MUST_TILE_TYPES) else ttnn_layout,
        memory_config=memory_config,
    )

    assert (
        ttnn_result_tensor.dtype == ttnn_dtype
    ), f"Expected result {ttnn_dtype}, got result tensor {ttnn_result_tensor.dtype} when converting torch tensor {torch_input_tensor.dtype}"

    return torch_input_tensor, ttnn_result_tensor


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
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
def test_from_torch_conversion(device, shape, ttnn_dtype, torch_dtype, ttnn_layout, convert_with_device):
    torch.manual_seed(205)
    torch_input_tensor, ttnn_result_tensor = create_from_torch_test_tensors(
        device,
        shape,
        ttnn_dtype,
        torch_dtype,
        ttnn_layout,
        convert_with_device,
        min_range=0,
        max_range=10 if torch_dtype in TORCH_FLOAT_TYPES else 100,
    )

    assert_with_pcc(
        expected_pytorch_result=torch_input_tensor,
        actual_pytorch_result=ttnn_result_tensor.cpu().to_torch(),
        pcc=get_expected_conversion_pcc(ttnn_dtype, torch_dtype),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),
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
def test_to_torch_conversion(device, shape, ttnn_dtype, torch_dtype, ttnn_layout, convert_with_device):
    ttnn_dtype_has_random = ttnn_dtype not in [ttnn.uint8, ttnn.int32]
    if ttnn_dtype_has_random:
        for store_input_on_device in [True, False]:
            ttnn_input_tensor = ttnn.rand(
                shape,
                dtype=ttnn_dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT if (ttnn_dtype in TTNN_MUST_TILE_TYPES) else ttnn_layout,
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
            pcc=get_expected_conversion_pcc(ttnn_dtype, torch_dtype),
        )


@pytest.mark.parametrize("seed", list(range(6)))
@pytest.mark.parametrize(
    "shape,ttnn_dtype,torch_dtype,ttnn_layout,convert_with_device,value_ranges,pcc_override,memory_config",
    [
        ((4, 4), ttnn.float32, torch.float64, ttnn.TILE_LAYOUT, True, (0, 100), None, None),
        ((32, 32), ttnn.bfloat16, torch.int64, ttnn.ROW_MAJOR_LAYOUT, False, (0, 255), None, None),
        ((32, 32, 64, 64), ttnn.bfloat8_b, torch.float16, ttnn.TILE_LAYOUT, True, (0, 127), None, None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.TILE_LAYOUT, True, (0, 123123), 1, None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.TILE_LAYOUT, True, (-2147482786, 2147482213), 1, None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.ROW_MAJOR_LAYOUT, True, (-2147482786, 2147482213), 1, None),
        ((1, 1, 32, 1024), ttnn.float32, torch.float32, ttnn.TILE_LAYOUT, True, (0, 123123), 1, None),
        ((2, 256, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), 1, None),
        ((2, 256, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), 1, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_from_torch_conversion_with_fixed_edge_case_params(
    seed,
    device,
    shape,
    ttnn_dtype,
    torch_dtype,
    ttnn_layout,
    convert_with_device,
    value_ranges,
    pcc_override,
    memory_config,
):
    """
    Test `from_torch` conversion with a fixed set of parameters for various edge cases
    """
    torch.manual_seed(seed)
    torch_input_tensor, ttnn_result_tensor = create_from_torch_test_tensors(
        device,
        shape,
        ttnn_dtype,
        torch_dtype,
        ttnn_layout,
        convert_with_device,
        min_range=value_ranges[0],
        max_range=value_ranges[1],
        memory_config=memory_config,
    )

    assert_with_pcc(
        expected_pytorch_result=torch_input_tensor,
        actual_pytorch_result=ttnn_result_tensor.cpu().to_torch(),
        pcc=pcc_override or get_expected_conversion_pcc(ttnn_dtype, torch_dtype),
    )


@pytest.mark.parametrize("seed", list(range(6)))
@pytest.mark.parametrize(
    "shape,ttnn_dtype,torch_dtype,ttnn_layout,convert_with_device,value_ranges,memory_config",
    [
        ((4, 4), ttnn.float32, torch.float64, ttnn.TILE_LAYOUT, True, (0, 100), None),
        ((32, 32), ttnn.bfloat16, torch.int64, ttnn.ROW_MAJOR_LAYOUT, False, (0, 255), None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.TILE_LAYOUT, True, (0, 123123), None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.TILE_LAYOUT, True, (-2147482786, 2147482213), None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.ROW_MAJOR_LAYOUT, True, (-2147482786, 2147482213), None),
        ((1, 1, 32, 1024), ttnn.float32, torch.float32, ttnn.TILE_LAYOUT, True, (0, 123123), None),
        ((2, 256, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), None),
        ((2, 256, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, False, (-100, 100), None),
        ((2, 256, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), ttnn.L1_MEMORY_CONFIG),
        ((2, 256, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, False, (-100, 100), ttnn.L1_MEMORY_CONFIG),
        ((2, 64, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), None),
        ((2, 64, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, False, (-100, 100), None),
        ((2, 64, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), ttnn.L1_MEMORY_CONFIG),
        ((2, 64, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, False, (-100, 100), ttnn.L1_MEMORY_CONFIG),
        ((4, 4, 4, 4), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), None),
        ((4, 4, 4, 4), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, False, (-100, 100), None),
        ((4, 4, 4, 4), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), ttnn.L1_MEMORY_CONFIG),
        ((4, 4, 4, 4), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, False, (-100, 100), ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_from_torch_conversion_close(
    seed, device, shape, ttnn_dtype, torch_dtype, ttnn_layout, convert_with_device, value_ranges, memory_config
):
    """
    Test `from_torch` conversion with a fixed set of parameters for various edge cases
    """
    torch.manual_seed(seed)
    torch_tensor, ttnn_tensor = create_from_torch_test_tensors(
        device,
        shape,
        ttnn_dtype,
        torch_dtype,
        ttnn_layout,
        convert_with_device,
        min_range=value_ranges[0],
        max_range=value_ranges[1],
        memory_config=memory_config,
    )

    torch.testing.assert_close(torch.Tensor(torch_tensor.tolist()), torch.Tensor(ttnn_tensor.to_list()))


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        return torch.rand(shape, dtype=torch.bfloat16)
    raise ValueError(f"Unsupported dtype for random_torch_tensor: {dtype}")


@pytest.mark.parametrize(
    "shape", [[1, 1, 32, 32], [2, 2, 32, 32], [32, 32, 32, 32], [1, 1, 64, 64], [2, 2, 64, 64], [32, 32, 64, 64]]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_dtype_conversion_pcc(device, shape, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    assert_with_pcc(torch_tensor, torch.Tensor(input_tensor.to_list()), 0.999999)


def create_from_numpy_test_tensors(
    device,
    shape,
    ttnn_dtype,
    numpy_dtype,
    ttnn_layout,
    convert_with_device,
    min_range=0,
    max_range=100,
    memory_config=None,
):
    if numpy_dtype in NUMPY_FLOAT_TYPES:
        numpy_input_tensor = np.random.rand(*shape).astype(numpy_dtype) * max_range
    else:
        numpy_input_tensor = np.random.randint(min_range, max_range, shape, dtype=numpy_dtype)

    ttnn_result_tensor = ttnn.from_torch(
        numpy_input_tensor,
        device=device if convert_with_device else None,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT if (ttnn_dtype in TTNN_MUST_TILE_TYPES) else ttnn_layout,
        memory_config=memory_config,
    )

    assert (
        ttnn_result_tensor.dtype == ttnn_dtype
    ), f"Expected result {ttnn_dtype}, got result tensor {ttnn_result_tensor.dtype} when converting numpy tensor {numpy_input_tensor.dtype}"

    return numpy_input_tensor, ttnn_result_tensor


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.float32,
        ttnn.uint8,
        ttnn.uint16,
        ttnn.uint32,
        ttnn.int32,
    ],
)
@pytest.mark.parametrize(
    "numpy_dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ],
)
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("convert_with_device", [True, False])
def test_from_numpy_conversion(device, shape, ttnn_dtype, numpy_dtype, ttnn_layout, convert_with_device):
    np.random.seed(205)
    numpy_input_tensor, ttnn_result_tensor = create_from_numpy_test_tensors(
        device,
        shape,
        ttnn_dtype,
        numpy_dtype,
        ttnn_layout,
        convert_with_device,
        min_range=0,
        max_range=10 if numpy_dtype in NUMPY_FLOAT_TYPES else 100,
    )

    # Convert numpy tensor to torch for comparison
    torch_input_tensor = torch.from_numpy(numpy_input_tensor)
    torch_computed = ttnn_result_tensor.cpu().to_torch()

    assert isinstance(torch_input_tensor, torch.Tensor)
    assert isinstance(torch_computed, torch.Tensor)

    assert_with_pcc(
        expected_pytorch_result=torch_input_tensor.to(torch.float64),
        actual_pytorch_result=torch_computed.to(torch.float64),
        pcc=get_expected_conversion_pcc(ttnn_dtype, numpy_dtype),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.uint16,
        ttnn.uint32,
    ],
)
@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_torch_conversion_unsigned_edge_cases_random(device, shape, ttnn_dtype, ttnn_layout, seed):
    torch.manual_seed(seed)

    if ttnn_dtype == ttnn.uint16:
        low = np.iinfo(np.uint16).min
        high = np.iinfo(np.uint16).max

    elif ttnn_dtype == ttnn.uint32:
        low = np.iinfo(np.uint32).min
        high = np.iinfo(np.uint32).max

    ttnn_input_tensor = ttnn.rand(
        shape,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn_layout,
        low=low,
        high=high,
    )

    torch_result_tensor: torch.Tensor = ttnn.to_torch(ttnn_input_tensor)

    torch.testing.assert_close(torch.tensor(torch_result_tensor.tolist()), torch.tensor(ttnn_input_tensor.to_list()))


@pytest.mark.parametrize(
    "tensor_data,ttnn_dtype,torch_input_type",
    [
        ([np.iinfo(np.uint16).max], ttnn.uint16, torch.uint16),
        ([np.iinfo(np.uint16).min], ttnn.uint16, torch.uint16),
        ([np.iinfo(np.uint32).max], ttnn.uint32, torch.uint32),
        ([np.iinfo(np.uint32).min], ttnn.uint32, torch.uint32),
    ],
)
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("with_device", [True, False])
def test_torch_conversion_unsigned_edge_cases(
    device, tensor_data, ttnn_dtype, torch_input_type, ttnn_layout, with_device
):
    torch_input_tensor = torch.tensor(tensor_data, dtype=torch_input_type)
    ttnn_input_tensor = ttnn.Tensor(
        torch_input_tensor,
        dtype=ttnn_dtype,
        layout=ttnn_layout,
        device=device if with_device else None,
    )

    torch_result_tensor: torch.Tensor = ttnn.to_torch(ttnn_input_tensor)

    torch.testing.assert_close(torch.tensor(torch_input_tensor.tolist()), torch.tensor(ttnn_input_tensor.to_list()))
    torch.testing.assert_close(torch.tensor(torch_result_tensor.tolist()), torch.tensor(ttnn_input_tensor.to_list()))
    torch.testing.assert_close(torch_input_tensor, torch_result_tensor)


@pytest.mark.parametrize(
    "tensor_data,ttnn_dtype,numpy_input_type",
    [
        ([np.iinfo(np.uint16).max], ttnn.uint16, np.uint16),
        ([np.iinfo(np.uint16).min], ttnn.uint16, np.uint16),
        ([np.iinfo(np.uint32).max], ttnn.uint32, np.uint32),
        ([np.iinfo(np.uint32).min], ttnn.uint32, np.uint32),
    ],
)
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("with_device", [True, False])
def test_numpy_conversion_unsigned_edge_cases_fixed(
    device, tensor_data, ttnn_dtype, numpy_input_type, ttnn_layout, with_device
):
    numpy_input_tensor = np.array(tensor_data, dtype=numpy_input_type)
    ttnn_input_tensor = ttnn.from_torch(
        numpy_input_tensor,
        dtype=ttnn_dtype,
        layout=ttnn_layout,
        device=device if with_device else None,
    )
    numpy_result_tensor = ttnn_input_tensor.cpu().to_numpy()
    np.testing.assert_allclose(np.array(numpy_input_tensor.tolist()), np.array(ttnn_input_tensor.to_list()))
    np.testing.assert_allclose(np.array(numpy_result_tensor.tolist()), np.array(ttnn_input_tensor.to_list()))
    np.testing.assert_allclose(numpy_input_tensor, numpy_result_tensor)


def generate_bfloat4_b_exact_tensor(shape, seed=0):
    """
    Generate a float32 torch tensor whose values survive a round-trip through
    bfloat4_b without any precision loss.

    bfloat4_b is a block floating-point format where every 16-element block
    (one face row in tile layout) shares a single 8-bit exponent, and each
    element stores 1 sign bit + 3 mantissa bits (with an implicit leading 1).

    To guarantee exact round-trip, this function ensures:
     - All 16 elements in each block share the same power-of-two exponent,
        so the shared exponent equals every element's exponent (no alignment
        shift, no mantissa bit loss).
     - Each element's mantissa is one of the 8 exactly representable values:
        {1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875}
        which correspond to the 3-bit patterns 000..111 with hidden bit.

    The last two dimensions of *shape* must be multiples of 32 (tile layout).
    """
    torch.manual_seed(seed)

    EXACT_MANTISSAS = torch.tensor([1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875], dtype=torch.float32)
    FACE_ROW_SIZE = 16

    assert len(shape) >= 2, "Shape must have at least 2 dimensions"
    H, W = shape[-2], shape[-1]
    assert H % 32 == 0 and W % 32 == 0, f"Last two dims must be multiples of 32 for tile layout, got ({H}, {W})"

    total_elements = 1
    for d in shape:
        total_elements *= d
    total_rows = total_elements // W
    num_blocks_per_row = W // FACE_ROW_SIZE
    total_blocks = total_rows * num_blocks_per_row

    exponents = torch.randint(-4, 5, (total_blocks, 1), dtype=torch.float32)
    mantissa_indices = torch.randint(0, 8, (total_blocks, FACE_ROW_SIZE))
    mantissas = EXACT_MANTISSAS[mantissa_indices]
    signs = torch.where(
        torch.randint(0, 2, (total_blocks, FACE_ROW_SIZE)).bool(),
        torch.ones(1, dtype=torch.float32),
        -torch.ones(1, dtype=torch.float32),
    )

    values = signs * torch.pow(2.0, exponents) * mantissas
    return values.reshape(shape)


def quantize_to_bf4(tensor, exp_bits=2, mant_bits=1):
    """
    Simulates a 4-bit float roundtrip.
    Default: 1 sign bit, 2 exponent bits, 1 mantissa bit (E2M1).
    """
    # 1. Capture the sign
    sign = torch.sign(tensor)
    abs_tensor = torch.abs(tensor)

    # 2. Handle zeros to avoid log errors
    abs_tensor[abs_tensor == 0] = 1e-8

    # 3. Log-scale to find the exponent
    # bfloat4 has a very narrow range (usually 2^exponent)
    exponent = torch.floor(torch.log2(abs_tensor))

    # Clip exponent to fit in 2 bits (e.g., range -1 to 2)
    max_exp = 2 ** (exp_bits - 1)
    exponent = torch.clamp(exponent, -max_exp, max_exp)

    # 4. Quantize the mantissa
    # With 1 mantissa bit, we only have two levels: 1.0 and 1.5
    mantissa = abs_tensor / (2**exponent)
    mantissa = torch.round(mantissa * (2**mant_bits)) / (2**mant_bits)
    mantissa = torch.clamp(mantissa, 1.0, 2.0 - (1 / (2**mant_bits)))

    # 5. Reconstruct the "crushed" value
    bf4_simulated = sign * mantissa * (2**exponent)

    return bf4_simulated.to(torch.float32)


@pytest.mark.parametrize("shape", [(1, 7168, 2304)])
@pytest.mark.parametrize("shard_shape", [(7168, 192)])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat4_b])
@pytest.mark.parametrize("torch_dtype", [torch.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])  # ttnn.ROW_MAJOR_LAYOUT
def test_from_torch_conversion_deep_seek(device, shape, shard_shape, ttnn_dtype, torch_dtype, layout):
    torch.manual_seed(0)
    # torch.rand(shape, dtype=torch_dtype)
    torch_input_tensor = generate_bfloat4_b_exact_tensor(shape)
    torch_input_tensor = quantize_to_bf4(torch_input_tensor)

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))])
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=ttnn_dtype,
        layout=layout,
        memory_config=memory_config,
    )

    assert (
        ttnn_result_tensor.dtype == ttnn_dtype
    ), f"Expected result {ttnn_dtype}, got result tensor {ttnn_result_tensor.dtype} when converting torch tensor {torch_input_tensor.dtype}"

    torch_result_tensor = ttnn.to_torch(ttnn_result_tensor)
    # torch.testing.assert_close(torch_input_tensor, torch_result_tensor)
    assert torch.equal(torch_input_tensor, torch_result_tensor)


@pytest.mark.parametrize("shape", [(1, 7168, 2304)])
@pytest.mark.parametrize("shard_shape", [(7168, 192)])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat4_b])
@pytest.mark.parametrize("torch_dtype", [torch.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])  # ttnn.ROW_MAJOR_LAYOUT
def test_from_torch_conversion_deep_seek_mc(device, shape, shard_shape, ttnn_dtype, torch_dtype, layout):
    torch.manual_seed(0)
    # torch.rand(shape, dtype=torch_dtype)
    torch_input_tensor = generate_bfloat4_b_exact_tensor(shape)
    torch_input_tensor = quantize_to_bf4(torch_input_tensor)

    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))]
    )  # TODO: Choose DRAM cores
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,  # ttnn.DRAM_MEMORY_CONFIG
    )

    print("to_layout")
    ttnn_result_tensor = ttnn.to_layout(ttnn_result_tensor, layout)
    ttnn.synchronize_device(device)

    print("typecast")
    ttnn_result_tensor = ttnn.typecast(ttnn_result_tensor, ttnn_dtype)
    ttnn.synchronize_device(device)

    assert (
        ttnn_result_tensor.dtype == ttnn_dtype
    ), f"Expected result {ttnn_dtype}, got result tensor {ttnn_result_tensor.dtype} when converting torch tensor {torch_input_tensor.dtype}"

    print("to_torch")
    # torch_result_tensor = ttnn.to_torch(ttnn_result_tensor)
    # torch.testing.assert_close(torch_input_tensor, torch_result_tensor)
    # assert torch.equal(torch_input_tensor, torch_result_tensor)


DRAM_CORE_GRID_12 = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))])


@pytest.mark.parametrize(
    "shape,shard_shape,memory_layout",
    [
        # wq_b: WIDTH_SHARDED, shard [1536, 256], 12 DRAM banks, shard_dims=(0, -1)
        ((1, 1536, 3072), (1536, 256), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        # wo: WIDTH_SHARDED, shard [16384, 96], 12 DRAM banks, shard_dims=(0, -1)
        ((1, 16384, 1152), (16384, 96), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        # wq_kv_a: WIDTH_SHARDED, shard [896, 192], 12 DRAM banks, shard_dims=(0, -2)
        ((1, 896, 2304), (896, 192), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        # wkv_b1: HEIGHT_SHARDED, shard [256, 512], 12 DRAM banks, shard_dims=(0, -3)
        ((1, 16, 128, 512), (256, 512), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        # wkv_b2: HEIGHT_SHARDED, shard [5632, 128], 12 DRAM banks, shard_dims=(0, None)
        # ((4, 128,512, 128), (5632, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ],
    ids=["wq_b", "wo", "wq_kv_a", "wkv_b1"],  # , "wkv_b2"
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat8_b])
def test_from_torch_deep_seek_sharded_weights_single_device(device, shape, shard_shape, memory_layout, ttnn_dtype):
    torch.manual_seed(0)
    if ttnn_dtype == ttnn.bfloat8_b:
        torch_input_tensor = generate_bfloat4_b_exact_tensor(shape)
        torch_input_tensor = quantize_to_bf4(torch_input_tensor)
    else:
        torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)  # float32?

    memory_config = ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(DRAM_CORE_GRID_12, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )

    assert ttnn_result_tensor.dtype == ttnn_dtype, f"Expected {ttnn_dtype}, got {ttnn_result_tensor.dtype}"

    torch_result_tensor = ttnn.to_torch(ttnn_result_tensor)
    assert torch.equal(torch_input_tensor, torch_result_tensor)


# TODO: Increase shape size according to shard_dims
@pytest.mark.parametrize(
    "shape,shard_shape,memory_layout,shard_dims",
    [
        # wq_b: WIDTH_SHARDED, shard [1536, 256], 12 DRAM banks, shard_dims=(0, -1)
        ([1, 1536, 3072], (1536, 256), ttnn.TensorMemoryLayout.WIDTH_SHARDED, (0, -1)),
        # wo: WIDTH_SHARDED, shard [16384, 96], 12 DRAM banks, shard_dims=(0, -1)
        ([1, 16384, 1152], (16384, 96), ttnn.TensorMemoryLayout.WIDTH_SHARDED, (0, -1)),
        # wq_kv_a: WIDTH_SHARDED, shard [896, 192], 12 DRAM banks, shard_dims=(0, -2)
        ([1, 896, 2304], (896, 192), ttnn.TensorMemoryLayout.WIDTH_SHARDED, (0, -2)),
        # wkv_b1: HEIGHT_SHARDED, shard [256, 512], 12 DRAM banks, shard_dims=(0, -3)
        ([1, 3072, 512], (256, 512), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (0, -3)),
        # wkv_b2: HEIGHT_SHARDED, shard [5632, 128], 12 DRAM banks, shard_dims=(0, None)
        # ([1, 67584, 128], (5632, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (0, None)),
    ],
    ids=["wq_b", "wo", "wq_kv_a", "wkv_b1"],  # , "wkv_b2"
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat8_b])
def test_from_torch_deep_seek_sharded_weights_galaxy(device, shape, shard_shape, memory_layout, shard_dims, ttnn_dtype):
    torch.manual_seed(0)
    if shard_dims[0] is not None and shard_dims[1] is not None:
        shape[shard_dims[0]] = shape[shard_dims[0]] * device.shape[shard_dims[0]]
        shape[shard_dims[1]] = shape[shard_dims[1]] * device.shape[shard_dims[1]]

    if ttnn_dtype == ttnn.bfloat4_b:
        torch_input_tensor = generate_bfloat4_b_exact_tensor(shape)
        torch_input_tensor = quantize_to_bf4(torch_input_tensor)
    else:
        torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)  # float32?

    memory_config = ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(DRAM_CORE_GRID_12, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    mesh_shape = (device.shape[0], device.shape[1])

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=mesh_shape),
    )

    assert ttnn_result_tensor.dtype == ttnn_dtype, f"Expected {ttnn_dtype}, got {ttnn_result_tensor.dtype}"

    torch_result_tensor = ttnn.to_torch(ttnn_result_tensor)
    assert torch.equal(torch_input_tensor, torch_result_tensor)


@pytest.mark.parametrize("shape", [(1, 1, 224, 32)])
def test_from_torch_deep_seek_interleaved(device, shape):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)

    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
    )

    assert ttnn_result_tensor.dtype == ttnn.bfloat16, f"Expected {ttnn.bfloat16}, got {ttnn_result_tensor.dtype}"

    torch_result_tensor = ttnn.to_torch(ttnn_result_tensor)
    assert torch.equal(torch_input_tensor, torch_result_tensor)
