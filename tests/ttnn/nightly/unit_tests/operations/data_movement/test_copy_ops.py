# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp, tt_dtype_to_torch_dtype
from models.common.utility_functions import is_grayskull
from tests.ttnn.unit_tests.operations.test_utils import get_ttnn_torch_dtype

pytestmark = pytest.mark.use_module_device


def run_copy_test(N, C, H, W, layout, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, ttnn.bfloat16, layout=layout, device=device)

    input_b = torch.zeros(shape, dtype=torch_dtype)
    input_b = ttnn.from_torch(input_b, ttnn.bfloat16, layout=layout, device=device)

    ttnn.copy(input, input_b)
    assert input_b.shape == input.shape
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(input_b), 0.99)


@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64),),
)
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
def test_copy(N, C, H, W, layout, device):
    run_copy_test(N, C, H, W, layout, device)


def run_assign_test(N, C, H, W, memory_config, dtype, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    tensor = ttnn.assign(input, memory_config=memory_config, dtype=dtype)
    assert tensor.shape == input.shape
    assert tensor.dtype == dtype
    assert tensor.memory_config() == memory_config
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(tensor), 0.99)


@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64),),
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_assign(N, C, H, W, memory_config, dtype, device):
    run_assign_test(N, C, H, W, memory_config, dtype, device)


def run_assign_test_opt_tensor(N, C, H, W, memory_config, dtype, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device, memory_config=memory_config)

    opt_tensor = torch.randn(shape, dtype=torch_dtype)
    opt_tensor = ttnn.from_torch(opt_tensor, dtype, layout=ttnn.Layout.TILE, device=device, memory_config=memory_config)

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.assign(input, memory_config=memory_config, dtype=dtype, output_tensor=opt_tensor)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    assert opt_tensor.shape == input.shape
    assert opt_tensor.dtype == dtype
    assert opt_tensor.memory_config() == memory_config
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(opt_tensor), 0.99)


@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64),),
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_assign_opt_tensor(N, C, H, W, memory_config, dtype, device):
    run_assign_test_opt_tensor(N, C, H, W, memory_config, dtype, device)


def run_binary_assign_test(N, C, H, W, layout, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, ttnn.bfloat16, layout=layout, device=device)

    input_b = torch.zeros(shape, dtype=torch_dtype)
    input_b = ttnn.from_torch(input_b, ttnn.bfloat16, layout=layout, device=device)

    ttnn.assign(input, input_b)
    assert input_b.shape == input.shape
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(input_b), 0.99)


@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64),),
)
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
def test_binary_assign(N, C, H, W, layout, device):
    run_binary_assign_test(N, C, H, W, layout, device)


def run_experimental_typecast_test(N, C, H, W, memory_config, input_dtype, output_dtype, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, input_dtype, layout=ttnn.Layout.TILE, device=device)
    tensor = ttnn.experimental.typecast(input, memory_config=memory_config, dtype=output_dtype)

    assert tensor.shape == input.shape
    assert tensor.dtype == output_dtype
    assert tensor.memory_config() == memory_config
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(tensor), 0.99)


@pytest.mark.parametrize(
    "input_dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64),),
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_experimental_typecast(N, C, H, W, memory_config, input_dtype, output_dtype, device):
    if input_dtype == output_dtype:
        pytest.skip("Input and output data types are the same")
    run_experimental_typecast_test(N, C, H, W, memory_config, input_dtype, output_dtype, device)


def run_typecast_test(N, C, H, W, memory_config, input_dtype, output_dtype, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, input_dtype, layout=ttnn.Layout.TILE, device=device)
    tensor = ttnn.typecast(input, memory_config=memory_config, dtype=output_dtype)

    assert tensor.shape == input.shape
    assert tensor.dtype == output_dtype
    assert tensor.memory_config() == memory_config
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(tensor), 0.99)


@pytest.mark.parametrize(
    "input_dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.float32,
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
        "FLOAT32",
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.float32,
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
        "FLOAT32",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64), (1, 1, 32, 32000)),
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_typecast(N, C, H, W, memory_config, input_dtype, output_dtype, device):
    if input_dtype == output_dtype:
        pytest.skip("Input and output data types are the same")
    run_typecast_test(N, C, H, W, memory_config, input_dtype, output_dtype, device)


# The idea of the test is to convert bfloat16 to uint32 into preallocated uint32 tensor
@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32/uint32/uint16 data types")
def test_typecast_output_tensor(device):
    torch.manual_seed(0)

    h = w = 32
    from_dtype = ttnn.bfloat16
    to_dtype = ttnn.uint32
    gold_tensor = ttnn.ones([h, w], to_dtype, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    bfloat16_tensor = ttnn.ones([h, w], from_dtype, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    uint32_preallocated = ttnn.empty([h, w], to_dtype, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)

    output_ttnn = ttnn.typecast(bfloat16_tensor, ttnn.uint32, memory_config=ttnn.L1_MEMORY_CONFIG)

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.typecast(bfloat16_tensor, to_dtype, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensor=uint32_preallocated)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    torch_gold = ttnn.to_torch(gold_tensor)
    torch_output_ttnn = ttnn.to_torch(output_ttnn)
    torch_output_ttnn_preallocated = ttnn.to_torch(uint32_preallocated)
    torch.equal(torch_gold, torch_output_ttnn)
    torch.equal(torch_gold, torch_output_ttnn_preallocated)


def test_typecast_community(device):
    shape = (1, 1, 32, 32)
    x = ttnn.to_device(ttnn.to_layout(ttnn.zeros(shape), ttnn.TILE_LAYOUT), device)
    y = ttnn.typecast(x, ttnn.types.bfloat16)
    assert y.dtype == ttnn.types.bfloat16
    assert_with_pcc(ttnn.to_torch(x), ttnn.to_torch(y), 0.99)


def run_typecast_row_major_test(shape, memory_config, input_dtype, output_dtype, device):
    """Test typecast operation on row-major device tensors."""
    torch.manual_seed(54321)
    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    # Create row-major tensor on device
    input_tensor = ttnn.from_torch(input, input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Verify input is row-major
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input tensor should be in ROW_MAJOR_LAYOUT"

    # Perform typecast
    output_tensor = ttnn.typecast(input_tensor, dtype=output_dtype, memory_config=memory_config)

    # Verify output properties
    assert output_tensor.shape == input_tensor.shape, "Output shape should match input shape"
    assert output_tensor.dtype == output_dtype, "Output dtype should match requested dtype"
    assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Output should maintain ROW_MAJOR_LAYOUT"
    assert output_tensor.memory_config() == memory_config, "Output memory config should match"

    # Convert original input to output dtype for comparison
    torch_output = ttnn.to_torch(output_tensor)

    # Map ttnn dtype to torch dtype for conversion
    torch_output_dtype = get_ttnn_torch_dtype(output_dtype)

    # Convert original input to match output dtype for comparison
    torch_expected = input.to(torch_output_dtype)

    if output_dtype == ttnn.int32:
        # For integer types, use exact equality (typecast from float to int may have rounding)
        assert torch.equal(
            torch_expected, torch_output
        ), f"Integer typecast mismatch: expected {torch_expected}, got {torch_output}"
    else:
        assert_with_ulp(torch_expected, torch_output, ulp_threshold=2)


@pytest.mark.parametrize(
    "input_dtype",
    (
        ttnn.bfloat16,
        ttnn.float32,
    ),
    ids=[
        "BFLOAT16",
        "FLOAT32",
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    (
        ttnn.int32,
        ttnn.bfloat16,
        ttnn.float32,
    ),
    ids=[
        "INT32",
        "BFLOAT16",
        "FLOAT32",
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (32, 64),
        (31, 1024 - 32),
        (32, 1024),
        (33, 1024 + 32),
        (1, 2, 2048 - 32),
        (7, 5, 3, 2048),
        (7, 5, 65, 2048 + 32),
        (1, 1, 320032),
    ],
    ids=[
        "32x64",
        "31x1024 - 32",
        "32x1024",
        "33x1024 + 32",
        "1x2x2048 - 32",
        "7x5x3x2048",
        "7x5x65x2048 + 32",
        "1x1x320032",
    ],
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_typecast_row_major(shape, memory_config, input_dtype, output_dtype, device):
    """
    Test that typecast works on row-major device tensors (fixes issue #16270).

    This test verifies:
    - Typecast works on row-major layout device tensors
    - Output maintains row-major layout
    - Various dtype conversions work correctly
    """
    if input_dtype == output_dtype:
        pytest.skip("Input and output data types are the same")

    run_typecast_row_major_test(shape, memory_config, input_dtype, output_dtype, device)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 64),
        (8, 2, 64, 32),
        (9, 2, 64, 32),
        (32, 32),
    ],
    ids=[
        "1x32x64",
        "8x2x64x32",
        "9x2x64x32",
        "32x32",
    ],
)
@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [
        (ttnn.bfloat16, ttnn.int32),
        (ttnn.bfloat16, ttnn.uint8),
        (ttnn.uint8, ttnn.bfloat16),
    ],
    ids=[
        "BFLOAT16_TO_INT32",
        "BFLOAT16_TO_UINT8",
        "UINT8_TO_BFLOAT16",
    ],
)
def test_typecast_row_major_vs_tile_layout(input_dtype, output_dtype, shape, device):
    """
    Test that row-major typecast produces same results as tile layout typecast.
    This ensures correctness of the row-major implementation.
    """
    torch.manual_seed(12345)

    # Use appropriate torch dtype and generation method based on input_dtype
    if input_dtype == ttnn.bfloat16:
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
    elif input_dtype == ttnn.uint8:
        torch_input = torch.randint(0, 256, shape, dtype=torch.uint8)
    else:
        torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # Create row-major tensor
    input_rm = ttnn.from_torch(torch_input, input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_rm = ttnn.typecast(input_rm, dtype=output_dtype)

    # Create tile layout tensor
    input_tile = ttnn.from_torch(torch_input, input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tile = ttnn.typecast(input_tile, dtype=output_dtype)

    # Convert both to torch and compare
    torch_output_rm = ttnn.to_torch(output_rm)
    torch_output_tile = ttnn.to_torch(output_tile)

    # Use appropriate assertion based on dtype
    # ULP is only applicable for floating-point types, not integers
    is_integer_type = output_dtype in (ttnn.uint8, ttnn.uint16, ttnn.uint32, ttnn.int32)
    if is_integer_type:
        assert torch.equal(
            torch_output_rm, torch_output_tile
        ), "Row-major and tile layouts should produce identical integer results"
    else:
        assert_with_ulp(torch_output_rm, torch_output_tile, ulp_threshold=2)

    # Verify layouts are preserved
    assert output_rm.layout == ttnn.ROW_MAJOR_LAYOUT, "Row-major output should maintain ROW_MAJOR_LAYOUT"
    assert output_tile.layout == ttnn.TILE_LAYOUT, "Tile output should maintain TILE_LAYOUT"


@pytest.mark.parametrize(
    "output_dtype",
    (
        ttnn.uint32,
        ttnn.int32,
        ttnn.uint16,
        ttnn.uint8,
        ttnn.bfloat16,
    ),
    ids=[
        "UINT32",
        "INT32",
        "UINT16",
        "UINT8",
        "BFLOAT16",
    ],
)
@pytest.mark.parametrize("preferred_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_typecast_host_tensor(output_dtype, preferred_layout, device):
    """
    Test that typecast works on host tensors (fixes issue #16279).

    This test reproduces the scenario from the bug report and tests various dtypes:
    - Create a host tensor from torch (no device parameter)
    - Call typecast on the host tensor
    - Verify it works without throwing an error
    """
    torch.manual_seed(2005)
    shape = [32, 2048]
    torch_tensor = torch.rand(shape, dtype=torch.float32)

    # Create host tensor
    ttnn_tensor_host = ttnn.from_torch(torch_tensor, layout=preferred_layout)

    # Verify input is on host
    assert ttnn_tensor_host.storage_type() == ttnn.StorageType.HOST

    # Perform typecast on host tensor
    ttnn_tensor_typecast = ttnn.typecast(ttnn_tensor_host, dtype=output_dtype)

    # Verify output properties
    assert ttnn_tensor_typecast.shape == ttnn_tensor_host.shape
    assert ttnn_tensor_typecast.dtype == output_dtype
    assert ttnn_tensor_typecast.storage_type() == ttnn.StorageType.HOST

    # Convert back to torch and verify the shape
    torch_output = ttnn.to_torch(ttnn_tensor_typecast)
    assert torch_output.shape == tuple(shape)

    # Verify values are reasonable
    torch_dtype = tt_dtype_to_torch_dtype[output_dtype]
    torch_expected = torch_tensor.to(torch_dtype)
    torch.equal(torch_expected, torch_output)
