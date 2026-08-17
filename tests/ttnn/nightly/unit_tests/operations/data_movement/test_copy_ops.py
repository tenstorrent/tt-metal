# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.python_api_testing.sweep_tests.ttnn_pytorch_ops import eltwise_typecast
from tests.ttnn.python_api_testing.typecast_test_helpers import (
    INTEGER_OUTPUT_DTYPES,
    assert_integer_typecast_equal,
    make_typecast_test_input,
    typecast_test_input_bounds,
)
from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc, assert_with_ulp, tt_dtype_to_torch_dtype
from tests.ttnn.unit_tests.operations.test_utils import get_ttnn_torch_dtype

pytestmark = pytest.mark.use_module_device


def _make_typecast_torch_input(shape, input_dtype, output_dtype):
    pt_dtype = tt_dtype_to_torch_dtype[input_dtype]
    if output_dtype in INTEGER_OUTPUT_DTYPES:
        in_low, in_high = typecast_test_input_bounds(input_dtype, output_dtype)
        return make_typecast_test_input(shape, pt_dtype, in_low, in_high)
    return torch.randn(shape, dtype=pt_dtype)


def _make_host_typecast_torch_input(shape, output_dtype):
    """Host typecast inputs; uint32 stays non-negative (host matches relu-style golden)."""
    in_low, in_high = typecast_test_input_bounds(ttnn.float32, output_dtype)
    if output_dtype == ttnn.uint32:
        in_low = 0
    return make_typecast_test_input(shape, tt_dtype_to_torch_dtype[ttnn.float32], in_low, in_high)


def _host_typecast_golden(torch_tensor, output_dtype):
    """Golden for host-resident typecast; host paths differ from device for some dtypes."""
    if output_dtype == ttnn.uint8:
        return torch.clamp(torch_tensor, 0, 255).to(torch.uint8)
    if output_dtype == ttnn.uint16:
        # Host fp32→uint16 truncates; device float_to_uint16 rounds in float32 first.
        return torch.clamp(torch_tensor.to(torch.int32), min=0, max=65535)
    return eltwise_typecast(torch_tensor, tt_input_dtype=ttnn.float32, tt_output_dtype=output_dtype)


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


@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64),),
)
def test_copy_uint16(N, C, H, W, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]

    input = torch.randint(0, 100, shape, dtype=torch.int16)
    input = ttnn.from_torch(input, ttnn.uint16, layout=ttnn.Layout.ROW_MAJOR, device=device)

    input_b = torch.zeros(shape, dtype=torch.int16)
    input_b = ttnn.from_torch(input_b, ttnn.uint16, layout=ttnn.Layout.ROW_MAJOR, device=device)

    ttnn.copy(input, input_b)
    assert input_b.shape == input.shape
    assert_equal(ttnn.to_torch(input), ttnn.to_torch(input_b))


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


@pytest.mark.parametrize(
    "to_dtype",
    (ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8),
    ids=["UINT32", "INT32", "UINT16", "UINT8"],
)
def test_typecast_output_tensor(to_dtype, device):
    """bf16→integer typecast on device: normal and preallocated output paths.

    Uses shared clamp helpers (typecast_test_input_bounds,
    make_typecast_test_input, eltwise_typecast, assert_integer_typecast_equal).
    Asserts both freshly allocated and preallocated outputs; preallocated path
    checks buffer-page count is unchanged.
    """
    torch.manual_seed(0)

    h = w = 32
    from_dtype = ttnn.bfloat16

    in_low, in_high = typecast_test_input_bounds(from_dtype, to_dtype)
    torch_input = make_typecast_test_input([h, w], torch.bfloat16, in_low, in_high)

    bfloat16_tensor = ttnn.from_torch(
        torch_input, dtype=from_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    preallocated = ttnn.empty([h, w], to_dtype, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)

    output_ttnn = ttnn.typecast(bfloat16_tensor, to_dtype, memory_config=ttnn.L1_MEMORY_CONFIG)

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.typecast(bfloat16_tensor, to_dtype, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensor=preallocated)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    torch_expected = eltwise_typecast(torch_input, tt_input_dtype=from_dtype, tt_output_dtype=to_dtype)
    assert_integer_typecast_equal(torch_expected, ttnn.to_torch(output_ttnn))
    assert_integer_typecast_equal(torch_expected, ttnn.to_torch(preallocated))


def test_typecast_community(device):
    shape = (1, 1, 32, 32)
    x = ttnn.to_device(ttnn.to_layout(ttnn.zeros(shape), ttnn.TILE_LAYOUT), device)
    y = ttnn.typecast(x, ttnn.types.bfloat16)
    assert y.dtype == ttnn.types.bfloat16
    assert_with_pcc(ttnn.to_torch(x), ttnn.to_torch(y), 0.99)


def run_typecast_row_major_test(shape, memory_config, input_dtype, output_dtype, device):
    """Test typecast operation on row-major device tensors."""
    torch.manual_seed(54321)
    torch_input = _make_typecast_torch_input(shape, input_dtype, output_dtype)

    # Create row-major tensor on device
    input_tensor = ttnn.from_torch(torch_input, input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Verify input is row-major
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input tensor should be in ROW_MAJOR_LAYOUT"

    # Perform typecast
    output_tensor = ttnn.typecast(input_tensor, dtype=output_dtype, memory_config=memory_config)

    # Verify output properties
    assert output_tensor.shape == input_tensor.shape, "Output shape should match input shape"
    assert output_tensor.dtype == output_dtype, "Output dtype should match requested dtype"
    assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Output should maintain ROW_MAJOR_LAYOUT"
    assert output_tensor.memory_config() == memory_config, "Output memory config should match"

    torch_output = ttnn.to_torch(output_tensor)

    if output_dtype in INTEGER_OUTPUT_DTYPES:
        torch_expected = eltwise_typecast(torch_input, tt_input_dtype=input_dtype, tt_output_dtype=output_dtype)
        assert_integer_typecast_equal(torch_expected, torch_output)
    else:
        torch_expected = torch_input.to(get_ttnn_torch_dtype(output_dtype))
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
        ttnn.uint8,
        ttnn.uint16,
        ttnn.uint32,
        ttnn.bfloat16,
        ttnn.float32,
    ),
    ids=[
        "INT32",
        "UINT8",
        "UINT16",
        "UINT32",
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
    - Integer outputs (int32, uint8, uint16, uint32): widened inputs + clamp-aware
      eltwise_typecast golden + assert_integer_typecast_equal
    - Float outputs: ULP check
    - Same input/output dtype pairs are skipped (bfloat16→bfloat16, float32→float32 only)
    """
    if input_dtype == output_dtype:
        pytest.skip("Input and output data types are the same")

    run_typecast_row_major_test(shape, memory_config, input_dtype, output_dtype, device)


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
def test_typecast_row_major_short_row_is_deterministic(input_dtype, output_dtype, device):
    """Regression for CB page overlap when a row is shorter than one hardware tile."""
    if input_dtype == output_dtype:
        pytest.skip("Input and output data types are the same")
    torch.manual_seed(54321)
    torch_input = torch.rand((1, 1, 361, 512), dtype=get_ttnn_torch_dtype(input_dtype))
    expected = torch_input.to(get_ttnn_torch_dtype(output_dtype))
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=input_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    first_output = None
    for run_idx in range(5):
        output = ttnn.typecast(
            input_tensor,
            dtype=output_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        torch_output = ttnn.to_torch(output)

        assert torch.equal(torch_output, expected), f"Incorrect output on run {run_idx}"
        if first_output is None:
            first_output = torch_output
        else:
            assert torch.equal(torch_output, first_output), f"Non-deterministic output on run {run_idx}"


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

    Host-resident fp32→uint8 clamps to [0, 255] (not wrap); fp32→uint16 truncates (device
    rounds); fp32→uint32 uses non-negative inputs so relu-style golden matches.
    """
    shape = [32, 2048]
    torch.manual_seed(2005)
    torch_tensor = _make_host_typecast_torch_input(shape, output_dtype)

    # Create host tensor
    ttnn_tensor_host = ttnn.from_torch(torch_tensor, dtype=ttnn.float32, layout=preferred_layout)

    # Verify input is on host
    assert ttnn_tensor_host.storage_type() == ttnn.StorageType.HOST

    # Perform typecast on host tensor
    ttnn_tensor_typecast = ttnn.typecast(ttnn_tensor_host, dtype=output_dtype)

    # Verify output properties
    assert ttnn_tensor_typecast.shape == ttnn_tensor_host.shape
    assert ttnn_tensor_typecast.dtype == output_dtype
    assert ttnn_tensor_typecast.storage_type() == ttnn.StorageType.HOST

    torch_output = ttnn.to_torch(ttnn_tensor_typecast)
    assert torch_output.shape == tuple(shape)

    if output_dtype in INTEGER_OUTPUT_DTYPES:
        torch_expected = _host_typecast_golden(torch_tensor, output_dtype)
        assert_integer_typecast_equal(torch_expected, torch_output)
    elif output_dtype == ttnn.bfloat16:
        torch_expected = torch_tensor.to(tt_dtype_to_torch_dtype[output_dtype])
        assert_with_ulp(torch_expected, torch_output, ulp_threshold=2)
