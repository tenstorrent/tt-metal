# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LayerNorm single-core generic op."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.layernorm.op import (
    CB_BETA_RM,
    CB_BETA_TILED,
    CB_GAMMA_RM,
    CB_GAMMA_TILED,
    CB_INPUT_RM,
    CB_INPUT_TILED,
    CB_INTERM,
    CB_OUTPUT_RM,
    CB_OUTPUT_TILED,
    CB_SCALARS,
    LayerNormSingleCore,
    _calculate_sizes,
    _create_cb_descriptors,
    _create_compute_descriptor,
    _create_reader_descriptor,
    _create_writer_descriptor,
)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 32],
        [1, 64],
        [4, 128],
    ],
)
def test_golden_vs_torch(shape):
    """
    Test that LayerNormSingleCore.golden() matches torch.nn.functional.layer_norm().

    Pass criteria: torch.allclose(golden, torch_ref, rtol=1e-4, atol=1e-4) for all shapes.
    """
    torch.manual_seed(42)

    # Create input tensor with random values
    input_tensor = torch.randn(shape, dtype=torch.float32)

    # Create gamma (scale) and beta (shift) parameters with shape matching last dimension
    W = shape[-1]
    gamma_tensor = torch.randn(W, dtype=torch.float32)
    beta_tensor = torch.randn(W, dtype=torch.float32)

    epsilon = 1e-6

    # Compute using our golden implementation
    golden_output = LayerNormSingleCore.golden(input_tensor, gamma_tensor, beta_tensor, epsilon)

    # Compute using torch.nn.functional.layer_norm (reference)
    torch_output = torch.nn.functional.layer_norm(
        input_tensor,
        normalized_shape=[W],
        weight=gamma_tensor,
        bias=beta_tensor,
        eps=epsilon,
    )

    # Verify outputs match
    assert torch.allclose(golden_output, torch_output, rtol=1e-4, atol=1e-4), (
        f"Golden output does not match torch.nn.functional.layer_norm for shape {shape}\n"
        f"Max absolute difference: {(golden_output - torch_output).abs().max().item()}"
    )


# =============================================================================
# Step 1.3.4: CB Configuration Tests
# =============================================================================


def test_cb_indices():
    """
    Test that CB indices are correctly defined and unique.
    """
    # Verify input/working CBs are in range 0-15
    input_cbs = [
        CB_INPUT_RM,
        CB_INPUT_TILED,
        CB_GAMMA_RM,
        CB_GAMMA_TILED,
        CB_BETA_RM,
        CB_BETA_TILED,
        CB_SCALARS,
        CB_INTERM,
    ]
    for cb in input_cbs:
        assert 0 <= cb <= 15, f"Input CB {cb} should be in range 0-15"

    # Verify output CBs are >= 16
    output_cbs = [CB_OUTPUT_TILED, CB_OUTPUT_RM]
    for cb in output_cbs:
        assert cb >= 16, f"Output CB {cb} should be >= 16"

    # Verify all CBs are unique
    all_cbs = input_cbs + output_cbs
    assert len(all_cbs) == len(set(all_cbs)), "CB indices must be unique"


@pytest.mark.parametrize(
    "shape,expected_tiles_per_row",
    [
        ([1, 32], 1),  # Single tile width
        ([1, 64], 2),  # Two tiles
        ([4, 128], 4),  # Four tiles
        ([2, 2, 256], 8),  # 3D input, 8 tiles
    ],
)
def test_calculate_sizes(shape, expected_tiles_per_row):
    """
    Test that _calculate_sizes correctly computes buffer dimensions.
    """
    dtype = ttnn.bfloat16
    sizes = _calculate_sizes(shape, dtype)

    # Verify W (final dimension)
    assert sizes["W"] == shape[-1], f"W should be {shape[-1]}"

    # Verify num_rows (product of all dims except last)
    expected_num_rows = 1
    for dim in shape[:-1]:
        expected_num_rows *= dim
    assert sizes["num_rows"] == expected_num_rows, f"num_rows should be {expected_num_rows}"

    # Verify tiles_per_row
    assert sizes["tiles_per_row"] == expected_tiles_per_row, f"tiles_per_row should be {expected_tiles_per_row}"

    # Verify stick_size is properly aligned to 32 bytes
    assert sizes["stick_size"] % 32 == 0, "stick_size must be aligned to 32 bytes"
    assert sizes["stick_size"] >= shape[-1] * sizes["element_size"], "stick_size must hold at least W elements"

    # Verify tile_size (32x32 * element_size)
    assert sizes["tile_size"] == 32 * 32 * sizes["element_size"], "tile_size should be 32*32*element_size"

    # Verify element_size for bfloat16
    assert sizes["element_size"] == 2, "bfloat16 element_size should be 2"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 32],
        [1, 64],
        [4, 128],
    ],
)
def test_cb_configuration(shape):
    """
    Test that CB descriptors are created correctly without errors.

    Verifies:
    - All 10 CBs are created
    - Sizes are positive
    - Sizes are properly aligned
    """
    dtype = ttnn.bfloat16
    sizes = _calculate_sizes(shape, dtype)

    # Create a single core grid
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # Create CB descriptors
    cb_descriptors = _create_cb_descriptors(core_grid, sizes, dtype)

    # Verify we have exactly 10 CB descriptors
    assert len(cb_descriptors) == 10, f"Expected 10 CB descriptors, got {len(cb_descriptors)}"

    # Verify each CB descriptor has positive, properly sized buffers
    stick_size = sizes["stick_size"]
    tile_size = sizes["tile_size"]
    tiles_per_row = sizes["tiles_per_row"]

    expected_sizes = {
        CB_INPUT_RM: 2 * stick_size,  # Double buffer
        CB_INPUT_TILED: tiles_per_row * tile_size,
        CB_GAMMA_RM: stick_size,  # Single buffer
        CB_GAMMA_TILED: tiles_per_row * tile_size,
        CB_BETA_RM: stick_size,  # Single buffer
        CB_BETA_TILED: tiles_per_row * tile_size,
        CB_SCALARS: tile_size,  # 1 tile
        CB_INTERM: tiles_per_row * tile_size,
        CB_OUTPUT_TILED: tiles_per_row * tile_size,
        CB_OUTPUT_RM: 2 * stick_size,  # Double buffer
    }

    for cb_desc in cb_descriptors:
        # Check that total_size is positive
        assert cb_desc.total_size > 0, f"CB total_size must be positive"

        # Check that format_descriptors has at least one entry
        assert len(cb_desc.format_descriptors) >= 1, "CB must have at least one format descriptor"

        # Get buffer_index from format descriptor
        buffer_index = cb_desc.format_descriptors[0].buffer_index

        # Verify total_size matches expected
        if buffer_index in expected_sizes:
            assert (
                cb_desc.total_size == expected_sizes[buffer_index]
            ), f"CB {buffer_index} total_size should be {expected_sizes[buffer_index]}, got {cb_desc.total_size}"


# =============================================================================
# Step 1.4.5: Kernel Descriptor Tests
# =============================================================================


@pytest.fixture
def device():
    """Get a device for testing."""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


def test_reader_descriptor(device):
    """
    Test that reader kernel descriptor is created correctly.

    Verifies:
    - Descriptor is created without errors
    - Compile-time args are populated (CB indices, sizes, TensorAccessorArgs)
    - Runtime args are populated (buffer addresses, start_stick_id)
    """
    torch.manual_seed(42)

    # Test shape
    shape = [4, 128]
    W = shape[-1]
    dtype = ttnn.bfloat16

    # Create torch tensors
    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(W, dtype=torch.bfloat16)
    beta_torch = torch.randn(W, dtype=torch.bfloat16)

    # Create device tensors (row-major, DRAM interleaved)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        gamma_torch.reshape(1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        beta_torch.reshape(1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Calculate sizes
    sizes = _calculate_sizes(shape, dtype)

    # Create core grid (single core)
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # Create reader descriptor
    reader_desc = _create_reader_descriptor(core_grid, input_tensor, gamma_tensor, beta_tensor, sizes)

    # Verify descriptor was created
    assert reader_desc is not None, "Reader descriptor should not be None"

    # Verify compile-time args are populated
    # At minimum: 5 base args + TensorAccessorArgs for 3 tensors
    assert (
        len(reader_desc.compile_time_args) >= 5
    ), f"Compile-time args should have at least 5 elements, got {len(reader_desc.compile_time_args)}"

    # Verify first 5 compile-time args match expected values
    assert reader_desc.compile_time_args[0] == CB_INPUT_RM, "First compile-time arg should be CB_INPUT_RM"
    assert reader_desc.compile_time_args[1] == CB_GAMMA_RM, "Second compile-time arg should be CB_GAMMA_RM"
    assert reader_desc.compile_time_args[2] == CB_BETA_RM, "Third compile-time arg should be CB_BETA_RM"
    assert reader_desc.compile_time_args[3] == sizes["stick_size"], "Fourth compile-time arg should be stick_size"
    assert reader_desc.compile_time_args[4] == sizes["num_rows"], "Fifth compile-time arg should be num_rows"

    # Verify runtime args are populated for core (0, 0)
    # Runtime args should have buffer addresses
    rt_args = reader_desc.runtime_args[0][0]
    assert len(rt_args) >= 4, f"Runtime args should have at least 4 elements, got {len(rt_args)}"

    # Verify buffer addresses are non-zero
    assert rt_args[0] > 0, "Input buffer address should be non-zero"
    assert rt_args[1] > 0, "Gamma buffer address should be non-zero"
    assert rt_args[2] > 0, "Beta buffer address should be non-zero"
    assert rt_args[3] == 0, "start_stick_id should be 0 for single core"


def test_compute_descriptor(device):
    """
    Test that compute kernel descriptor is created correctly.

    Verifies:
    - Descriptor is created without errors
    - Compile-time args include all CB indices, tiles_per_row, num_rows, W
    - Runtime args contain epsilon (packed as uint32)
    """
    torch.manual_seed(42)

    # Test shape
    shape = [4, 128]
    dtype = ttnn.bfloat16
    epsilon = 1e-6

    # Calculate sizes
    sizes = _calculate_sizes(shape, dtype)

    # Create core grid (single core)
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # Create compute descriptor
    compute_desc = _create_compute_descriptor(core_grid, sizes, epsilon)

    # Verify descriptor was created
    assert compute_desc is not None, "Compute descriptor should not be None"

    # Verify compile-time args are populated
    # Expected: 10 CB indices + 3 size params = 13 args
    assert (
        len(compute_desc.compile_time_args) >= 13
    ), f"Compile-time args should have at least 13 elements, got {len(compute_desc.compile_time_args)}"

    # Verify CB index compile-time args
    assert compute_desc.compile_time_args[0] == CB_INPUT_RM, "Arg 0 should be CB_INPUT_RM"
    assert compute_desc.compile_time_args[1] == CB_INPUT_TILED, "Arg 1 should be CB_INPUT_TILED"
    assert compute_desc.compile_time_args[2] == CB_GAMMA_RM, "Arg 2 should be CB_GAMMA_RM"
    assert compute_desc.compile_time_args[3] == CB_GAMMA_TILED, "Arg 3 should be CB_GAMMA_TILED"
    assert compute_desc.compile_time_args[4] == CB_BETA_RM, "Arg 4 should be CB_BETA_RM"
    assert compute_desc.compile_time_args[5] == CB_BETA_TILED, "Arg 5 should be CB_BETA_TILED"
    assert compute_desc.compile_time_args[6] == CB_SCALARS, "Arg 6 should be CB_SCALARS"
    assert compute_desc.compile_time_args[7] == CB_INTERM, "Arg 7 should be CB_INTERM"
    assert compute_desc.compile_time_args[8] == CB_OUTPUT_TILED, "Arg 8 should be CB_OUTPUT_TILED"
    assert compute_desc.compile_time_args[9] == CB_OUTPUT_RM, "Arg 9 should be CB_OUTPUT_RM"

    # Verify size parameters
    assert compute_desc.compile_time_args[10] == sizes["tiles_per_row"], "Arg 10 should be tiles_per_row"
    assert compute_desc.compile_time_args[11] == sizes["num_rows"], "Arg 11 should be num_rows"
    assert compute_desc.compile_time_args[12] == sizes["W"], "Arg 12 should be W"

    # Verify runtime args are populated for core (0, 0)
    rt_args = compute_desc.runtime_args[0][0]
    assert len(rt_args) >= 1, f"Runtime args should have at least 1 element, got {len(rt_args)}"

    # Verify epsilon is packed as uint32 (non-zero value)
    assert rt_args[0] > 0, "Epsilon (packed as uint32) should be non-zero"


def test_writer_descriptor(device):
    """
    Test that writer kernel descriptor is created correctly.

    Verifies:
    - Descriptor is created without errors
    - Compile-time args include CB index, sizes, TensorAccessorArgs
    - Runtime args contain buffer address and start_stick_id
    """
    torch.manual_seed(42)

    # Test shape
    shape = [4, 128]
    dtype = ttnn.bfloat16

    # Create torch tensor and device tensor for output
    output_torch = torch.zeros(shape, dtype=torch.bfloat16)
    output_tensor = ttnn.from_torch(
        output_torch,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Calculate sizes
    sizes = _calculate_sizes(shape, dtype)

    # Create core grid (single core)
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # Create writer descriptor
    writer_desc = _create_writer_descriptor(core_grid, output_tensor, sizes)

    # Verify descriptor was created
    assert writer_desc is not None, "Writer descriptor should not be None"

    # Verify compile-time args are populated
    # At minimum: CB_OUTPUT_RM + stick_size + num_rows + TensorAccessorArgs
    assert (
        len(writer_desc.compile_time_args) >= 3
    ), f"Compile-time args should have at least 3 elements, got {len(writer_desc.compile_time_args)}"

    # Verify first 3 compile-time args match expected values
    assert writer_desc.compile_time_args[0] == CB_OUTPUT_RM, "First compile-time arg should be CB_OUTPUT_RM"
    assert writer_desc.compile_time_args[1] == sizes["stick_size"], "Second compile-time arg should be stick_size"
    assert writer_desc.compile_time_args[2] == sizes["num_rows"], "Third compile-time arg should be num_rows"

    # Verify runtime args are populated for core (0, 0)
    rt_args = writer_desc.runtime_args[0][0]
    assert len(rt_args) >= 2, f"Runtime args should have at least 2 elements, got {len(rt_args)}"

    # Verify buffer address is non-zero
    assert rt_args[0] > 0, "Output buffer address should be non-zero"
    assert rt_args[1] == 0, "start_stick_id should be 0 for single core"


def test_kernel_descriptors(device):
    """
    Test that all kernel descriptors (reader, compute, writer) are created correctly.

    This is the main test for Step 1.4.5, verifying the complete kernel descriptor setup.

    Verifies:
    - All three kernel descriptors are created without errors
    - Compile-time args are properly populated for each
    - Runtime args are properly populated for each
    """
    torch.manual_seed(42)

    # Test shape
    shape = [4, 128]
    W = shape[-1]
    dtype = ttnn.bfloat16
    epsilon = 1e-6

    # Create torch tensors
    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(W, dtype=torch.bfloat16)
    beta_torch = torch.randn(W, dtype=torch.bfloat16)
    output_torch = torch.zeros(shape, dtype=torch.bfloat16)

    # Create device tensors (row-major, DRAM interleaved)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        gamma_torch.reshape(1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        beta_torch.reshape(1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.from_torch(
        output_torch,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Calculate sizes
    sizes = _calculate_sizes(shape, dtype)

    # Create core grid (single core)
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # Create all kernel descriptors
    reader_desc = _create_reader_descriptor(core_grid, input_tensor, gamma_tensor, beta_tensor, sizes)
    compute_desc = _create_compute_descriptor(core_grid, sizes, epsilon)
    writer_desc = _create_writer_descriptor(core_grid, output_tensor, sizes)

    # Verify all descriptors were created successfully
    assert reader_desc is not None, "Reader descriptor should not be None"
    assert compute_desc is not None, "Compute descriptor should not be None"
    assert writer_desc is not None, "Writer descriptor should not be None"

    # Verify each descriptor has kernel source set
    assert reader_desc.kernel_source is not None, "Reader kernel source should be set"
    assert compute_desc.kernel_source is not None, "Compute kernel source should be set"
    assert writer_desc.kernel_source is not None, "Writer kernel source should be set"

    # Verify each descriptor has compile-time args
    assert len(reader_desc.compile_time_args) > 0, "Reader should have compile-time args"
    assert len(compute_desc.compile_time_args) > 0, "Compute should have compile-time args"
    assert len(writer_desc.compile_time_args) > 0, "Writer should have compile-time args"

    # Verify each descriptor has runtime args for core (0, 0)
    assert len(reader_desc.runtime_args[0][0]) > 0, "Reader should have runtime args"
    assert len(compute_desc.runtime_args[0][0]) > 0, "Compute should have runtime args"
    assert len(writer_desc.runtime_args[0][0]) > 0, "Writer should have runtime args"

    # Verify kernel configs are correct types
    # Reader should use ReaderConfigDescriptor
    # Compute should use ComputeConfigDescriptor
    # Writer should use WriterConfigDescriptor
    assert reader_desc.config is not None, "Reader config should be set"
    assert compute_desc.config is not None, "Compute config should be set"
    assert writer_desc.config is not None, "Writer config should be set"


# =============================================================================
# Step 1.5.2: Program Execution Test (placeholder kernels)
# =============================================================================


def test_program_executes(device):
    """
    Test that the program descriptor executes without errors using placeholder kernels.

    This test verifies the complete pipeline:
    - Input, gamma, beta tensors are created on device (row-major, DRAM)
    - Output tensor is allocated on device
    - LayerNormSingleCore.op() executes without exceptions
    - No kernel compilation errors occur

    Pass criteria: No exceptions, no kernel compilation errors.

    Note: With placeholder kernels, the output will not contain correct LayerNorm
    results. This test only verifies the infrastructure is correctly assembled.
    """
    torch.manual_seed(42)

    # Test shape - using a simple shape for the first execution test
    shape = [1, 32]  # Single row, single tile width
    W = shape[-1]
    dtype = ttnn.bfloat16

    # Create torch tensors
    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(W, dtype=torch.bfloat16)
    beta_torch = torch.randn(W, dtype=torch.bfloat16)
    output_torch = torch.zeros(shape, dtype=torch.bfloat16)

    # Create device tensors (row-major, DRAM interleaved)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        gamma_torch.reshape(1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        beta_torch.reshape(1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.from_torch(
        output_torch,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    epsilon = 1e-6

    # Execute the LayerNorm op - this should compile and run without errors
    # With placeholder kernels, the output won't be correct, but the infrastructure
    # should work
    result = LayerNormSingleCore.op(input_tensor, gamma_tensor, beta_tensor, output_tensor, epsilon)

    # Verify we got a result tensor back
    assert result is not None, "LayerNormSingleCore.op() should return a tensor"

    # Verify result shape matches input shape
    result_shape = list(result.shape)
    assert result_shape == shape, f"Result shape {result_shape} should match input shape {shape}"

    # Note: We do NOT verify numerical correctness here - that's for Stage 2 tests
    # This test only verifies the program infrastructure works


# =============================================================================
# Step 2.1.1: Reader/Writer Passthrough Test
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [
        [1, 32],  # Single row, single tile width
        [1, 64],  # Single row, two tiles
        [4, 128],  # 4 rows, 4 tiles
    ],
)
def test_reader_writer_passthrough(device, shape):
    """
    Test that reader reads data correctly and the program executes without hanging.

    This test verifies Step 2.1.1 (basic input reading) by:
    1. Creating an input tensor with known values
    2. Running the operation with the minimal passthrough compute kernel
    3. Verifying the program completes without hanging or errors

    Pass criteria: Program completes without timeout or exceptions.

    Note: This test uses a minimal passthrough compute kernel that verifies
    the reader can push data to CB_INPUT_RM and the writer can read from CB_OUTPUT_RM.
    The output data is NOT the same as input (no tilize/untilize copy yet).
    Full data passthrough will be tested in Step 2.2.1 (tilize/untilize passthrough).
    """
    torch.manual_seed(42)

    W = shape[-1]
    dtype = ttnn.bfloat16

    # Create input tensor with known values
    input_torch = torch.arange(1, shape[0] * shape[1] + 1, dtype=torch.float32).reshape(shape)
    input_torch = input_torch / (shape[0] * shape[1])  # Normalize to [0, 1] range for bfloat16
    input_torch = input_torch.to(torch.bfloat16)

    # Gamma and beta (not used in passthrough, but required by the op)
    gamma_torch = torch.ones(W, dtype=torch.bfloat16)
    beta_torch = torch.zeros(W, dtype=torch.bfloat16)
    output_torch = torch.zeros(shape, dtype=torch.bfloat16)

    # Create device tensors
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        gamma_torch.reshape(1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        beta_torch.reshape(1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.from_torch(
        output_torch,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    epsilon = 1e-6

    # Execute the LayerNorm op (passthrough mode)
    # This verifies the reader can read data and push to CB, and writer can consume from CB
    result = LayerNormSingleCore.op(input_tensor, gamma_tensor, beta_tensor, output_tensor, epsilon)

    # Verify we got a result tensor back with correct shape
    assert result is not None, "LayerNormSingleCore.op() should return a tensor"
    result_shape = list(result.shape)
    assert result_shape == shape, f"Result shape {result_shape} should match input shape {shape}"

    # Note: We do NOT check that output equals input here because the current
    # passthrough compute kernel doesn't copy data (no tilize/untilize yet).
    # That will be verified in Step 2.2.1 (tilize/untilize passthrough test).
