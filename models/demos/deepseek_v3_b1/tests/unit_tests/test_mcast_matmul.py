# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Mcast Matmul Unit Tests

Tests for multi-core matmul with mcast input distribution pattern:
- Input activations on a dedicated mcast core
- Weights WIDTH_SHARDED across compute cores
- Mcast broadcasts input to all compute cores
- Each core computes local matmul with its weight shard

Test configurations:
- Various grid sizes (8-core, 9-core, 48-core, 72-core)
- Different weight precisions (bfloat16, bfloat8_b, bfloat4_b)
- Origin and non-origin grid placements

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_gate_mcast_matmul_72c.py -v
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.mcast_matmul.op import McastMatmulMultiCore

DEFAULT_MCAST_CORE = ttnn.CoreCoord(11, 9)


def run_mcast_matmul_test(
    device,
    M,
    K,
    N,
    grid_start,
    grid_end,
    mcast_core,
    in0_dtype,
    in1_dtype,
    pcc_threshold=0.98,
):
    """
    Run a single mcast matmul test with the given configuration.

    Args:
        device: TT device
        M, K, N: Matrix dimensions [M, K] × [K, N] → [M, N]
        grid_start: Start coordinate of compute grid
        grid_end: End coordinate of compute grid
        mcast_core: Core for input tensor and mcast source
        in0_dtype: Input tensor dtype
        in1_dtype: Weights tensor dtype
        pcc_threshold: Minimum PCC for test to pass

    Returns:
        Tuple of (passing, pcc_value)
    """
    # Calculate grid dimensions
    num_cores_x = grid_end.x - grid_start.x + 1
    num_cores_y = grid_end.y - grid_start.y + 1
    num_cores = num_cores_x * num_cores_y
    N_per_core = N // num_cores

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    logger.info(f"  Shape: [{M}, {K}] × [{K}, {N}] → [{M}, {N}]")
    logger.info(f"  Grid: ({grid_start.x},{grid_start.y}) to ({grid_end.x},{grid_end.y}) = {num_cores} cores")
    logger.info(f"  Per-core: [{M}, {K}] × [{K}, {N_per_core}] → [{M}, {N_per_core}]")
    logger.info(f"  Mcast core: ({mcast_core.x}, {mcast_core.y})")
    logger.info(f"  Dtypes: input={in0_dtype}, weights={in1_dtype}")

    # Create test tensors
    torch.manual_seed(42)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)

    # Golden reference
    torch_expected = McastMatmulMultiCore.golden(torch_input.float(), torch_weights.float()).bfloat16()

    # Input tensor: HEIGHT_SHARDED on mcast core
    mcast_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)})
    input_shard_spec = ttnn.ShardSpec(
        mcast_core_grid,
        (M, K),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        input_shard_spec,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=a_tile,
    )

    # Weights tensor: WIDTH_SHARDED across compute grid
    matmul_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(grid_start, grid_end)})
    weights_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        (K, N_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        weights_shard_spec,
    )

    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=b_tile,
    )

    # Output tensor: WIDTH_SHARDED across same compute grid
    output_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        (M, N_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )

    torch_output_zeros = torch.zeros((M, N), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    # Run mcast matmul
    logger.info("  Running mcast matmul...")
    ttnn_result = McastMatmulMultiCore.op(
        ttnn_input,
        ttnn_weights,
        ttnn_output,
        fp32_dest_acc_en=False,
    )

    # Verify
    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == (M, N), f"Expected shape ({M}, {N}), got {output_torch.shape}"

    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(f"  {pcc_message}")

    return passing, pcc_message


@pytest.mark.parametrize(
    "M, K, N, num_cores_x, num_cores_y, in1_dtype, pcc_threshold, description",
    [
        # Small test case for quick validation
        (1, 256, 256, 2, 4, ttnn.bfloat8_b, 0.98, "small 8-core"),
        # Scaled test case
        (1, 896, 288, 3, 3, ttnn.bfloat8_b, 0.98, "scaled 9-core"),
        # Full gate dimensions (72 cores)
        pytest.param(1, 7168, 2304, 9, 8, ttnn.bfloat8_b, 0.98, "gate 72-core bfp8", marks=pytest.mark.slow),
        # Full gate with bfloat4_b weights
        pytest.param(1, 7168, 2304, 9, 8, ttnn.bfloat4_b, 0.90, "gate 72-core bfp4", marks=pytest.mark.slow),
        # q_b_proj dimensions (48 cores)
        pytest.param(1, 7168, 1536, 8, 6, ttnn.bfloat8_b, 0.98, "q_b_proj 48-core", marks=pytest.mark.slow),
    ],
)
def test_mcast_matmul_origin_grid(device, M, K, N, num_cores_x, num_cores_y, in1_dtype, pcc_threshold, description):
    """
    Test mcast matmul with compute grid starting at origin (0,0).

    This is the standard configuration where the compute grid occupies
    the top-left portion of the device grid.
    """
    # Adjust mcast core for device grid
    device_grid = device.compute_with_storage_grid_size()
    mcast_core = DEFAULT_MCAST_CORE
    if mcast_core.x >= device_grid.x:
        mcast_core = ttnn.CoreCoord(device_grid.x - 1, mcast_core.y)
    if mcast_core.y >= device_grid.y:
        mcast_core = ttnn.CoreCoord(mcast_core.x, device_grid.y - 1)

    # Adjust grid for device
    num_cores_x = min(num_cores_x, device_grid.x)
    num_cores_y = min(num_cores_y, device_grid.y)

    grid_start = ttnn.CoreCoord(0, 0)
    grid_end = ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1)

    logger.info(f"Testing mcast matmul: {description}")
    passing, pcc_message = run_mcast_matmul_test(
        device, M, K, N, grid_start, grid_end, mcast_core, ttnn.bfloat16, in1_dtype, pcc_threshold
    )

    assert passing, pcc_message
    logger.info(f"✓ {description} passed")


"""
def test_mcast_matmul_column_grid(device):
    M, K, N = 1, 2304, 2048

    # Grid: single column at x=8
    grid_start = ttnn.CoreCoord(8, 0)
    grid_end = ttnn.CoreCoord(8, 7)
    mcast_core = ttnn.CoreCoord(11, 9)

    # Check device grid size
    device_grid = device.compute_with_storage_grid_size()
    if grid_end.x >= device_grid.x or grid_end.y >= device_grid.y:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small")
    if mcast_core.x >= device_grid.x or mcast_core.y >= device_grid.y:
        mcast_core = ttnn.CoreCoord(min(mcast_core.x, device_grid.x - 1), min(mcast_core.y, device_grid.y - 1))

    logger.info("Testing mcast matmul: column grid at x=8")
    passing, pcc_message = run_mcast_matmul_test(
        device, M, K, N, grid_start, grid_end, mcast_core, ttnn.bfloat16, ttnn.bfloat16, 0.98
    )

    assert passing, pcc_message
    logger.info("✓ Column grid test passed")


@pytest.mark.parametrize(
    "M, K, N",
    [
        (1, 256, 256),
        (1, 896, 288),
    ],
)
def test_mcast_matmul_golden(M, K, N):
    torch.manual_seed(42)

    torch_input = torch.randn((M, K), dtype=torch.float32)
    torch_weights = torch.randn((K, N), dtype=torch.float32)

    expected = torch_input @ torch_weights
    result = McastMatmulMultiCore.golden(torch_input, torch_weights)

    assert result.shape == expected.shape
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)

    logger.info(f"✓ Golden test passed: [{M}, {K}] × [{K}, {N}]")

"""


# =============================================================================
# Mcast to Many, Compute on Few - Phantom Core Tests
#
# These tests demonstrate the pattern where:
# - Mcast sends data to a LARGER grid (mcast_grid)
# - Only a SUBSET of cores (matmul_grid) actually perform computation
# - "Phantom cores" (in mcast_grid but not matmul_grid) must still:
#   1. Receive the mcast data (hardware writes to their L1)
#   2. Wait on and reset the mcast semaphore (so sender can proceed)
#   3. NOT do CB operations or compute (no weights, no output buffer)
#
# This is useful when the mcast destination must be a rectangular bounding box
# but the actual compute grid is non-rectangular or smaller.
# =============================================================================


def run_mcast_matmul_with_phantom_cores_test(
    device,
    M,
    K,
    N,
    mcast_grid_start,
    mcast_grid_end,
    matmul_grid_start,
    matmul_grid_end,
    mcast_core,
    in0_dtype,
    in1_dtype,
    pcc_threshold=0.98,
):
    """
    Run a mcast matmul test where the mcast grid is larger than the matmul grid.

    The mcast grid defines the rectangular bounding box that receives data.
    The matmul grid defines the subset of cores that actually compute.
    Cores in (mcast_grid - matmul_grid) are "phantom cores" that receive data
    but don't participate in computation.

    Args:
        device: TT device
        M, K, N: Matrix dimensions [M, K] × [K, N] → [M, N]
        mcast_grid_start: Start coordinate of mcast bounding box
        mcast_grid_end: End coordinate of mcast bounding box
        matmul_grid_start: Start coordinate of compute grid (subset of mcast grid)
        matmul_grid_end: End coordinate of compute grid
        mcast_core: Core for input tensor and mcast source
        in0_dtype: Input tensor dtype
        in1_dtype: Weights tensor dtype
        pcc_threshold: Minimum PCC for test to pass

    Returns:
        Tuple of (passing, pcc_value)
    """
    from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
        UnifiedCompileTimeCoreDescriptor,
        UnifiedKernelDescriptor,
    )

    # Calculate grid dimensions
    mcast_num_cores_x = mcast_grid_end.x - mcast_grid_start.x + 1
    mcast_num_cores_y = mcast_grid_end.y - mcast_grid_start.y + 1
    mcast_num_cores = mcast_num_cores_x * mcast_num_cores_y

    matmul_num_cores_x = matmul_grid_end.x - matmul_grid_start.x + 1
    matmul_num_cores_y = matmul_grid_end.y - matmul_grid_start.y + 1
    matmul_num_cores = matmul_num_cores_x * matmul_num_cores_y

    phantom_num_cores = mcast_num_cores - matmul_num_cores

    N_per_core = N // matmul_num_cores

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    logger.info(f"  Shape: [{M}, {K}] × [{K}, {N}] → [{M}, {N}]")
    logger.info(
        f"  Mcast grid: ({mcast_grid_start.x},{mcast_grid_start.y}) to ({mcast_grid_end.x},{mcast_grid_end.y}) = {mcast_num_cores} cores"
    )
    logger.info(
        f"  Matmul grid: ({matmul_grid_start.x},{matmul_grid_start.y}) to ({matmul_grid_end.x},{matmul_grid_end.y}) = {matmul_num_cores} cores"
    )
    logger.info(f"  Phantom cores: {phantom_num_cores} (in mcast grid but not matmul grid)")
    logger.info(f"  Per-core: [{M}, {K}] × [{K}, {N_per_core}] → [{M}, {N_per_core}]")
    logger.info(f"  Mcast source core: ({mcast_core.x}, {mcast_core.y})")
    logger.info(f"  Dtypes: input={in0_dtype}, weights={in1_dtype}")

    # Create test tensors
    torch.manual_seed(42)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)

    # Golden reference
    torch_expected = McastMatmulMultiCore.golden(torch_input.float(), torch_weights.float()).bfloat16()

    # Input tensor: HEIGHT_SHARDED on mcast source core
    mcast_source_grid = ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)})
    input_shard_spec = ttnn.ShardSpec(
        mcast_source_grid,
        (M, K),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        input_shard_spec,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=a_tile,
    )

    # Weights tensor: WIDTH_SHARDED across MATMUL grid (not full mcast grid!)
    matmul_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(matmul_grid_start, matmul_grid_end)})
    weights_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        (K, N_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        weights_shard_spec,
    )

    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=b_tile,
    )

    # Output tensor: WIDTH_SHARDED across MATMUL grid
    output_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        (M, N_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )

    torch_output_zeros = torch.zeros((M, N), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    # =========================================================================
    # Build program descriptor with explicit phantom core handling
    # =========================================================================

    data_format = ttnn_input.dtype
    in0_tile = ttnn_input.get_tile()
    in1_tile = ttnn_weights.get_tile()

    input_shard_shape = input_mem_config.shard_spec.shape
    output_shard_shape = output_mem_config.shard_spec.shape

    k_num_tiles = input_shard_shape[1] // in0_tile.tile_shape[1]
    out_w_per_core = output_shard_shape[1] // out_tile.tile_shape[1]

    # Mcast grid = the full bounding box
    mcast_grid = ttnn.CoreRange(mcast_grid_start, mcast_grid_end)

    # Check if sender is part of mcast grid
    is_sender_in_mcast_grid = mcast_grid.contains(mcast_core)

    # Get NOC coordinates for mcast destination
    mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
    mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)

    # Compute phantom cores: cores in mcast grid but not in matmul grid
    phantom_cores = []
    for x in range(mcast_grid_start.x, mcast_grid_end.x + 1):
        for y in range(mcast_grid_start.y, mcast_grid_end.y + 1):
            core = ttnn.CoreCoord(x, y)
            core_range = ttnn.CoreRange(core, core)
            is_sender = x == mcast_core.x and y == mcast_core.y
            if not matmul_core_grid.contains(core_range) and not is_sender:
                phantom_cores.append(core)

    # Build phantom core grid if any exist
    phantom_core_grid = None
    if phantom_cores:
        phantom_core_ranges = [ttnn.CoreRange(c, c) for c in phantom_cores]
        phantom_core_grid = ttnn.CoreRangeSet(phantom_core_ranges)
        logger.info(f"  Phantom cores: {[f'({c.x},{c.y})' for c in phantom_cores]}")

    # Calculate data sizes
    input_tile_size = in0_tile.get_tile_size(data_format)
    mcast_data_size_bytes = k_num_tiles * input_tile_size

    # Semaphore IDs
    mcast_sender_semaphore_id = 0
    mcast_receiver_semaphore_id = 1

    # Get full device grid for semaphores
    device_grid_size = device.compute_with_storage_grid_size()
    full_device_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
    )

    # CB indices
    src_cb = 0
    dst_cb = 1
    in1_cb = 2
    out_cb = 3

    # =========================================================================
    # CB descriptors
    # =========================================================================

    sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])

    # CB 0: Source input (on sender core)
    src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(src_cb, ttnn_input)

    # CB 0 placeholder on matmul + phantom cores for consistent L1 layout
    placeholder_core_ranges = matmul_core_grid
    if phantom_core_grid is not None:
        placeholder_core_ranges = placeholder_core_ranges.merge(phantom_core_grid)
    src_cb_placeholder_format = ttnn.CBFormatDescriptor(
        buffer_index=src_cb,
        data_format=data_format,
        page_size=input_tile_size,
        tile=ttnn.TileDescriptor(in0_tile),
    )
    src_cb_placeholder_descriptor = ttnn.CBDescriptor(
        total_size=k_num_tiles * input_tile_size,
        core_ranges=placeholder_core_ranges,
        format_descriptors=[src_cb_placeholder_format],
    )

    # CB 1: Mcast destination (on matmul + sender + phantom cores)
    dst_cb_core_ranges = matmul_core_grid.merge(sender_core_grid)
    if phantom_core_grid is not None:
        dst_cb_core_ranges = dst_cb_core_ranges.merge(phantom_core_grid)
    dst_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=dst_cb,
        data_format=data_format,
        page_size=input_tile_size,
        tile=ttnn.TileDescriptor(in0_tile),
    )
    dst_cb_descriptor = ttnn.CBDescriptor(
        total_size=k_num_tiles * input_tile_size,
        core_ranges=dst_cb_core_ranges,
        format_descriptors=[dst_cb_format],
    )

    # CB 2: Weights (only on matmul cores)
    in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in1_cb, ttnn_weights)

    # CB 3: Output (only on matmul cores)
    out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, ttnn_output)

    # =========================================================================
    # Semaphore descriptors
    # =========================================================================

    sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
        id=mcast_sender_semaphore_id,
        core_ranges=full_device_grid,
        initial_value=0,
    )

    receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
        id=mcast_receiver_semaphore_id,
        core_ranges=full_device_grid,
        initial_value=0,
    )

    # =========================================================================
    # Compile-time args
    # =========================================================================

    ncrisc_named_compile_time_args = [
        ("mcast_src_cb", src_cb),
        ("mcast_src_num_pages", k_num_tiles),
        ("mcast_data_receiver_semaphore", mcast_receiver_semaphore_id),
        ("mcast_dst_cb", dst_cb),
        ("mcast_dst_num_pages", k_num_tiles),
        ("matmul_in1", in1_cb),
        ("matmul_in1_num_pages", k_num_tiles * out_w_per_core),
    ]

    brisc_named_compile_time_args = [
        ("mcast_dest_noc_start_x", mcast_dest_noc_start.x),
        ("mcast_dest_noc_start_y", mcast_dest_noc_start.y),
        ("mcast_dest_noc_end_x", mcast_dest_noc_end.x),
        ("mcast_dest_noc_end_y", mcast_dest_noc_end.y),
        ("mcast_num_cores", mcast_num_cores),
        ("mcast_data_sender_semaphore", mcast_sender_semaphore_id),
        ("mcast_data_receiver_semaphore", mcast_receiver_semaphore_id),
        ("mcast_data_size_bytes", mcast_data_size_bytes),
        ("mcast_src_cb", src_cb),
        ("mcast_src_num_pages", k_num_tiles),
        ("mcast_dst_cb", dst_cb),
        ("mcast_is_part_of_receiver_grid", is_sender_in_mcast_grid),
    ]

    trisc_named_compile_time_args = [
        ("mcast_dst_cb", dst_cb),
        ("matmul_in1", in1_cb),
        ("matmul_out", out_cb),
        ("matmul_k_num_tiles", k_num_tiles),
        ("matmul_out_w_per_core", out_w_per_core),
    ]

    # =========================================================================
    # Kernel descriptor - the key part!
    # =========================================================================

    # All cores = sender + matmul + phantom
    all_cores = matmul_core_grid.merge(sender_core_grid)
    if phantom_core_grid is not None:
        all_cores = all_cores.merge(phantom_core_grid)

    # Mcast grid cores = matmul + phantom (all cores that receive mcast)
    mcast_grid_cores = matmul_core_grid
    if phantom_core_grid is not None:
        mcast_grid_cores = mcast_grid_cores.merge(phantom_core_grid)

    unified_kernel = UnifiedKernelDescriptor(
        kernel_source="models/demos/deepseek_v3_b1/micro_ops/mcast_matmul/kernels/mcast_matmul_kernel.cpp",
        core_ranges=all_cores,
        ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
        brisc_named_compile_time_args=brisc_named_compile_time_args,
        trisc_named_compile_time_args=trisc_named_compile_time_args,
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
        unified_compile_time_core_descriptors=[
            # is_sender_core: only the mcast source core
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_sender_core",
                core_range=mcast_core,
                value=1,
                other_value=0,
            ),
            # is_matmul_core: only cores that do computation (NOT phantom cores)
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_matmul_core",
                core_range=matmul_core_grid,
                value=1,
                other_value=0,
            ),
            # is_mcast_grid_core: ALL cores in mcast bounding box (matmul + phantom)
            # These cores receive mcast data and MUST reset the semaphore
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_mcast_grid_core",
                core_range=mcast_grid_cores,
                value=1,
                other_value=0,
            ),
        ],
    )

    # Create program descriptor
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=unified_kernel.get_kernel_descriptors(),
        cbs=[src_cb_descriptor, src_cb_placeholder_descriptor, dst_cb_descriptor, in1_cb_descriptor, out_cb_descriptor],
        semaphores=[sender_semaphore_descriptor, receiver_semaphore_descriptor],
    )

    # Execute
    logger.info("  Running mcast matmul with phantom cores...")
    io_tensors = [ttnn_input, ttnn_weights, ttnn_output]
    ttnn_result = ttnn.generic_op(io_tensors, program_descriptor)

    # Verify
    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == (M, N), f"Expected shape ({M}, {N}), got {output_torch.shape}"

    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(f"  {pcc_message}")

    return passing, pcc_message


@pytest.mark.parametrize(
    "mcast_grid_x, mcast_grid_y, matmul_grid_x, matmul_grid_y, description",
    [
        # Mcast to 4x4=16 cores, compute on 2x4=8 cores (8 phantom cores)
        (4, 4, 2, 4, "4x4 mcast, 2x4 matmul"),
        # Mcast to 4x2=8 cores, compute on 2x2=4 cores (4 phantom cores)
        (4, 2, 2, 2, "4x2 mcast, 2x2 matmul"),
        # Mcast to 6x3=18 cores, compute on 3x3=9 cores (9 phantom cores)
        (6, 3, 3, 3, "6x3 mcast, 3x3 matmul"),
    ],
)
def test_mcast_matmul_phantom_cores(device, mcast_grid_x, mcast_grid_y, matmul_grid_x, matmul_grid_y, description):
    """
    Test mcast matmul where mcast grid > matmul grid.

    This tests the "phantom core" pattern where:
    - Mcast sends to a larger rectangular grid
    - Only a subset of cores have weights and compute
    - Phantom cores receive mcast data but just reset semaphore

    The matmul grid starts at (0,0) and is a subset of the mcast grid.
    """
    # Check device grid size
    device_grid = device.compute_with_storage_grid_size()
    if mcast_grid_x > device_grid.x or mcast_grid_y > device_grid.y:
        pytest.skip(
            f"Device grid {device_grid.x}x{device_grid.y} too small for mcast grid {mcast_grid_x}x{mcast_grid_y}"
        )

    # Mcast source core outside the mcast grid
    mcast_core = ttnn.CoreCoord(min(mcast_grid_x + 2, device_grid.x - 1), min(mcast_grid_y + 2, device_grid.y - 1))

    # Matrix dimensions - must divide evenly by matmul cores
    matmul_num_cores = matmul_grid_x * matmul_grid_y
    M = 1
    K = 256
    N = 32 * matmul_num_cores  # 32 columns per matmul core

    # Grids
    mcast_grid_start = ttnn.CoreCoord(0, 0)
    mcast_grid_end = ttnn.CoreCoord(mcast_grid_x - 1, mcast_grid_y - 1)
    matmul_grid_start = ttnn.CoreCoord(0, 0)
    matmul_grid_end = ttnn.CoreCoord(matmul_grid_x - 1, matmul_grid_y - 1)

    logger.info(f"Testing mcast matmul with phantom cores: {description}")
    passing, pcc_message = run_mcast_matmul_with_phantom_cores_test(
        device,
        M,
        K,
        N,
        mcast_grid_start,
        mcast_grid_end,
        matmul_grid_start,
        matmul_grid_end,
        mcast_core,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        pcc_threshold=0.98,
    )

    assert passing, pcc_message
    logger.info(f"✓ {description} passed")


def test_mcast_matmul_phantom_cores_offset_grid(device):
    """
    Test phantom cores where matmul grid is NOT at origin.

    Mcast grid: (0,0) to (3,3) = 16 cores
    Matmul grid: (1,1) to (2,2) = 4 cores (center of mcast grid)
    Phantom cores: 12 cores around the edges
    """
    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 6 or device_grid.y < 6:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small")

    mcast_core = ttnn.CoreCoord(5, 5)

    M = 1
    K = 256
    N = 128  # 4 matmul cores × 32 cols/core

    mcast_grid_start = ttnn.CoreCoord(0, 0)
    mcast_grid_end = ttnn.CoreCoord(3, 3)
    matmul_grid_start = ttnn.CoreCoord(1, 1)
    matmul_grid_end = ttnn.CoreCoord(2, 2)

    logger.info("Testing mcast matmul with offset matmul grid (phantom cores on edges)")
    passing, pcc_message = run_mcast_matmul_with_phantom_cores_test(
        device,
        M,
        K,
        N,
        mcast_grid_start,
        mcast_grid_end,
        matmul_grid_start,
        matmul_grid_end,
        mcast_core,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        pcc_threshold=0.98,
    )

    assert passing, pcc_message
    logger.info("✓ Offset matmul grid test passed")
