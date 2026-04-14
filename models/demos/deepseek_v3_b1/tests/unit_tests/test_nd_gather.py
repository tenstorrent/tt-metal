# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ND Gather Reproduction Test

Replicates the gather portion of GatherReduce after matmul5 in the attention block:
  - 96 sender cores (12x8 grid) each write 64B (1x32 bf16 tile) to a single receiver core
  - Half-based offset: cores 0-47 write to half0, cores 48-95 write to half1
  - NOC API: noc_async_write_one_packet + noc_semaphore_inc + noc_async_posted_writes_flushed
  - No TRISC reduction

Run multiple iterations to detect non-deterministic (ND) PCC corruption.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

# =============================================================================
# Constants matching GatherReduce3 in the attention block
# =============================================================================

SENDER_GRID_START = (0, 0)
SENDER_GRID_END = (11, 7)
NUM_SENDER_COLS = SENDER_GRID_END[0] - SENDER_GRID_START[0] + 1  # 12
NUM_SENDER_ROWS = SENDER_GRID_END[1] - SENDER_GRID_START[1] + 1  # 8
NUM_SENDERS = NUM_SENDER_COLS * NUM_SENDER_ROWS  # 96
HALF_NUM_CORES = NUM_SENDERS // 2  # 48
WIDTH_PER_CORE = 32  # 1x32 tile
TILE_1X32 = ttnn.Tile([1, 32])
DATA_SIZE_BYTES = 64  # 1x32 bf16 tile = 32 * 2 bytes
HALF_SIZE_BYTES = HALF_NUM_CORES * DATA_SIZE_BYTES  # 48 * 64 = 3072
TOTAL_WIDTH = NUM_SENDERS * WIDTH_PER_CORE  # 96 * 32 = 3072 elements

# Gather+Reduce constants (16x32 tiles for TRISC reduction)
NUM_REDUCE_TILES = 3  # 3 tiles of 16x32 per half
TILE_16X32 = ttnn.Tile([16, 32])
TILE_16X32_SIZE = 1024  # 16 * 32 * 2 bytes (bf16)
SCRATCH_SIZE_BYTES = 2 * NUM_REDUCE_TILES * TILE_16X32_SIZE  # 6144 bytes

RECEIVER_CORE = ttnn.CoreCoord(12, 9)  # rmsnorm core in attention block

GATHER_KERNEL_PATH = "models/demos/deepseek_v3_b1/tests/unit_tests/kernels/test_nd_gather_kernel.cpp"
GATHER_REDUCE_KERNEL_PATH = "models/demos/deepseek_v3_b1/tests/unit_tests/kernels/test_nd_gather_reduce_kernel.cpp"

SEMAPHORE_ID = 0


# =============================================================================
# Helpers
# =============================================================================


def create_sharded_input(device, torch_input, sender_grid, shard_shape):
    """Create WIDTH_SHARDED input tensor on the sender grid."""
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({sender_grid}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=TILE_1X32,
    )


def create_output_tensor(device, output_shape, receiver_core):
    """Create HEIGHT_SHARDED output tensor on the receiver core."""
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(receiver_core, receiver_core)}),
        output_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    return ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=TILE_1X32,
    )


def build_program(device, input_tensor, output_tensor, sender_grid, receiver_core):
    """Build the gather program descriptor (identical NOC pattern to GatherReduce)."""
    receiver_noc_core = device.worker_core_from_logical_core(receiver_core)
    receiver_data_addr = output_tensor.buffer_address()

    sender_core_range = ttnn.CoreRangeSet({sender_grid})
    receiver_core_range = ttnn.CoreRangeSet({ttnn.CoreRange(receiver_core, receiver_core)})
    all_cores = sender_core_range.merge(receiver_core_range)

    # --- Semaphore ---
    semaphore = ttnn.SemaphoreDescriptor(
        id=SEMAPHORE_ID,
        core_ranges=all_cores,
        initial_value=0,
    )

    # --- Sender kernel (NCRISC) on 12x8 grid ---
    sender_named_args = [
        ("is_sender_core", 1),
        ("is_receiver_core", 0),
        ("src_cb", 0),
        ("src_num_pages", 1),
        ("dest_noc_x", receiver_noc_core.x),
        ("dest_noc_y", receiver_noc_core.y),
        ("data_size_bytes", DATA_SIZE_BYTES),
        ("receiver_semaphore_id", SEMAPHORE_ID),
        ("grid_start_x", sender_grid.start.x),
        ("grid_start_y", sender_grid.start.y),
        ("grid_end_x", sender_grid.end.x),
        ("grid_end_y", sender_grid.end.y),
        ("half_num_cores", HALF_NUM_CORES),
        ("half_size_bytes", HALF_SIZE_BYTES),
    ]

    sender_kernel = ttnn.KernelDescriptor(
        kernel_source=GATHER_KERNEL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_core_range,
        named_compile_time_args=sender_named_args,
        common_runtime_args=[receiver_data_addr],
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1,
            noc=ttnn.NOC.NOC_0,
        ),
    )

    # --- Receiver kernel (BRISC) on receiver core ---
    receiver_named_args = [
        ("is_sender_core", 0),
        ("is_receiver_core", 1),
        ("noc0_num_senders", NUM_SENDERS),
        ("noc0_receiver_semaphore_id", SEMAPHORE_ID),
    ]

    receiver_kernel = ttnn.KernelDescriptor(
        kernel_source=GATHER_KERNEL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=receiver_core_range,
        named_compile_time_args=receiver_named_args,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.NOC_1,
        ),
    )

    # --- Compute kernels (no-op) ---
    sender_compute = ttnn.KernelDescriptor(
        kernel_source=GATHER_KERNEL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_core_range,
        named_compile_time_args=[("is_sender_core", 1), ("is_receiver_core", 0)],
        config=ttnn.ComputeConfigDescriptor(),
    )

    receiver_compute = ttnn.KernelDescriptor(
        kernel_source=GATHER_KERNEL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=receiver_core_range,
        named_compile_time_args=[("is_sender_core", 0), ("is_receiver_core", 1)],
        config=ttnn.ComputeConfigDescriptor(),
    )

    # --- CBs ---
    src_cb = ttnn.cb_descriptor_from_sharded_tensor(0, input_tensor)

    return ttnn.ProgramDescriptor(
        kernels=[sender_kernel, receiver_kernel, sender_compute, receiver_compute],
        cbs=[src_cb],
        semaphores=[semaphore],
    )


# =============================================================================
# Test
# =============================================================================


@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.parametrize("num_iterations", [20])
def test_nd_gather(device, check_requires_grid_size, num_iterations):
    """
    Run the gather pattern from GatherReduce (96 senders -> 1 receiver, 64B each)
    multiple times and check for ND PCC corruption.
    """
    sender_grid = ttnn.CoreRange(
        ttnn.CoreCoord(*SENDER_GRID_START),
        ttnn.CoreCoord(*SENDER_GRID_END),
    )

    shard_shape = (1, WIDTH_PER_CORE)
    output_shape = (1, TOTAL_WIDTH)

    logger.info(f"Testing ND gather: {NUM_SENDERS} senders (12x8) -> receiver ({RECEIVER_CORE.x},{RECEIVER_CORE.y})")
    logger.info(f"  data_size_bytes={DATA_SIZE_BYTES}, half_num_cores={HALF_NUM_CORES}, iterations={num_iterations}")

    all_pass = True
    pcc_values = []

    for i in range(num_iterations):
        torch.manual_seed(i)
        torch_input = torch.randn(output_shape, dtype=torch.bfloat16)

        ttnn_input = create_sharded_input(device, torch_input, sender_grid, shard_shape)
        ttnn_output = create_output_tensor(device, output_shape, RECEIVER_CORE)

        program = build_program(device, ttnn_input, ttnn_output, sender_grid, RECEIVER_CORE)
        result = ttnn.generic_op([ttnn_input, ttnn_output], program)

        result_torch = ttnn.to_torch(result)
        passing, pcc_val = comp_pcc(torch_input, result_torch, 0.9999)
        pcc_values.append(float(pcc_val))

        status = "PASS" if passing else "FAIL"
        logger.info(f"  [{i+1:>2}/{num_iterations}] {status} PCC={pcc_val:.6f}")

        if not passing:
            all_pass = False

        ttnn_input.deallocate()
        ttnn_output.deallocate()

    # Summary
    logger.info(f"Results: {sum(1 for p in pcc_values if p >= 0.9999)}/{num_iterations} passed (PCC >= 0.9999)")
    logger.info(f"  min PCC = {min(pcc_values):.6f}, max PCC = {max(pcc_values):.6f}")

    assert all_pass, f"ND gather failed on {sum(1 for p in pcc_values if p < 0.9999)}/{num_iterations} iterations"


# =============================================================================
# Gather + Reduce Test
# =============================================================================


def create_scratch_anchor(device, full_grid_set, num_grid_cores):
    """Create HEIGHT_SHARDED anchor tensor on full grid for scratch CB address allocation."""
    scratch_shard_u32 = SCRATCH_SIZE_BYTES // 4  # 6144 / 4 = 1536 uint32 elements
    shard_spec = ttnn.ShardSpec(full_grid_set, (1, scratch_shard_u32), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.from_torch(
        torch.zeros([num_grid_cores, scratch_shard_u32], dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_config,
    )


def create_reduce_output_tensor(device, receiver_core):
    """Create HEIGHT_SHARDED output tensor on receiver core for reduce result (3 tiles of 16x32)."""
    receiver_core_range = ttnn.CoreRangeSet({ttnn.CoreRange(receiver_core, receiver_core)})
    output_shape = (NUM_REDUCE_TILES * 16, 32)  # (48, 32) = 3 tiles of 16x32
    shard_spec = ttnn.ShardSpec(receiver_core_range, output_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=TILE_16X32,
    )


def build_gather_reduce_program(device, input_tensor, anchor_tensor, output_tensor, sender_grid, receiver_core):
    """Build the gather+reduce program (identical NOC pattern + TRISC reduction as GatherReduce)."""
    receiver_noc_core = device.worker_core_from_logical_core(receiver_core)

    sender_core_range = ttnn.CoreRangeSet({sender_grid})
    receiver_core_range = ttnn.CoreRangeSet({ttnn.CoreRange(receiver_core, receiver_core)})
    all_cores = sender_core_range.merge(receiver_core_range)

    # Bounding box grid for scratch CB (must cover all sender + receiver cores)
    full_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 9))
    full_grid_set = ttnn.CoreRangeSet({full_grid})

    # CB indices
    src_cb_id = 0
    scratch_cb_id = 1
    out_cb_id = 2

    # --- Semaphore ---
    semaphore = ttnn.SemaphoreDescriptor(
        id=SEMAPHORE_ID,
        core_ranges=all_cores,
        initial_value=0,
    )

    # --- Sender kernel (NCRISC) on 12x8 grid ---
    sender_named_args = [
        ("is_sender_core", 1),
        ("is_receiver_core", 0),
        ("src_cb", src_cb_id),
        ("src_num_pages", 1),
        ("dest_noc_x", receiver_noc_core.x),
        ("dest_noc_y", receiver_noc_core.y),
        ("data_size_bytes", DATA_SIZE_BYTES),
        ("receiver_semaphore_id", SEMAPHORE_ID),
        ("grid_start_x", sender_grid.start.x),
        ("grid_start_y", sender_grid.start.y),
        ("grid_end_x", sender_grid.end.x),
        ("grid_end_y", sender_grid.end.y),
        ("half_num_cores", HALF_NUM_CORES),
        ("half_size_bytes", HALF_SIZE_BYTES),
        ("scratch_cb", scratch_cb_id),
    ]

    sender_kernel = ttnn.KernelDescriptor(
        kernel_source=GATHER_REDUCE_KERNEL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_core_range,
        named_compile_time_args=sender_named_args,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1,
            noc=ttnn.NOC.NOC_0,
        ),
    )

    # --- Receiver kernel (BRISC) on receiver core ---
    receiver_named_args = [
        ("is_sender_core", 0),
        ("is_receiver_core", 1),
        ("noc0_num_senders", NUM_SENDERS),
        ("noc0_receiver_semaphore_id", SEMAPHORE_ID),
        ("scratch_cb", scratch_cb_id),
        ("num_tiles", NUM_REDUCE_TILES),
    ]

    receiver_kernel = ttnn.KernelDescriptor(
        kernel_source=GATHER_REDUCE_KERNEL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=receiver_core_range,
        named_compile_time_args=receiver_named_args,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.NOC_1,
        ),
    )

    # --- Compute kernels ---
    # Sender compute (no-op, but must pass TRISC args for constexpr resolution)
    sender_compute = ttnn.KernelDescriptor(
        kernel_source=GATHER_REDUCE_KERNEL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_core_range,
        named_compile_time_args=[
            ("is_sender_core", 1),
            ("is_receiver_core", 0),
            ("scratch_cb", scratch_cb_id),
            ("out_cb", out_cb_id),
            ("num_tiles", NUM_REDUCE_TILES),
        ],
        config=ttnn.ComputeConfigDescriptor(),
    )

    # Receiver compute (TRISC add_half_tiles)
    receiver_compute_args = [
        ("is_sender_core", 0),
        ("is_receiver_core", 1),
        ("scratch_cb", scratch_cb_id),
        ("out_cb", out_cb_id),
        ("num_tiles", NUM_REDUCE_TILES),
    ]

    receiver_compute = ttnn.KernelDescriptor(
        kernel_source=GATHER_REDUCE_KERNEL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=receiver_core_range,
        named_compile_time_args=receiver_compute_args,
        config=ttnn.ComputeConfigDescriptor(),
    )

    # --- CBs ---
    # CB0: src input (from sharded input tensor, on sender cores only)
    src_cb = ttnn.cb_descriptor_from_sharded_tensor(src_cb_id, input_tensor)

    # CB1: scratch (from anchor tensor, on ALL cores so get_write_ptr returns same addr)
    scratch_cb = ttnn.cb_descriptor_from_sharded_tensor(
        scratch_cb_id,
        anchor_tensor,
        total_size=SCRATCH_SIZE_BYTES,
        core_ranges=full_grid_set,
    )
    scratch_cb.format_descriptors = [
        ttnn.CBFormatDescriptor(
            buffer_index=scratch_cb_id,
            data_format=ttnn.bfloat16,
            page_size=TILE_16X32_SIZE,
            tile=ttnn.TileDescriptor(TILE_16X32),
        )
    ]

    # CB2: output (from output tensor, on receiver core only)
    out_cb = ttnn.cb_descriptor_from_sharded_tensor(out_cb_id, output_tensor)
    out_cb.format_descriptors = [
        ttnn.CBFormatDescriptor(
            buffer_index=out_cb_id,
            data_format=ttnn.bfloat16,
            page_size=TILE_16X32_SIZE,
            tile=ttnn.TileDescriptor(TILE_16X32),
        )
    ]

    return ttnn.ProgramDescriptor(
        kernels=[sender_kernel, receiver_kernel, sender_compute, receiver_compute],
        cbs=[src_cb, scratch_cb, out_cb],
        semaphores=[semaphore],
    )


@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.parametrize("num_iterations", [20])
def test_nd_gather_reduce(device, check_requires_grid_size, num_iterations):
    """
    Run the gather+reduce pattern from GatherReduce (96 senders -> 1 receiver, 64B each,
    then TRISC add_half_tiles reduction) multiple times and check for ND PCC corruption.

    Uses same input every iteration. If output varies across iterations, that's ND.
    """
    sender_grid = ttnn.CoreRange(
        ttnn.CoreCoord(*SENDER_GRID_START),
        ttnn.CoreCoord(*SENDER_GRID_END),
    )

    # Bounding box grid for anchor tensor
    full_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 9))
    full_grid_set = ttnn.CoreRangeSet({full_grid})
    num_grid_cores = 13 * 10  # 130

    shard_shape = (1, WIDTH_PER_CORE)

    logger.info(
        f"Testing ND gather+reduce: {NUM_SENDERS} senders (12x8) -> receiver ({RECEIVER_CORE.x},{RECEIVER_CORE.y})"
    )
    logger.info(f"  data_size_bytes={DATA_SIZE_BYTES}, half_num_cores={HALF_NUM_CORES}")
    logger.info(f"  reduce: {NUM_REDUCE_TILES} tiles of 16x32, scratch={SCRATCH_SIZE_BYTES}B")
    logger.info(f"  iterations={num_iterations} (same input each time, checking output consistency)")

    # Fixed input for all iterations
    torch.manual_seed(42)
    torch_input = torch.randn(1, TOTAL_WIDTH, dtype=torch.bfloat16)

    reference_output = None
    all_pass = True
    pcc_values = []

    for i in range(num_iterations):
        ttnn_input = create_sharded_input(device, torch_input, sender_grid, shard_shape)
        anchor = create_scratch_anchor(device, full_grid_set, num_grid_cores)
        ttnn_output = create_reduce_output_tensor(device, RECEIVER_CORE)

        program = build_gather_reduce_program(device, ttnn_input, anchor, ttnn_output, sender_grid, RECEIVER_CORE)
        result = ttnn.generic_op([ttnn_input, anchor, ttnn_output], program)

        result_torch = ttnn.to_torch(result)

        if reference_output is None:
            reference_output = result_torch.clone()
            # Sanity: output should not be all zeros
            is_nonzero = result_torch.abs().sum().item() > 0
            logger.info(f"  [{i+1:>2}/{num_iterations}] Reference run (non-zero={is_nonzero})")
            if not is_nonzero:
                logger.warning("  Reference output is all zeros!")
        else:
            passing, pcc_val = comp_pcc(reference_output, result_torch, 0.9999)
            pcc_values.append(float(pcc_val))
            status = "PASS" if passing else "FAIL"
            logger.info(f"  [{i+1:>2}/{num_iterations}] {status} PCC={pcc_val:.6f} (vs reference)")
            if not passing:
                all_pass = False

        ttnn_input.deallocate()
        anchor.deallocate()
        ttnn_output.deallocate()

    # Summary
    if pcc_values:
        num_pass = sum(1 for p in pcc_values if p >= 0.9999)
        logger.info(f"Results: {num_pass}/{len(pcc_values)} passed (PCC >= 0.9999 vs reference)")
        logger.info(f"  min PCC = {min(pcc_values):.6f}, max PCC = {max(pcc_values):.6f}")
    else:
        logger.info("Only 1 iteration run, no cross-iteration comparison")

    assert (
        all_pass
    ), f"ND gather+reduce failed: {sum(1 for p in pcc_values if p < 0.9999)}/{len(pcc_values)} iterations differ from reference"
