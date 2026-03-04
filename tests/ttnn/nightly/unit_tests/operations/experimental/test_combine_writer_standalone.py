# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Standalone kernel-level test for the combine-write data path of moe_gpt_fused.

Isolates ONLY the combine-write from matmul_dm1.cpp (lines 143-208).
No ring A2A, no matmul, no tilize. Runs on a single Wormhole device.

Tests that 12 sender cores correctly scatter untilized ROW_MAJOR data
from cb_c2s_out (c_14) to 12 combine cores in a 4x3 block-sharded layout.

Dimensions (from moe_gpt_fused_ring_common.h):
  K = 2880 (hidden_size) -> 90 tiles
  E = 4 experts per device
  tokens_per_chunk = 32
  combine grid: 4 height x 3 width = 12 cores
  combine shard: [32, 960] per core (32 tokens x 30 tiles x 32 tile_width)
  source_width_tiles = 8 (max tiles per ring core)
  W2_TILES_PER_CORE_A = [8,8,7,7,8,8,7,7,8,8,7,7]
"""

import pytest
import torch
import ttnn
from loguru import logger


# Constants from moe_gpt_fused_ring_common.h
NUM_CORES = 12
NUM_EXPERTS = 4
TOKENS_PER_CHUNK = 32
HIDDEN_SIZE = 2880
K_TILES = HIDDEN_SIZE // 32  # 90
COMBINE_WIDTH_SHARD_DIM = 3
COMBINE_HEIGHT_SHARD_DIM = 4
COMBINE_SHARD_WIDTH_TILES = K_TILES // COMBINE_WIDTH_SHARD_DIM  # 30
SOURCE_WIDTH_TILES = 8
TILE_WIDTH = 32
TILE_WIDTH_SIZE_BYTES = TILE_WIDTH * 2  # bfloat16 = 2 bytes

# Per-core W2 tile count (from W2_TILES_PER_CORE_A)
W2_TILES_PER_CORE = [8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7]

# Prefix-sum: combine width offset per ring core
COMBINE_W_OFFSET = []
_s = 0
for t in W2_TILES_PER_CORE:
    COMBINE_W_OFFSET.append(_s)
    _s += t

RING_CORES_PER_COMBINE_COL = NUM_CORES // COMBINE_WIDTH_SHARD_DIM  # 4


def serialize_physical_core_coords(logical_cores, device):
    """
    Replicate C++ serialize_physical_core_coords:
    Returns "{x0, y0, x1, y1, ...}" for use as OUTPUT_SHARD_CORE_MAP define.
    """
    coords = []
    for c in logical_cores:
        pc = device.worker_core_from_logical_core(c)
        coords.append(str(pc.x))
        coords.append(str(pc.y))
    return "{" + ", ".join(coords) + "}"


TOKENS_PER_HEIGHT_SHARD = TOKENS_PER_CHUNK // COMBINE_HEIGHT_SHARD_DIM  # 8


def build_golden(input_data):
    """
    Build golden reference output matching the DeepSeek moe_compute combine-write layout.

    Tokens are distributed round-robin across height shards. Within each shard,
    expert blocks are stacked: [E0 block][E1 block][E2 block][E3 block].

    input_data: dict mapping ring_core_id -> torch tensor of shape
                [NUM_EXPERTS * TOKENS_PER_CHUNK, SOURCE_WIDTH_TILES * TILE_WIDTH]
                (i.e. [128, 256])

    Returns: torch tensor of shape [NUM_EXPERTS * TOKENS_PER_CHUNK, HIDDEN_SIZE]
             i.e. [128, 2880]
    """
    # Output shard layout per shard (32 rows):
    #   Expert 0: rows 0..7   (TOKENS_PER_HEIGHT_SHARD rows)
    #   Expert 1: rows 8..15
    #   Expert 2: rows 16..23
    #   Expert 3: rows 24..31
    # 4 shards × 32 rows = 128 total rows
    output = torch.zeros(NUM_EXPERTS * TOKENS_PER_CHUNK, HIDDEN_SIZE, dtype=torch.bfloat16)

    shard_rows_per_shard = NUM_EXPERTS * TOKENS_PER_HEIGHT_SHARD  # 4 * 8 = 32

    for ring_core_id in range(NUM_CORES):
        w_offset = COMBINE_W_OFFSET[ring_core_id]
        tiles = W2_TILES_PER_CORE[ring_core_id]
        src = input_data[ring_core_id]

        for e in range(NUM_EXPERTS):
            max_tokens_per_height_shard = (TOKENS_PER_CHUNK + COMBINE_HEIGHT_SHARD_DIM - 1) // COMBINE_HEIGHT_SHARD_DIM
            expert_offset_rows = e * max_tokens_per_height_shard  # e * 8

            for bt in range(TOKENS_PER_CHUNK):
                # Which height shard does this token go to?
                dest_shard = bt // max_tokens_per_height_shard
                shard_row = bt % max_tokens_per_height_shard

                # Destination row in the flattened output tensor
                dst_row = dest_shard * shard_rows_per_shard + expert_offset_rows + shard_row

                src_row = e * TOKENS_PER_CHUNK + bt
                src_start = 0
                src_end = tiles * TILE_WIDTH
                dst_start = w_offset * TILE_WIDTH
                dst_end = (w_offset + tiles) * TILE_WIDTH
                output[dst_row, dst_start:dst_end] = src[src_row, src_start:src_end]

    return output


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW},
            id="dispatch_row",
        )
    ],
    indirect=True,
)
def test_combine_writer_standalone(device):
    """
    Test the combine-write data path in isolation using generic_op.

    Sets up:
    - 12 sender cores running combine_writer_test_kernel.cpp
    - 12 combine cores running combine_dm1.cpp
    - Validates output matches golden reference (exact match, no compute)
    """
    logger.info("Setting up combine writer standalone test")

    # =========================================================================
    # Core assignments (same as moe_gpt_fused_program_factory.cpp)
    # =========================================================================

    # Sender (matmul) cores: 12 cores aligned to DRAM banks
    matmul_logical_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(0)
    assert len(matmul_logical_cores) == NUM_CORES, f"Expected {NUM_CORES} matmul cores, got {len(matmul_logical_cores)}"

    # Ring ordering: sort by physical (y desc, x desc) - same as production
    core2bank = {}
    for bank_id, core in enumerate(matmul_logical_cores):
        core2bank[(core.x, core.y)] = bank_id

    bank_ids_sorted = list(range(NUM_CORES))
    bank_ids_sorted.sort(
        key=lambda b: (
            -device.worker_core_from_logical_core(matmul_logical_cores[b]).y,
            -device.worker_core_from_logical_core(matmul_logical_cores[b]).x,
        )
    )
    # bank_ids_sorted[ring_pos] = bank_id
    # We need: ring_pos -> bank_id mapping and inverse
    ring2bank = {ring_pos: bank_id for ring_pos, bank_id in enumerate(bank_ids_sorted)}
    bank2ring = {bank_id: ring_pos for ring_pos, bank_id in ring2bank.items()}

    matmul_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in matmul_logical_cores]
    )

    # Combine cores: CoreRange({1,0},{3,3}) = 12 cores, sorted by (y asc, x asc)
    combine_core_range = ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 3))
    combine_core_range_set = ttnn.CoreRangeSet([combine_core_range])

    # Get sorted combine cores (y asc, x asc) - same as production
    combine_logical_cores = []
    for y in range(0, 4):
        for x in range(1, 4):
            combine_logical_cores.append(ttnn.CoreCoord(x, y))
    assert len(combine_logical_cores) == 12

    # Verify no overlap between sender and combine core sets
    sender_set = {(c.x, c.y) for c in matmul_logical_cores}
    combine_set = {(c.x, c.y) for c in combine_logical_cores}
    assert sender_set.isdisjoint(combine_set), "Sender and combine cores must not overlap"

    logger.info(f"Sender cores: {[(c.x, c.y) for c in matmul_logical_cores]}")
    logger.info(f"Combine cores: {[(c.x, c.y) for c in combine_logical_cores]}")

    # =========================================================================
    # Serialize combine core physical coords for OUTPUT_SHARD_CORE_MAP define
    # =========================================================================
    output_shard_core_map_str = serialize_physical_core_coords(combine_logical_cores, device)
    logger.info(f"OUTPUT_SHARD_CORE_MAP = {output_shard_core_map_str}")

    # =========================================================================
    # Create input data on sender cores
    # HEIGHT_SHARDED L1 tensor, shard = [128, 256] per core
    # 128 = NUM_EXPERTS * TOKENS_PER_CHUNK, 256 = SOURCE_WIDTH_TILES * TILE_WIDTH
    # =========================================================================
    shard_height = NUM_EXPERTS * TOKENS_PER_CHUNK  # 128
    shard_width = SOURCE_WIDTH_TILES * TILE_WIDTH  # 256

    # Create deterministic input: value = ring_core_id * 1000 + expert * 100 + token + tile * 0.01
    # We need to lay out data in the order the cores appear in the shard spec
    input_per_ring_core = {}
    all_shards = []

    for bank_id in range(NUM_CORES):
        ring_pos = bank2ring[bank_id]
        shard_data = torch.zeros(shard_height, shard_width, dtype=torch.bfloat16)

        for e in range(NUM_EXPERTS):
            for bt in range(TOKENS_PER_CHUNK):
                row = e * TOKENS_PER_CHUNK + bt
                for t in range(SOURCE_WIDTH_TILES):
                    col_start = t * TILE_WIDTH
                    col_end = (t + 1) * TILE_WIDTH
                    val = ring_pos * 1000 + e * 100 + bt + t * 0.01
                    shard_data[row, col_start:col_end] = val

        input_per_ring_core[ring_pos] = shard_data
        all_shards.append(shard_data)

    # Stack shards into full tensor: [NUM_CORES * shard_height, shard_width]
    full_input = torch.cat(all_shards, dim=0)  # [1536, 256]

    input_shard_spec = ttnn.ShardSpec(
        matmul_core_range_set,
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # Create on DRAM first, then reshard to L1
    tt_input_dram = ttnn.from_torch(
        full_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_input = ttnn.to_memory_config(tt_input_dram, input_mem_config)
    ttnn.deallocate(tt_input_dram)

    logger.info(f"Input tensor shape: {tt_input.shape}, memory: {tt_input.memory_config()}")

    # =========================================================================
    # Create output tensor on combine cores
    # BLOCK_SHARDED L1, shape [128, 2880], shard [32, 960]
    # =========================================================================
    output_shard_height = TOKENS_PER_CHUNK  # 32
    output_shard_width = COMBINE_SHARD_WIDTH_TILES * TILE_WIDTH  # 960
    output_shape = [NUM_EXPERTS * TOKENS_PER_CHUNK, HIDDEN_SIZE]  # [128, 2880]

    output_shard_spec = ttnn.ShardSpec(
        combine_core_range_set,
        [output_shard_height, output_shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Allocate output tensor initialized to zeros
    tt_output = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=output_mem_config,
    )

    output_base_l1_addr = tt_output.buffer_address()
    logger.info(f"Output tensor shape: {tt_output.shape}, memory: {tt_output.memory_config()}")
    logger.info(f"Output buffer L1 addr: {output_base_l1_addr}")

    # =========================================================================
    # Circular Buffers
    # =========================================================================

    # c_14 on sender cores: backed by input tensor (SOURCE_WIDTH_TILES * TILE_WIDTH * 2 = 512 bytes per page)
    c14_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(14, tt_input)

    # c_0 on combine cores: backed by output tensor
    c0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(0, tt_output)

    # =========================================================================
    # Semaphore: combine_semaphore on combine cores
    # =========================================================================
    combine_semaphore_id = 0
    combine_semaphore = ttnn.SemaphoreDescriptor(
        id=combine_semaphore_id,
        core_ranges=combine_core_range_set,
        initial_value=0,
    )

    # =========================================================================
    # Named compile-time args for the test kernel (same as production dm1)
    # =========================================================================
    named_compile_args = [
        ("num_experts", NUM_EXPERTS),
        ("height_shard_dim", COMBINE_HEIGHT_SHARD_DIM),
        ("width_shard_dim", COMBINE_WIDTH_SHARD_DIM),
        ("combine_shard_width_tiles", COMBINE_SHARD_WIDTH_TILES),
        ("tile_width", TILE_WIDTH),
        ("tile_width_size_bytes", TILE_WIDTH_SIZE_BYTES),
        ("num_tokens_total", TOKENS_PER_CHUNK),
    ]

    # =========================================================================
    # Build per-core runtime args for sender kernel
    # =========================================================================
    sender_rt_args = ttnn.RuntimeArgs()

    for bank_id in range(NUM_CORES):
        ring_pos = bank2ring[bank_id]
        core = matmul_logical_cores[bank_id]
        vchannel = bank_id & 0x3

        sender_rt_args[core.x][core.y] = [
            ring_pos,  # [0] ring_core_id
            combine_semaphore_id,  # [1] combine_semaphore_id
            output_base_l1_addr,  # [2] output_base_l1_addr
            vchannel,  # [3] vchannel
        ]

    # =========================================================================
    # Define kernels
    # =========================================================================
    kernel_path = "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt_fused/device/kernels"

    # Sender kernel: our test kernel on matmul cores (RISCV_0 / NOC_1)
    sender_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{kernel_path}/combine_writer_test_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=matmul_core_range_set,
        named_compile_time_args=named_compile_args,
        defines=[("OUTPUT_SHARD_CORE_MAP", output_shard_core_map_str)],
        runtime_args=sender_rt_args,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.NOC_1,
        ),
    )

    # Receiver kernel: real combine_dm1.cpp on combine cores (RISCV_0 / NOC_1)
    combine_rt_args = ttnn.RuntimeArgs()
    for c in combine_logical_cores:
        combine_rt_args[c.x][c.y] = [combine_semaphore_id]

    receiver_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{kernel_path}/combine_dm1.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=combine_core_range_set,
        runtime_args=combine_rt_args,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.NOC_1,
        ),
    )

    # =========================================================================
    # Build and run program
    # =========================================================================
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[sender_kernel, receiver_kernel],
        cbs=[c14_cb_descriptor, c0_cb_descriptor],
        semaphores=[combine_semaphore],
    )

    logger.info("Running combine writer test kernel...")
    ttnn.generic_op([tt_input, tt_output], program_descriptor)
    logger.info("Kernel execution complete")

    # =========================================================================
    # Validate output
    # =========================================================================
    output_torch = ttnn.to_torch(tt_output)
    logger.info(f"Output shape: {output_torch.shape}")

    # Build golden reference
    golden = build_golden(input_per_ring_core)

    # Exact match since no compute, just data movement
    if torch.equal(output_torch, golden):
        logger.info("PASS: Output exactly matches golden reference")
    else:
        # Find mismatches for debugging
        mismatch_mask = output_torch != golden
        num_mismatches = mismatch_mask.sum().item()
        total_elements = output_torch.numel()
        logger.error(f"FAIL: {num_mismatches}/{total_elements} elements mismatch")

        # Show first few mismatches
        mismatch_indices = torch.nonzero(mismatch_mask)
        for i in range(min(10, len(mismatch_indices))):
            idx = tuple(mismatch_indices[i].tolist())
            logger.error(f"  [{idx}] got={output_torch[idx].item():.4f}, expected={golden[idx].item():.4f}")

    assert torch.equal(output_torch, golden), "Output does not match golden reference"
    logger.info("Combine writer standalone test PASSED")
