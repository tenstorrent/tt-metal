# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ND GatherReduce3 Reproduction Test

Replicates GatherReduce3 after matmul5 in the bliu/deepseek attention block:
  - 112 sender cores (o_proj grid: 12x8 + 8x2) each write 64B (1x32 bf16 tile)
  - UsePerCoreSenderIdx mode: per-core contiguous index 0..111
  - Half-based offset: cores 0-55 write to half0, cores 56-111 write to half1
  - NOC API: noc_async_write + noc_async_write_barrier + noc_semaphore_inc
  - TRISC reduces using 32x32 tiles: out[i] = half0[i] + half1[i]
  - Receiver core (11, 9) = CCL sender core

Run multiple iterations to detect non-deterministic (ND) PCC corruption.
"""

import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

# =============================================================================
# Constants matching GatherReduce3 on bliu/deepseek
# =============================================================================

# Non-rectangular sender grid: o_proj cores = 12x8 + 8x2 = 112 cores
SENDER_RANGE1_START = (0, 0)
SENDER_RANGE1_END = (11, 7)  # 12 cols x 8 rows = 96 cores
SENDER_RANGE2_START = (0, 8)
SENDER_RANGE2_END = (7, 9)  # 8 cols x 2 rows = 16 cores
NUM_SENDERS = 112
HALF_NUM_CORES = 56

RECEIVER_CORE = ttnn.CoreCoord(11, 9)  # CCL sender core in bliu/deepseek

WIDTH_PER_CORE = 32  # 1x32 tile per sender
TILE_1X32 = ttnn.Tile([1, 32])
DATA_SIZE_BYTES = 64  # 1x32 bf16 = 32 * 2
TOTAL_WIDTH = NUM_SENDERS * WIDTH_PER_CORE  # 3584

# Reduction uses 32x32 tiles: ceil(56 senders / 32 rows per tile) = 2 tiles per half
TILE_32X32 = ttnn.Tile([32, 32])
TILE_32X32_SIZE = 2048  # 32 * 32 * 2 bytes (bf16)
NUM_REDUCE_TILES = 2  # 2 tiles of 32x32 per half
HALF_SIZE_BYTES = NUM_REDUCE_TILES * TILE_32X32_SIZE  # 4096
SCRATCH_SIZE_BYTES = 2 * HALF_SIZE_BYTES  # 8192

GATHER_REDUCE_KERNEL_PATH = "models/demos/deepseek_v3_b1/tests/unit_tests/kernels/test_nd_gather_reduce_kernel.cpp"

SEMAPHORE_ID = 0


# =============================================================================
# Helpers
# =============================================================================


def get_sender_core_range_set():
    """Build the non-rectangular sender CoreRangeSet matching o_proj grid."""
    range1 = ttnn.CoreRange(ttnn.CoreCoord(*SENDER_RANGE1_START), ttnn.CoreCoord(*SENDER_RANGE1_END))
    range2 = ttnn.CoreRange(ttnn.CoreCoord(*SENDER_RANGE2_START), ttnn.CoreCoord(*SENDER_RANGE2_END))
    return ttnn.CoreRangeSet({range1, range2})


def create_sharded_input(device, torch_input, sender_core_range_set):
    """Create WIDTH_SHARDED input on the non-rectangular sender grid."""
    shard_shape = (1, WIDTH_PER_CORE)
    shard_spec = ttnn.ShardSpec(sender_core_range_set, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=TILE_1X32,
    )


def create_scratch_anchor(device, full_grid_set, num_grid_cores):
    """Create HEIGHT_SHARDED anchor tensor on full grid for scratch CB address allocation.
    Uses 32x32 tiles to match production GatherReduce3 scratch format."""
    # 4 tiles of 32x32 per shard = (128, 32)
    shard_h = 2 * NUM_REDUCE_TILES * 32  # 4 * 32 = 128
    shard_w = 32
    shard_spec = ttnn.ShardSpec(full_grid_set, (shard_h, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.from_torch(
        torch.zeros([num_grid_cores * shard_h, shard_w], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=TILE_32X32,
    )


def create_output_tensor(device, receiver_core):
    """Create HEIGHT_SHARDED output tensor on receiver core for reduce result (2 tiles of 32x32)."""
    receiver_core_range = ttnn.CoreRangeSet({ttnn.CoreRange(receiver_core, receiver_core)})
    output_shape = (NUM_REDUCE_TILES * 32, 32)  # (64, 32) = 2 tiles of 32x32
    shard_spec = ttnn.ShardSpec(receiver_core_range, output_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=TILE_32X32,
    )


def build_program(device, input_tensor, anchor_tensor, output_tensor, sender_core_range_set, receiver_core):
    """Build the GatherReduce3 program using UnifiedKernelDescriptor (matches bliu/deepseek production)."""
    from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
        PerCoreCompileTimeDescriptor,
        UnifiedCompileTimeCoreDescriptor,
        UnifiedKernelDescriptor,
    )

    receiver_noc_core = device.worker_core_from_logical_core(receiver_core)
    receiver_core_range = ttnn.CoreRangeSet({ttnn.CoreRange(receiver_core, receiver_core)})
    all_cores = sender_core_range_set.merge(receiver_core_range)

    # Bounding box grid for scratch CB (must cover all sender + receiver cores)
    full_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 9))
    full_grid_set = ttnn.CoreRangeSet({full_grid})

    src_cb_id = 0
    scratch_cb_id = 1
    out_cb_id = 2

    # --- Semaphore ---
    semaphore = ttnn.SemaphoreDescriptor(id=SEMAPHORE_ID, core_ranges=all_cores, initial_value=0)

    # --- Per-core sender_idx (UsePerCoreSenderIdx mode) ---
    sender_cores = ttnn.corerange_to_cores(sender_core_range_set, row_wise=True)
    sender_idx_core_values = [(core, idx) for idx, core in enumerate(sender_cores)]

    # --- Unified kernel ---
    unified_kernel = UnifiedKernelDescriptor(
        kernel_source=GATHER_REDUCE_KERNEL_PATH,
        core_ranges=all_cores,
        ncrisc_named_compile_time_args=[
            ("src_cb", src_cb_id),
            ("src_num_pages", 1),
            ("dest_noc_x", receiver_noc_core.x),
            ("dest_noc_y", receiver_noc_core.y),
            ("data_size_bytes", DATA_SIZE_BYTES),
            ("receiver_semaphore_id", SEMAPHORE_ID),
            ("half_num_cores", HALF_NUM_CORES),
            ("half_size_bytes", HALF_SIZE_BYTES),
            ("scratch_cb", scratch_cb_id),
        ],
        brisc_named_compile_time_args=[
            ("noc0_num_senders", NUM_SENDERS),
            ("noc0_receiver_semaphore_id", SEMAPHORE_ID),
            ("scratch_cb", scratch_cb_id),
            ("num_tiles", NUM_REDUCE_TILES),
        ],
        trisc_named_compile_time_args=[
            ("scratch_cb", scratch_cb_id),
            ("out_cb", out_cb_id),
            ("num_tiles", NUM_REDUCE_TILES),
        ],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
        unified_compile_time_core_descriptors=[
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_sender_core",
                core_range=sender_core_range_set,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_receiver_core",
                core_range=receiver_core_range,
                value=1,
                other_value=0,
            ),
        ],
        per_core_compile_time_descriptors=[
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="sender_idx",
                core_values=sender_idx_core_values,
                other_value=0,
            ),
        ],
    )

    kernel_result = unified_kernel.get_kernel_descriptors()

    # --- CBs ---
    # CB0: src input (from sharded input tensor, on sender cores)
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
            page_size=TILE_32X32_SIZE,
            tile=ttnn.TileDescriptor(TILE_32X32),
        )
    ]

    # CB2: output (from output tensor, on receiver core only)
    out_cb = ttnn.cb_descriptor_from_sharded_tensor(out_cb_id, output_tensor)
    out_cb.format_descriptors = [
        ttnn.CBFormatDescriptor(
            buffer_index=out_cb_id,
            data_format=ttnn.bfloat16,
            page_size=TILE_32X32_SIZE,
            tile=ttnn.TileDescriptor(TILE_32X32),
        )
    ]

    return ttnn.ProgramDescriptor(
        kernels=kernel_result.kernels,
        cbs=[src_cb, scratch_cb, out_cb],
        semaphores=[semaphore],
    )


# =============================================================================
# Test
# =============================================================================


@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.parametrize("num_iterations", [50])
def test_nd_gather_reduce(device, check_requires_grid_size, num_iterations):
    """
    Run the GatherReduce3 pattern (112 senders -> receiver (11,9), 64B each,
    then TRISC add_half_tiles reduction with 32x32 tiles) multiple times
    and check for ND PCC corruption.

    Matches bliu/deepseek production: non-rectangular o_proj grid,
    UsePerCoreSenderIdx, noc_async_write + barrier.

    Uses same input every iteration. If output varies across iterations, that's ND.
    """
    sender_core_range_set = get_sender_core_range_set()

    # Bounding box grid for anchor tensor
    full_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 9))
    full_grid_set = ttnn.CoreRangeSet({full_grid})
    num_grid_cores = 13 * 10  # 130

    logger.info(
        f"Testing GatherReduce3: {NUM_SENDERS} senders (12x8+8x2) -> receiver ({RECEIVER_CORE.x},{RECEIVER_CORE.y})"
    )
    logger.info(f"  data_size_bytes={DATA_SIZE_BYTES}, half_num_cores={HALF_NUM_CORES}")
    logger.info(f"  reduce: {NUM_REDUCE_TILES} tiles of 32x32, scratch={SCRATCH_SIZE_BYTES}B")
    logger.info(f"  iterations={num_iterations} (same input each time, checking output consistency)")

    # Fixed input for all iterations
    torch.manual_seed(42)
    torch_input = torch.randn(1, TOTAL_WIDTH, dtype=torch.bfloat16)

    # Allocate tensors ONCE — reuse across all iterations to keep L1 addresses stable
    anchor = create_scratch_anchor(device, full_grid_set, num_grid_cores)
    ttnn_input = create_sharded_input(device, torch_input, sender_core_range_set)
    ttnn_output = create_output_tensor(device, RECEIVER_CORE)
    logger.info(
        f"  L1 addrs: anchor=0x{anchor.buffer_address():x}, "
        f"input=0x{ttnn_input.buffer_address():x}, output=0x{ttnn_output.buffer_address():x}"
    )

    reference_output = None
    all_pass = True
    pcc_values = []

    for i in range(num_iterations):
        logger.info(f"  [{i+1:>2}] Building program...")
        program = build_program(device, ttnn_input, anchor, ttnn_output, sender_core_range_set, RECEIVER_CORE)
        logger.info(f"  [{i+1:>2}] Program built. Launching generic_op...")
        sys.stdout.flush()
        sys.stderr.flush()
        result = ttnn.generic_op([ttnn_input, anchor, ttnn_output], program)
        logger.info(f"  [{i+1:>2}] generic_op completed. Reading output...")
        sys.stdout.flush()
        sys.stderr.flush()

        result_torch = ttnn.to_torch(ttnn_output)
        logger.info(f"  [{i+1:>2}] output read OK, shape={result_torch.shape}")

        if i < 3:
            logger.info(f"  [{i+1:>2}] output[:5]={result_torch.flatten()[:5].tolist()}")
            logger.info(f"  [{i+1:>2}] output nonzero={result_torch.count_nonzero().item()}/{result_torch.numel()}")

        if reference_output is None:
            reference_output = result_torch.clone()
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

    anchor.deallocate()
    ttnn_input.deallocate()
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
    ), f"ND GatherReduce3 failed: {sum(1 for p in pcc_values if p < 0.9999)}/{len(pcc_values)} iterations differ from reference"
