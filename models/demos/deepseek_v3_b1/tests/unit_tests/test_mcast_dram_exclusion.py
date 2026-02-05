# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test mcast to full grid with DRAM worker exclusion pattern.

Pattern: Mcast to N cores, compute on M < N cores.
- Mcast grid: full rectangular bounding box (e.g., 12x10 = 120 cores)
- Matmul cores: subset excluding DRAM workers (e.g., 112 cores)
- Phantom cores: DRAM workers receive mcast but don't compute

DRAM worker positions (excluded from matmul):
    (0,0), (0,3), (0,7), (0,9) - column 0
    (7,1), (7,4), (7,6), (7,9) - column 7

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_mcast_dram_exclusion.py -v -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.mcast_matmul.op import McastMatmulMultiCore
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

# DRAM worker positions to exclude from matmul (col, row)
DRAM_WORKER_POSITIONS = [
    (0, 0),
    (0, 3),
    (0, 7),
    (0, 9),
    (7, 1),
    (7, 4),
    (7, 6),
    (7, 9),
]


def build_matmul_core_grid(mcast_grid_start, mcast_grid_end, excluded_positions):
    """
    Build a CoreRangeSet for matmul cores, excluding specified positions.
    Creates contiguous row-segments where possible.
    """
    excluded_set = set(excluded_positions)
    core_ranges = []

    for y in range(mcast_grid_start.y, mcast_grid_end.y + 1):
        segment_start = None
        for x in range(mcast_grid_start.x, mcast_grid_end.x + 1):
            is_excluded = (x, y) in excluded_set
            if not is_excluded:
                if segment_start is None:
                    segment_start = x
            else:
                if segment_start is not None:
                    core_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(segment_start, y), ttnn.CoreCoord(x - 1, y)))
                    segment_start = None
        if segment_start is not None:
            core_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(segment_start, y), ttnn.CoreCoord(mcast_grid_end.x, y)))

    return ttnn.CoreRangeSet(core_ranges)


def test_mcast_dram_exclusion_120to112(device):
    """
    Mcast to 120 cores, compute on 112 (excluding 8 DRAM workers).

    Strategy:
    - Weights/output sharded across FULL 120-core contiguous grid
    - DRAM workers (is_matmul_core=0) receive mcast, reset semaphore, skip compute
    - Output from DRAM workers is zeros (unused)
    - Only 112 cores actually produce valid output
    """
    device_grid = device.compute_with_storage_grid_size()
    logger.info(f"Device grid: {device_grid.x}x{device_grid.y}")

    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small")

    # Mcast source outside mcast grid
    mcast_core = ttnn.CoreCoord(12, 0)

    # Full mcast grid: 12x10 = 120 cores
    mcast_grid_start = ttnn.CoreCoord(0, 0)
    mcast_grid_end = ttnn.CoreCoord(11, 9)
    mcast_grid = ttnn.CoreRange(mcast_grid_start, mcast_grid_end)
    full_mcast_grid = ttnn.CoreRangeSet({mcast_grid})

    mcast_num_cores = 12 * 10  # 120

    # Matmul cores (112) - used for is_matmul_core flag
    matmul_core_grid = build_matmul_core_grid(mcast_grid_start, mcast_grid_end, DRAM_WORKER_POSITIONS)
    matmul_num_cores = matmul_core_grid.num_cores()
    phantom_num_cores = mcast_num_cores - matmul_num_cores

    logger.info(f"Mcast grid: 12x10 = {mcast_num_cores} cores")
    logger.info(f"Matmul cores: {matmul_num_cores}")
    logger.info(f"Phantom (DRAM workers): {phantom_num_cores}")

    # Matrix dimensions - weights on ALL 120 cores
    M = 1
    K = 256
    N_per_core = 32
    N = N_per_core * mcast_num_cores  # 120 * 32 = 3840

    logger.info(f"Shape: [{M}, {K}] × [{K}, {N}] → [{M}, {N}]")

    # Tiles
    a_tile = ttnn.Tile([M, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    # Test tensors
    torch.manual_seed(42)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)

    # Input on sender
    sender_grid = ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)})
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sender_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=a_tile,
    )

    # Weights on FULL 120-core grid (contiguous, easier sharding)
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(full_mcast_grid, (K, N_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=b_tile,
    )

    # Output on FULL 120-core grid
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(full_mcast_grid, (M, N_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    # Build program
    data_format = ttnn_input.dtype
    in0_tile = ttnn_input.get_tile()
    k_num_tiles = K // in0_tile.tile_shape[1]
    out_w_per_core = N_per_core // out_tile.tile_shape[1]
    input_tile_size = in0_tile.get_tile_size(data_format)

    mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
    mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)

    # Phantom cores = DRAM workers
    phantom_cores = [ttnn.CoreCoord(x, y) for x, y in DRAM_WORKER_POSITIONS]
    phantom_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in phantom_cores])

    full_device_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid.x - 1, device_grid.y - 1))]
    )

    src_cb, dst_cb, in1_cb, out_cb = 0, 1, 2, 3

    # CBs
    src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(src_cb, ttnn_input)
    src_cb_placeholder_descriptor = ttnn.CBDescriptor(
        total_size=k_num_tiles * input_tile_size,
        core_ranges=full_mcast_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=src_cb,
                data_format=data_format,
                page_size=input_tile_size,
                tile=ttnn.TileDescriptor(in0_tile),
            )
        ],
    )
    dst_cb_descriptor = ttnn.CBDescriptor(
        total_size=k_num_tiles * input_tile_size,
        core_ranges=full_mcast_grid.merge(sender_grid),
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=dst_cb,
                data_format=data_format,
                page_size=input_tile_size,
                tile=ttnn.TileDescriptor(in0_tile),
            )
        ],
    )
    in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in1_cb, ttnn_weights)
    out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, ttnn_output)

    # All cores = sender + full mcast grid
    all_cores = full_mcast_grid.merge(sender_grid)

    unified_kernel = UnifiedKernelDescriptor(
        kernel_source="models/demos/deepseek_v3_b1/micro_ops/mcast_matmul/kernels/mcast_matmul_kernel.cpp",
        core_ranges=all_cores,
        ncrisc_named_compile_time_args=[
            ("mcast_src_cb", src_cb),
            ("mcast_src_num_pages", k_num_tiles),
            ("mcast_data_receiver_semaphore", 1),
            ("mcast_dst_cb", dst_cb),
            ("mcast_dst_num_pages", k_num_tiles),
            ("matmul_in1", in1_cb),
            ("matmul_in1_num_pages", k_num_tiles * out_w_per_core),
        ],
        brisc_named_compile_time_args=[
            ("mcast_dest_noc_start_x", mcast_dest_noc_start.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end.y),
            ("mcast_num_cores", mcast_num_cores),
            ("mcast_data_sender_semaphore", 0),
            ("mcast_data_receiver_semaphore", 1),
            ("mcast_data_size_bytes", k_num_tiles * input_tile_size),
            ("mcast_src_cb", src_cb),
            ("mcast_src_num_pages", k_num_tiles),
            ("mcast_dst_cb", dst_cb),
            ("mcast_is_part_of_receiver_grid", False),
            ("mcast_loopback", 0),
        ],
        trisc_named_compile_time_args=[
            ("mcast_dst_cb", dst_cb),
            ("matmul_in1", in1_cb),
            ("matmul_out", out_cb),
            ("matmul_k_num_tiles", k_num_tiles),
            ("matmul_out_w_per_core", out_w_per_core),
        ],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
        unified_compile_time_core_descriptors=[
            UnifiedCompileTimeCoreDescriptor("is_sender_core", mcast_core, 1, 0),
            # ONLY 112 cores are matmul cores (excludes DRAM workers)
            UnifiedCompileTimeCoreDescriptor("is_matmul_core", matmul_core_grid, 1, 0),
            # ALL 120 cores in mcast grid receive data
            UnifiedCompileTimeCoreDescriptor("is_mcast_grid_core", full_mcast_grid, 1, 0),
        ],
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=unified_kernel.get_kernel_descriptors().kernels,
        cbs=[src_cb_descriptor, src_cb_placeholder_descriptor, dst_cb_descriptor, in1_cb_descriptor, out_cb_descriptor],
        semaphores=[
            ttnn.SemaphoreDescriptor(id=0, core_ranges=full_device_grid, initial_value=0),
            ttnn.SemaphoreDescriptor(id=1, core_ranges=full_device_grid, initial_value=0),
        ],
    )

    # Execute
    logger.info(f"Running: mcast to {mcast_num_cores}, compute on {matmul_num_cores}, {phantom_num_cores} phantom...")
    ttnn_result = ttnn.generic_op([ttnn_input, ttnn_weights, ttnn_output], program_descriptor)

    # Get output
    output_torch = ttnn.to_torch(ttnn_result)
    logger.info(f"Output shape: {output_torch.shape}")

    # Golden: compute full matmul, then zero out DRAM worker columns
    torch_expected = McastMatmulMultiCore.golden(torch_input.float(), torch_weights.float()).bfloat16()

    # Zero out columns corresponding to DRAM workers
    # Core (x, y) in row-major order maps to columns [core_idx * N_per_core : (core_idx+1) * N_per_core]
    for x, y in DRAM_WORKER_POSITIONS:
        core_idx = y * 12 + x  # row-major within 12x10 grid
        col_start = core_idx * N_per_core
        col_end = col_start + N_per_core
        torch_expected[:, col_start:col_end] = 0

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.98)
    logger.info(f"PCC: {pcc_message}")

    # Also check non-zero columns have good PCC
    non_zero_mask = torch_expected.abs().sum(dim=0) > 0
    if non_zero_mask.any():
        expected_valid = torch_expected[:, non_zero_mask]
        output_valid = output_torch[:, non_zero_mask]
        passing_valid, pcc_valid = comp_pcc(expected_valid, output_valid, 0.98)
        logger.info(f"PCC (valid columns only): {pcc_valid}")

    assert passing, pcc_message
    logger.info(f"✓ DRAM exclusion test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
