# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test mcast to 130 cores (full 13x10 BH grid) with different matmul configurations.

Tests:
- test_mcast_130_cores_sender_inside: 112-core matmul (12x10 - 8 DRAM workers)
- test_mcast_130_cores_128_matmul: 128-core matmul (12x10 + 8 from col 12)

Mcast source: (12, 9) - INSIDE the mcast grid (corner)
Mcast grid: (0,0) to (12,9) = 13x10 = 130 cores

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_mcast_130_cores.py -v -s
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

# DRAM worker positions on BH 13x10 grid (col, row)
# These are in mcast grid but excluded from matmul
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
    """Build CoreRangeSet excluding specified positions, using contiguous row-segments."""
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


def test_mcast_130_cores_sender_inside(device):
    """
    Mcast to 130 cores (full 13x10 grid), matmul on 112 cores (12x10 - 8 DRAM).

    Mcast to full grid, matmul only on 12x10 subgrid minus DRAM workers.
    - mcast_is_part_of_receiver_grid=True (sender in grid)
    - mcast_loopback=True (sender receives own mcast)

    - Mcast grid: (0,0) to (12,9) = 13x10 = 130 cores
    - Matmul grid: (0,0) to (11,9) = 12x10 = 120 cores, minus 8 DRAM = 112 cores
    - Phantom cores: 18 (10 in column 12 including sender + 8 DRAM workers)
    - Sender: (12,9) - phantom core
    """
    device_grid = device.compute_with_storage_grid_size()
    logger.info(f"Device grid: {device_grid.x}x{device_grid.y}")

    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for 13x10")

    # Sender at (12, 9) - INSIDE the mcast grid
    mcast_core = ttnn.CoreCoord(12, 9)

    # Full mcast grid: 13x10 = 130 cores (entire device)
    mcast_grid_start = ttnn.CoreCoord(0, 0)
    mcast_grid_end = ttnn.CoreCoord(12, 9)
    mcast_grid = ttnn.CoreRange(mcast_grid_start, mcast_grid_end)
    full_mcast_grid = ttnn.CoreRangeSet({mcast_grid})

    mcast_num_cores = 13 * 10  # 130

    # Sender is INSIDE mcast grid
    is_sender_in_mcast_grid = mcast_grid.contains(mcast_core)
    logger.info(f"Sender ({mcast_core.x}, {mcast_core.y}) in mcast grid: {is_sender_in_mcast_grid}")

    # Matmul grid: 12x10 = 120 cores (columns 0-11), minus 8 DRAM workers = 112 cores
    # Column 12 (10 cores including sender) are all phantom cores
    matmul_grid_start = ttnn.CoreCoord(0, 0)
    matmul_grid_end = ttnn.CoreCoord(11, 9)  # 12x10, not 13x10

    # Build matmul grid excluding DRAM workers only (column 12 is already excluded)
    matmul_core_grid = build_matmul_core_grid(matmul_grid_start, matmul_grid_end, DRAM_WORKER_POSITIONS)
    matmul_num_cores = matmul_core_grid.num_cores()
    phantom_num_cores = mcast_num_cores - matmul_num_cores  # Column 12 + DRAM workers

    logger.info(f"Mcast grid: 13x10 = {mcast_num_cores} cores")
    logger.info(f"Matmul grid: 12x10 - 8 DRAM = {matmul_num_cores} cores")
    logger.info(f"Phantom cores: {phantom_num_cores} (10 in col 12 + 8 DRAM workers)")

    # Matrix dimensions - weights/output on 112 matmul cores only
    M = 1
    K = 256
    N_per_core = 64
    N = N_per_core * matmul_num_cores  # 112 * 32 = 3584

    logger.info(f"Shape: [{M}, {K}] × [{K}, {N}] → [{M}, {N}]")

    # Tiles
    a_tile = ttnn.Tile([M, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    # Test tensors
    torch.manual_seed(42)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)

    # Input on sender (which is inside mcast grid)
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

    # Weights on 112 matmul cores only (not phantom cores)
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, N_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=b_tile,
    )

    # Output on 112 matmul cores only (not phantom cores)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, N_per_core), ttnn.ShardOrientation.ROW_MAJOR),
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

    full_device_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid.x - 1, device_grid.y - 1))]
    )

    src_cb, dst_cb, in1_cb, out_cb = 0, 1, 2, 3

    # CBs - note: sender is part of mcast grid, so it needs dst_cb too
    src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(src_cb, ttnn_input)

    # Placeholder for src_cb on non-sender cores (for consistent L1 layout)
    # Exclude sender since it has the actual tensor
    non_sender_grid_ranges = []
    for y in range(mcast_grid_start.y, mcast_grid_end.y + 1):
        for x in range(mcast_grid_start.x, mcast_grid_end.x + 1):
            if not (x == mcast_core.x and y == mcast_core.y):
                non_sender_grid_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)))
    non_sender_grid = ttnn.CoreRangeSet(non_sender_grid_ranges)

    src_cb_placeholder_descriptor = ttnn.CBDescriptor(
        total_size=k_num_tiles * input_tile_size,
        core_ranges=non_sender_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=src_cb,
                data_format=data_format,
                page_size=input_tile_size,
                tile=ttnn.TileDescriptor(in0_tile),
            )
        ],
    )

    # dst_cb on ALL cores (including sender for loopback)
    dst_cb_descriptor = ttnn.CBDescriptor(
        total_size=k_num_tiles * input_tile_size,
        core_ranges=full_mcast_grid,
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

    # All cores = full mcast grid (sender is inside)
    all_cores = full_mcast_grid

    # mcast_grid_cores = all cores that receive mcast data
    # With loopback=true (sender is matmul core), ALL 130 cores receive including sender
    mcast_grid_cores = full_mcast_grid

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
            # Sender IS part of receiver grid AND receives (loopback=true)
            ("mcast_is_part_of_receiver_grid", True),
            # loopback=true: sender receives its own mcast (even though not a matmul core)
            ("mcast_loopback", True),
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
            # 112 cores do matmul (12x10=120 - 8 DRAM workers)
            UnifiedCompileTimeCoreDescriptor("is_matmul_core", matmul_core_grid, 1, 0),
            # ALL 130 cores receive mcast data (18 phantom cores with loopback)
            UnifiedCompileTimeCoreDescriptor("is_mcast_grid_core", mcast_grid_cores, 1, 0),
        ],
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=unified_kernel.get_kernel_descriptors().kernels,
        cbs=[
            src_cb_descriptor,
            src_cb_placeholder_descriptor,
            dst_cb_descriptor,
            in1_cb_descriptor,
            out_cb_descriptor,
        ],
        semaphores=[
            ttnn.SemaphoreDescriptor(id=0, core_ranges=full_device_grid, initial_value=0),
            ttnn.SemaphoreDescriptor(id=1, core_ranges=full_device_grid, initial_value=0),
        ],
    )

    # Execute
    logger.info(f"Running: mcast to {mcast_num_cores}, compute on {matmul_num_cores}...")
    ttnn_result = ttnn.generic_op([ttnn_input, ttnn_weights, ttnn_output], program_descriptor)

    # Get output
    output_torch = ttnn.to_torch(ttnn_result)
    logger.info(f"Output shape: {output_torch.shape}")

    # Golden: just compute matmul - all output columns are valid (no phantom columns)
    torch_expected = McastMatmulMultiCore.golden(torch_input.float(), torch_weights.float()).bfloat16()

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.98)
    logger.info(f"PCC: {pcc_message}")

    assert passing, pcc_message
    logger.info(f"✓ 130-core mcast with sender inside grid PASSED")


def test_mcast_130_cores_128_matmul(device):
    """
    Mcast to 130 cores (full 13x10 grid), matmul on 128 cores.

    128 matmul cores = 120 (columns 0-11) + 8 (column 12, rows 0-7)

         Col:  0   1   2   3   4   5   6   7   8   9  10  11  12
    Row 0:    M   M   M   M   M   M   M   M   M   M   M   M   M
    Row 1:    M   M   M   M   M   M   M   M   M   M   M   M   M
    Row 2:    M   M   M   M   M   M   M   M   M   M   M   M   M
    Row 3:    M   M   M   M   M   M   M   M   M   M   M   M   M
    Row 4:    M   M   M   M   M   M   M   M   M   M   M   M   M
    Row 5:    M   M   M   M   M   M   M   M   M   M   M   M   M
    Row 6:    M   M   M   M   M   M   M   M   M   M   M   M   M
    Row 7:    M   M   M   M   M   M   M   M   M   M   M   M   M
    Row 8:    M   M   M   M   M   M   M   M   M   M   M   M   P  <- (12,8) phantom
    Row 9:    M   M   M   M   M   M   M   M   M   M   M   M   S  <- (12,9) sender

    Legend: M = matmul (128), P = phantom (1), S = sender (1)

    - mcast_is_part_of_receiver_grid=True (sender in grid)
    - mcast_loopback=True (sender receives own mcast)
    - Mcast grid: (0,0) to (12,9) = 13x10 = 130 cores
    - Matmul grid: 128 cores (120 in cols 0-11 + 8 in col 12 rows 0-7)
    - Phantom cores: 2 ((12,8) + sender (12,9))
    """
    device_grid = device.compute_with_storage_grid_size()
    logger.info(f"Device grid: {device_grid.x}x{device_grid.y}")

    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for 13x10")

    # Sender at (12, 9) - INSIDE the mcast grid
    mcast_core = ttnn.CoreCoord(12, 9)

    # Full mcast grid: 13x10 = 130 cores (entire device)
    mcast_grid_start = ttnn.CoreCoord(0, 0)
    mcast_grid_end = ttnn.CoreCoord(12, 9)
    mcast_grid = ttnn.CoreRange(mcast_grid_start, mcast_grid_end)
    full_mcast_grid = ttnn.CoreRangeSet({mcast_grid})

    mcast_num_cores = 13 * 10  # 130

    # Sender is INSIDE mcast grid
    is_sender_in_mcast_grid = mcast_grid.contains(mcast_core)
    logger.info(f"Sender ({mcast_core.x}, {mcast_core.y}) in mcast grid: {is_sender_in_mcast_grid}")

    # Matmul grid: columns 0-11 (120 cores) + column 12 rows 0-7 (8 cores) = 128 cores
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9)),  # 120 cores
            ttnn.CoreRange(ttnn.CoreCoord(12, 0), ttnn.CoreCoord(12, 7)),  # 8 cores
        ]
    )
    matmul_num_cores = 128
    phantom_num_cores = mcast_num_cores - matmul_num_cores  # 130 - 128 = 2

    logger.info(f"Mcast grid: 13x10 = {mcast_num_cores} cores")
    logger.info(f"Matmul grid: 120 (cols 0-11) + 8 (col 12 rows 0-7) = {matmul_num_cores} cores")
    logger.info(f"Phantom cores: {phantom_num_cores} ((12,8) + sender (12,9))")

    # Matrix dimensions - weights/output on 128 matmul cores
    M = 1
    K = 256
    N_per_core = 64
    N = N_per_core * matmul_num_cores  # 64 * 128 = 8192

    logger.info(f"Shape: [{M}, {K}] × [{K}, {N}] → [{M}, {N}]")

    # Tiles
    a_tile = ttnn.Tile([M, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    # Test tensors
    torch.manual_seed(42)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)

    # Input on sender (which is inside mcast grid)
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

    # Weights on 128 matmul cores only (not phantom cores)
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, N_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=b_tile,
    )

    # Output on 128 matmul cores only (not phantom cores)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, N_per_core), ttnn.ShardOrientation.ROW_MAJOR),
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

    full_device_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid.x - 1, device_grid.y - 1))]
    )

    src_cb, dst_cb, in1_cb, out_cb = 0, 1, 2, 3

    # CBs - note: sender is part of mcast grid, so it needs dst_cb too
    src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(src_cb, ttnn_input)

    # Placeholder for src_cb on non-sender cores (for consistent L1 layout)
    # Exclude sender since it has the actual tensor
    non_sender_grid_ranges = []
    for y in range(mcast_grid_start.y, mcast_grid_end.y + 1):
        for x in range(mcast_grid_start.x, mcast_grid_end.x + 1):
            if not (x == mcast_core.x and y == mcast_core.y):
                non_sender_grid_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)))
    non_sender_grid = ttnn.CoreRangeSet(non_sender_grid_ranges)

    src_cb_placeholder_descriptor = ttnn.CBDescriptor(
        total_size=k_num_tiles * input_tile_size,
        core_ranges=non_sender_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=src_cb,
                data_format=data_format,
                page_size=input_tile_size,
                tile=ttnn.TileDescriptor(in0_tile),
            )
        ],
    )

    # dst_cb on ALL cores (including sender for loopback)
    dst_cb_descriptor = ttnn.CBDescriptor(
        total_size=k_num_tiles * input_tile_size,
        core_ranges=full_mcast_grid,
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

    # All cores = full mcast grid (sender is inside)
    all_cores = full_mcast_grid

    # mcast_grid_cores = matmul cores + phantom at (12,8)
    # Sender (12,9) receives via loopback (handled by mcast_loopback=True)
    mcast_grid_cores = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9)),  # 120 cores
            ttnn.CoreRange(ttnn.CoreCoord(12, 0), ttnn.CoreCoord(12, 8)),  # 9 cores (incl phantom)
        ]
    )

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
            # Sender IS part of receiver grid AND receives (loopback=true)
            ("mcast_is_part_of_receiver_grid", True),
            # loopback=true: sender receives its own mcast
            ("mcast_loopback", True),
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
            # 128 cores do matmul
            UnifiedCompileTimeCoreDescriptor("is_matmul_core", matmul_core_grid, 1, 0),
            # 129 cores receive mcast data (excludes sender which uses loopback)
            UnifiedCompileTimeCoreDescriptor("is_mcast_grid_core", mcast_grid_cores, 1, 0),
        ],
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=unified_kernel.get_kernel_descriptors().kernels,
        cbs=[
            src_cb_descriptor,
            src_cb_placeholder_descriptor,
            dst_cb_descriptor,
            in1_cb_descriptor,
            out_cb_descriptor,
        ],
        semaphores=[
            ttnn.SemaphoreDescriptor(id=0, core_ranges=full_device_grid, initial_value=0),
            ttnn.SemaphoreDescriptor(id=1, core_ranges=full_device_grid, initial_value=0),
        ],
    )

    # Execute
    logger.info(f"Running: mcast to {mcast_num_cores}, compute on {matmul_num_cores}...")
    ttnn_result = ttnn.generic_op([ttnn_input, ttnn_weights, ttnn_output], program_descriptor)

    # Get output
    output_torch = ttnn.to_torch(ttnn_result)
    logger.info(f"Output shape: {output_torch.shape}")

    # Golden: just compute matmul - all output columns are valid (no phantom columns)
    torch_expected = McastMatmulMultiCore.golden(torch_input.float(), torch_weights.float()).bfloat16()

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.98)
    logger.info(f"PCC: {pcc_message}")

    assert passing, pcc_message
    logger.info(f"✓ 130-core mcast with 128-core matmul PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
