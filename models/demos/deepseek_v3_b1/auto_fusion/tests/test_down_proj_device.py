# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device test: Auto-fused down_proj vs hand-fused.

Builds the 5-op down_proj graph using auto-fusion infrastructure
(Mcast1 + Mcast2 + Matmul + ResidualAdd + Gather) and compares:
1. Correctness vs golden (PyTorch reference: input @ weights + add_input)
2. Correctness vs hand-fused kernel (same kernel source, hand-built descriptors)
3. Performance via timing (Tracy for detailed profiling)

Grid-configurable: works on any Blackhole device grid size (e.g., 4x4, 7x7, 13x10).
The production 13x10 grid uses 112 matmul cores with DRAM/phantom exclusions.
Smaller grids use a simple rectangular layout with sender at the corner.

NOTE: Requires Blackhole architecture. The matmul micro-op uses custom_mm LLK
intrinsics that are only available on Blackhole (not Wormhole).

Run:
    pytest models/demos/deepseek_v3_b1/auto_fusion/tests/test_down_proj_device.py -xvs
"""

import time
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph
from models.demos.deepseek_v3_b1.auto_fusion.specs import GATHER, MATMUL, MCAST, RESIDUAL_ADD
from models.demos.deepseek_v3_b1.auto_fusion.types import CBConfig
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

# ===========================================================================
# Grid topology helper
# ===========================================================================


@dataclass
class DownProjTopology:
    """Grid topology for down_proj. Derived from grid dimensions."""

    grid_x: int
    grid_y: int
    sender_core: object  # ttnn.CoreCoord
    sender_core_grid: object  # ttnn.CoreRangeSet (single core)
    mcast_grid: object  # ttnn.CoreRange (full rectangle)
    mcast_grid_set: object  # ttnn.CoreRangeSet
    matmul_core_grid: object  # ttnn.CoreRangeSet (all except sender)
    mcast_receiver_grid: object  # ttnn.CoreRangeSet (all except sender)
    num_mcast_cores: int
    num_matmul_cores: int
    matmul_cores_list: list  # List[CoreCoord]


def build_topology(grid_x: int, grid_y: int) -> DownProjTopology:
    """
    Build a down_proj topology for a given grid size.

    Sender/gather core at (grid_x-1, grid_y-1).
    All other cores in the rectangle are matmul cores.
    No DRAM worker / phantom exclusions for arbitrary grids.
    """
    sender = ttnn.CoreCoord(grid_x - 1, grid_y - 1)
    sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender, sender)])

    mcast_grid = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(grid_x - 1, grid_y - 1),
    )
    mcast_grid_set = ttnn.CoreRangeSet([mcast_grid])
    num_mcast = grid_x * grid_y

    # Matmul cores = all cores except sender
    matmul_ranges = []
    for row in range(grid_y):
        for col in range(grid_x):
            if col == sender.x and row == sender.y:
                continue
            matmul_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
    matmul_core_grid = ttnn.CoreRangeSet(matmul_ranges)
    matmul_cores_list = ttnn.corerange_to_cores(matmul_core_grid)

    # Mcast receivers = all except sender (same as matmul for simple grids)
    receiver_ranges = []
    for row in range(grid_y):
        for col in range(grid_x):
            if col == sender.x and row == sender.y:
                continue
            receiver_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
    mcast_receiver_grid = ttnn.CoreRangeSet(receiver_ranges)

    return DownProjTopology(
        grid_x=grid_x,
        grid_y=grid_y,
        sender_core=sender,
        sender_core_grid=sender_grid,
        mcast_grid=mcast_grid,
        mcast_grid_set=mcast_grid_set,
        matmul_core_grid=matmul_core_grid,
        mcast_receiver_grid=mcast_receiver_grid,
        num_mcast_cores=num_mcast,
        num_matmul_cores=len(matmul_cores_list),
        matmul_cores_list=matmul_cores_list,
    )


# ===========================================================================
# Tensor creation
# ===========================================================================


def create_tensors(device, topo: DownProjTopology, M, K, N_per_core, weights_dtype=ttnn.bfloat8_b, seed=42):
    """
    Create input/output tensors for down_proj on the given topology.

    Returns: (golden, input_tensor, weights_tensor, output_tensor, add_input_tensor)
    """
    N = N_per_core * topo.num_matmul_cores

    torch.manual_seed(seed)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)
    torch_add_input = torch.randn((M, N), dtype=torch.bfloat16)

    golden = (torch_input.float() @ torch_weights.float() + torch_add_input.float()).bfloat16()

    a_tile = ttnn.Tile([M, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    # Input: [M, K] HEIGHT_SHARDED on sender core
    input_shard = ttnn.ShardSpec(topo.sender_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem,
        tile=a_tile,
    )

    # Weights: [K, N] WIDTH_SHARDED across matmul cores
    weights_shard = ttnn.ShardSpec(topo.matmul_core_grid, (K, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    weights_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights_shard)
    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem,
        tile=b_tile,
    )

    # Output: [M, N] HEIGHT_SHARDED on sender core
    output_shard = ttnn.ShardSpec(topo.sender_core_grid, (M, N), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem,
        tile=out_tile,
    )

    # Add input: [M, N] HEIGHT_SHARDED on sender core
    add_shard = ttnn.ShardSpec(topo.sender_core_grid, (M, N), ttnn.ShardOrientation.ROW_MAJOR)
    add_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, add_shard)
    ttnn_add = ttnn.from_torch(
        torch_add_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=add_mem,
        tile=out_tile,
    )

    return golden, ttnn_input, ttnn_weights, ttnn_output, ttnn_add


# ===========================================================================
# Hand-fused runner (same kernel source, hand-built descriptors)
# ===========================================================================


def run_hand_fused(
    device, topo: DownProjTopology, input_tensor, weights_tensor, output_tensor, add_input_tensor, M, K, N_per_core
):
    """
    Run down_proj using the hand-fused kernel source with manually built descriptors.
    Same kernel source as DownProj.op() but with configurable grid topology.
    """
    data_format = input_tensor.dtype
    input_tile = input_tensor.get_tile()
    input_tile_size = input_tile.get_tile_size(data_format)
    k_num_tiles = K // input_tile.tile_shape[1]

    TILE_1x32 = ttnn.Tile((1, 32))
    tile_1x32_size = TILE_1x32.get_tile_size(data_format)
    out_w = N_per_core // TILE_1x32.tile_shape[1]
    total_residual_tiles = topo.num_matmul_cores * out_w

    # NOC coordinates
    gather_noc = device.worker_core_from_logical_core(topo.sender_core)
    mcast_noc_start = device.worker_core_from_logical_core(topo.mcast_grid.start)
    mcast_noc_end = device.worker_core_from_logical_core(topo.mcast_grid.end)

    gather_receiver_data_addr = output_tensor.buffer_address()
    mcast_is_part_of_receiver_grid = topo.mcast_grid.contains(topo.sender_core)

    # CB indices (same as hand-fused op.py layout)
    mcast_src_cb = 0
    mcast_dst_cb = 1
    matmul_in1_cb = 2
    matmul_out_cb = 3
    gather_dst_cb = 4
    res_mcast_src_cb = 5
    res_mcast_dst_cb = 6
    res_add_out_cb = 7

    # Per-core gather indices
    per_core_gather_idx = PerCoreCompileTimeDescriptor(
        named_compile_time_arg="gather_sender_idx",
        core_values=[(core, idx) for idx, core in enumerate(topo.matmul_cores_list)],
        other_value=0,
    )

    # NCRISC CT args
    ncrisc_ct = [
        ("mcast_src_cb", mcast_src_cb),
        ("mcast_src_num_pages", k_num_tiles),
        ("mcast_data_receiver_semaphore", 1),
        ("mcast_dst_cb", mcast_dst_cb),
        ("mcast_dst_num_pages", k_num_tiles),
        ("mcast2_src_cb", res_mcast_src_cb),
        ("mcast2_src_num_pages", total_residual_tiles),
        ("mcast2_data_receiver_semaphore", 4),
        ("mcast2_dst_cb", res_mcast_dst_cb),
        ("mcast2_dst_num_pages", total_residual_tiles),
        ("matmul_in0", mcast_dst_cb),
        ("matmul_in1", matmul_in1_cb),
        ("matmul_out", matmul_out_cb),
        ("matmul_k_num_tiles", k_num_tiles),
        ("matmul_out_w_per_core", out_w),
        ("residual_add_out_w", out_w),
        ("gather_dest_noc_x", gather_noc.x),
        ("gather_dest_noc_y", gather_noc.y),
        ("gather_data_size_bytes", out_w * tile_1x32_size),
        ("gather_receiver_semaphore_id", 2),
        ("gather_src_cb", res_add_out_cb),
        ("gather_src_num_pages", out_w),
        ("gather_sender_grid_start_x", 0),
        ("gather_sender_grid_start_y", 0),
        ("gather_sender_grid_end_x", 0),
        ("gather_sender_grid_end_y", 0),
        ("gather_row_major", 1),
        ("gather_receiver_data_addr", gather_receiver_data_addr),
    ]

    # BRISC CT args
    brisc_ct = [
        ("mcast_dest_noc_start_x", mcast_noc_start.x),
        ("mcast_dest_noc_start_y", mcast_noc_start.y),
        ("mcast_dest_noc_end_x", mcast_noc_end.x),
        ("mcast_dest_noc_end_y", mcast_noc_end.y),
        ("mcast_num_cores", topo.num_mcast_cores),
        ("mcast_data_sender_semaphore", 0),
        ("mcast_data_receiver_semaphore", 1),
        ("mcast_data_size_bytes", k_num_tiles * input_tile_size),
        ("mcast_src_cb", mcast_src_cb),
        ("mcast_src_num_pages", k_num_tiles),
        ("mcast_dst_cb", mcast_dst_cb),
        ("mcast_is_part_of_receiver_grid", mcast_is_part_of_receiver_grid),
        ("mcast2_data_sender_semaphore", 0),
        ("mcast2_data_receiver_semaphore", 4),
        ("mcast2_data_size_bytes", total_residual_tiles * tile_1x32_size),
        ("mcast2_src_cb", res_mcast_src_cb),
        ("mcast2_src_num_pages", total_residual_tiles),
        ("mcast2_dst_cb", res_mcast_dst_cb),
        ("residual_add_out_w", out_w),
        ("gather_noc0_num_senders", topo.num_matmul_cores),
        ("gather_noc1_num_senders", 0),
        ("gather_noc0_receiver_semaphore_id", 2),
        ("gather_noc1_receiver_semaphore_id", 3),
        ("gather_dst_cb", gather_dst_cb),
        ("gather_dst_num_pages", topo.num_matmul_cores * out_w),
    ]

    # TRISC CT args
    trisc_ct = [
        ("matmul_in0", mcast_dst_cb),
        ("matmul_in1", matmul_in1_cb),
        ("matmul_out", matmul_out_cb),
        ("matmul_k_num_tiles", k_num_tiles),
        ("matmul_out_w_per_core", out_w),
        ("residual_add_in0", matmul_out_cb),
        ("residual_add_in1", res_mcast_dst_cb),
        ("residual_add_out", res_add_out_cb),
        ("residual_add_out_w", out_w),
        ("residual_add_total_in1_tiles", total_residual_tiles),
    ]

    # CB descriptors
    tile_desc = ttnn.TileDescriptor(TILE_1x32)

    def internal_cb(idx, total_size, page_size, core_ranges):
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=idx,
            data_format=data_format,
            page_size=page_size,
            tile=tile_desc,
        )
        return ttnn.CBDescriptor(total_size=total_size, core_ranges=core_ranges, format_descriptors=[fmt])

    cb_descs = [
        ttnn.cb_descriptor_from_sharded_tensor(mcast_src_cb, input_tensor),
        internal_cb(mcast_dst_cb, k_num_tiles * input_tile_size, input_tile_size, topo.mcast_grid_set),
        ttnn.cb_descriptor_from_sharded_tensor(matmul_in1_cb, weights_tensor),
        internal_cb(matmul_out_cb, out_w * tile_1x32_size, tile_1x32_size, topo.matmul_core_grid),
        ttnn.cb_descriptor_from_sharded_tensor(gather_dst_cb, output_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(res_mcast_src_cb, add_input_tensor),
        internal_cb(res_mcast_dst_cb, total_residual_tiles * tile_1x32_size, tile_1x32_size, topo.mcast_grid_set),
        internal_cb(res_add_out_cb, out_w * tile_1x32_size, tile_1x32_size, topo.matmul_core_grid),
    ]

    # Core descriptors
    core_descs = [
        UnifiedCompileTimeCoreDescriptor("is_mcast_sender_core", topo.sender_core_grid, 1, 0),
        UnifiedCompileTimeCoreDescriptor("is_mcast_receiver_core", topo.mcast_receiver_grid, 1, 0),
        UnifiedCompileTimeCoreDescriptor("is_matmul_core", topo.matmul_core_grid, 1, 0),
        UnifiedCompileTimeCoreDescriptor("is_gather_receiver_core", topo.sender_core_grid, 1, 0),
        UnifiedCompileTimeCoreDescriptor("gather_use_per_core_sender_idx", topo.matmul_core_grid, 1, 0),
    ]

    # Semaphores
    full_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(
                    device.compute_with_storage_grid_size().x - 1, device.compute_with_storage_grid_size().y - 1
                ),
            )
        ]
    )
    semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=full_grid, initial_value=0) for i in range(5)]

    unified_kernel = UnifiedKernelDescriptor(
        kernel_source="models/demos/deepseek_v3_b1/fused_ops/down_proj/kernels/down_proj_kernel.cpp",
        core_ranges=topo.mcast_grid_set,
        ncrisc_named_compile_time_args=ncrisc_ct,
        brisc_named_compile_time_args=brisc_ct,
        trisc_named_compile_time_args=trisc_ct,
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
        unified_compile_time_core_descriptors=core_descs,
        per_core_compile_time_descriptors=[per_core_gather_idx],
    )

    program_desc = ttnn.ProgramDescriptor(
        kernels=unified_kernel.get_kernel_descriptors().kernels,
        cbs=cb_descs,
        semaphores=semaphores,
    )

    io_tensors = [input_tensor, weights_tensor, output_tensor, add_input_tensor]
    ttnn.generic_op(io_tensors, program_desc)
    return output_tensor


# ===========================================================================
# Auto-fused runner
# ===========================================================================


def run_auto_fused(
    device, topo: DownProjTopology, input_tensor, weights_tensor, output_tensor, add_input_tensor, M, K, N_per_core
):
    """Run the auto-fused down_proj via FusionGraph."""
    data_format = input_tensor.dtype
    input_tile = input_tensor.get_tile()
    input_tile_size = input_tile.get_tile_size(data_format)
    k_num_tiles = K // input_tile.tile_shape[1]

    TILE_1x32 = ttnn.Tile((1, 32))
    tile_1x32_size = TILE_1x32.get_tile_size(data_format)
    out_w = N_per_core // TILE_1x32.tile_shape[1]
    total_residual_tiles = topo.num_matmul_cores * out_w

    # NOC coordinates
    gather_noc = device.worker_core_from_logical_core(topo.sender_core)
    mcast_noc_start = device.worker_core_from_logical_core(topo.mcast_grid.start)
    mcast_noc_end = device.worker_core_from_logical_core(topo.mcast_grid.end)
    gather_receiver_data_addr = output_tensor.buffer_address()

    # Per-core sender indices for gather
    per_core_sender_idx = [(core, idx) for idx, core in enumerate(topo.matmul_cores_list)]

    g = FusionGraph()

    g.add(
        "mcast1",
        MCAST,
        cores=topo.mcast_grid_set,
        ct_args={
            "num_cores": topo.num_mcast_cores,
            "is_part_of_receiver_grid": True,
            "dest_noc_start_x": mcast_noc_start.x,
            "dest_noc_start_y": mcast_noc_start.y,
            "dest_noc_end_x": mcast_noc_end.x,
            "dest_noc_end_y": mcast_noc_end.y,
            "data_sender_semaphore": 0,
            "data_receiver_semaphore": 1,
            "data_size_bytes": k_num_tiles * input_tile_size,
            "src_num_pages": k_num_tiles,
            "dst_num_pages": k_num_tiles,
            "_sender_cores": topo.sender_core_grid,
            "_receiver_cores": topo.mcast_receiver_grid,
        },
    )

    g.add(
        "mcast2",
        MCAST,
        cores=topo.mcast_grid_set,
        ct_args={
            "num_cores": topo.num_mcast_cores,
            "is_part_of_receiver_grid": True,
            "dest_noc_start_x": mcast_noc_start.x,
            "dest_noc_start_y": mcast_noc_start.y,
            "dest_noc_end_x": mcast_noc_end.x,
            "dest_noc_end_y": mcast_noc_end.y,
            "data_sender_semaphore": 0,
            "data_receiver_semaphore": 4,
            "data_size_bytes": total_residual_tiles * tile_1x32_size,
            "src_num_pages": total_residual_tiles,
            "dst_num_pages": total_residual_tiles,
            "_sender_cores": topo.sender_core_grid,
            "_receiver_cores": topo.mcast_receiver_grid,
        },
    )

    g.add(
        "matmul",
        MATMUL,
        cores=topo.matmul_core_grid,
        ct_args={
            "out_w": out_w,
            "transpose": False,
            "fused_activation": 0,
            "k_num_tiles": k_num_tiles,
            "in1_num_pages": k_num_tiles * out_w,
            "pop_in0": True,
            "pop_in1": False,
        },
        inputs={"in0": ("mcast1", "dst")},
    )

    g.add(
        "residual_add",
        RESIDUAL_ADD,
        cores=topo.matmul_core_grid,
        ct_args={
            "out_w": out_w,
            "total_in1_tiles": total_residual_tiles,
            "core_idx": 0,
        },
        inputs={"in0": ("matmul", "out"), "in1": ("mcast2", "dst")},
    )

    g.add(
        "gather",
        GATHER,
        cores=topo.mcast_grid_set,
        ct_args={
            "dest_noc_x": gather_noc.x,
            "dest_noc_y": gather_noc.y,
            "data_size_bytes": out_w * tile_1x32_size,
            "receiver_semaphore_id": 2,
            "src_num_pages": out_w,
            "sender_grid_start_x": 0,
            "sender_grid_start_y": 0,
            "sender_grid_end_x": 0,
            "sender_grid_end_y": 0,
            "row_major": 1,
            "receiver_data_addr": gather_receiver_data_addr,
            "sender_idx": 0,
            "noc0_num_senders": topo.num_matmul_cores,
            "noc1_num_senders": 0,
            "noc0_receiver_semaphore_id": 2,
            "noc1_receiver_semaphore_id": 3,
            "dst_num_pages": topo.num_matmul_cores * out_w,
            "use_per_core_sender_idx": True,
            "_sender_cores": topo.matmul_core_grid,
            "_receiver_cores": topo.sender_core_grid,
            "_per_core_sender_idx": per_core_sender_idx,
            "_sender_idx_default": 0,
        },
        inputs={"src": ("residual_add", "out")},
    )

    # Internal CB configs
    for key, cfg in [
        (("mcast1", "dst"), CBConfig(input_tile_size, k_num_tiles, "bfloat16", M, 32)),
        (("mcast2", "dst"), CBConfig(tile_1x32_size, total_residual_tiles, "bfloat16", 1, 32)),
        (("matmul", "out"), CBConfig(tile_1x32_size, out_w, "bfloat16", 1, 32)),
        (("residual_add", "out"), CBConfig(tile_1x32_size, out_w, "bfloat16", 1, 32)),
    ]:
        g._cb_configs[key] = cfg

    io_tensors = {
        ("mcast1", "src"): input_tensor,
        ("mcast2", "src"): add_input_tensor,
        ("matmul", "in1"): weights_tensor,
        ("gather", "dst"): output_tensor,
    }

    fused_op = g.build(device, io_tensors)
    result = fused_op.run()
    return result, fused_op


# ===========================================================================
# Correctness Tests — small grid (runs on any device)
# ===========================================================================


@pytest.mark.parametrize(
    "grid_x, grid_y, M, K, N_per_core, weights_dtype",
    [
        (4, 4, 1, 256, 64, ttnn.bfloat8_b),  # 15 matmul cores, N=960
        (4, 4, 1, 256, 32, ttnn.bfloat8_b),  # 15 matmul cores, N=480
        (7, 7, 1, 256, 64, ttnn.bfloat8_b),  # 48 matmul cores, N=3072
        (4, 4, 1, 256, 64, ttnn.bfloat4_b),  # bfloat4 weights
    ],
)
def test_auto_fused_vs_golden(
    device, silicon_arch_name, silicon_arch_blackhole, grid_x, grid_y, M, K, N_per_core, weights_dtype
):
    """Auto-fused down_proj matches golden reference (input @ weights + add_input)."""
    topo = build_topology(grid_x, grid_y)
    golden, inp, wgt, out, add = create_tensors(device, topo, M, K, N_per_core, weights_dtype)

    result, _ = run_auto_fused(device, topo, inp, wgt, out, add, M, K, N_per_core)
    output_torch = ttnn.to_torch(result)

    pcc_threshold = 0.97
    passing, pcc_msg = comp_pcc(golden, output_torch, pcc_threshold)
    logger.info(f"Auto-fused vs golden ({grid_x}x{grid_y}, {topo.num_matmul_cores} matmul cores): {pcc_msg}")
    assert passing, f"Auto-fused PCC too low: {pcc_msg}"


@pytest.mark.parametrize(
    "grid_x, grid_y, M, K, N_per_core, weights_dtype",
    [
        (4, 4, 1, 256, 64, ttnn.bfloat8_b),
        (7, 7, 1, 256, 64, ttnn.bfloat8_b),
    ],
)
def test_auto_fused_vs_hand_fused(
    device, silicon_arch_name, silicon_arch_blackhole, grid_x, grid_y, M, K, N_per_core, weights_dtype
):
    """Auto-fused matches hand-fused down_proj kernel (same kernel source, different descriptors)."""
    topo = build_topology(grid_x, grid_y)

    # Hand-fused
    golden, inp, wgt, out, add = create_tensors(device, topo, M, K, N_per_core, weights_dtype)
    hand_result = run_hand_fused(device, topo, inp, wgt, out, add, M, K, N_per_core)
    hand_torch = ttnn.to_torch(hand_result)

    # Auto-fused (fresh tensors)
    _, inp2, wgt2, out2, add2 = create_tensors(device, topo, M, K, N_per_core, weights_dtype)
    auto_result, _ = run_auto_fused(device, topo, inp2, wgt2, out2, add2, M, K, N_per_core)
    auto_torch = ttnn.to_torch(auto_result)

    pcc_threshold = 0.999
    passing, pcc_msg = comp_pcc(hand_torch, auto_torch, pcc_threshold)
    logger.info(f"Auto vs hand ({grid_x}x{grid_y}): {pcc_msg}")
    assert passing, f"Auto vs hand PCC too low: {pcc_msg}"


# ===========================================================================
# Performance Test
# ===========================================================================


@pytest.mark.parametrize(
    "grid_x, grid_y, M, K, N_per_core, weights_dtype, single_run_only",
    [
        (4, 4, 1, 256, 64, ttnn.bfloat8_b, False),
        (7, 7, 1, 256, 64, ttnn.bfloat8_b, False),
    ],
)
def test_auto_fused_performance(
    device, silicon_arch_name, silicon_arch_blackhole, grid_x, grid_y, M, K, N_per_core, weights_dtype, single_run_only
):
    """
    Performance comparison: auto-fused vs hand-fused on configurable grid.

    For accurate device FW timing, use Tracy:
        export TT_METAL_DEVICE_PROFILER=1
        python -m tracy -r -m pytest <this_file>::test_auto_fused_performance -xvs
    """
    topo = build_topology(grid_x, grid_y)
    warmup = 3
    num_runs = 1 if single_run_only else 20

    # --- Hand-fused timing ---
    hand_times = []
    for i in range(warmup + num_runs):
        _, inp, wgt, out, add = create_tensors(device, topo, M, K, N_per_core, weights_dtype, seed=42 + i)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        run_hand_fused(device, topo, inp, wgt, out, add, M, K, N_per_core)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        if i >= warmup:
            hand_times.append(t1 - t0)

    # --- Auto-fused timing ---
    auto_times = []
    for i in range(warmup + num_runs):
        _, inp, wgt, out, add = create_tensors(device, topo, M, K, N_per_core, weights_dtype, seed=42 + i)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        run_auto_fused(device, topo, inp, wgt, out, add, M, K, N_per_core)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        if i >= warmup:
            auto_times.append(t1 - t0)

    hand_avg = sum(hand_times) / len(hand_times) * 1000
    auto_avg = sum(auto_times) / len(auto_times) * 1000
    ratio = auto_avg / hand_avg if hand_avg > 0 else float("inf")

    logger.info(f"\n{'='*60}")
    logger.info(f"PERFORMANCE: {grid_x}x{grid_y} grid, {topo.num_matmul_cores} matmul cores")
    logger.info(f"  M={M}, K={K}, N_per_core={N_per_core}, N={N_per_core * topo.num_matmul_cores}")
    logger.info(f"  Hand-fused:  {hand_avg:.3f} ms (avg of {num_runs} runs)")
    logger.info(f"  Auto-fused:  {auto_avg:.3f} ms (avg of {num_runs} runs)")
    logger.info(f"  Ratio:       {ratio:.3f}x (auto/hand, lower is better)")
    logger.info(f"{'='*60}")


# ===========================================================================
# Full-scale tests (13x10 grid, requires Blackhole/large Wormhole)
# ===========================================================================


@pytest.mark.parametrize(
    "M, K, N_per_core, weights_dtype",
    [
        (1, 256, 64, ttnn.bfloat8_b),
    ],
)
@pytest.mark.requires_grid_size((13, 10))
def test_auto_fused_vs_golden_full_grid(
    device, silicon_arch_name, silicon_arch_blackhole, M, K, N_per_core, weights_dtype
):
    """Full 13x10 grid auto-fused down_proj vs golden."""
    from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj

    topo = build_topology(DownProj.MCAST_GRID_X, DownProj.MCAST_GRID_Y)
    golden, inp, wgt, out, add = create_tensors(device, topo, M, K, N_per_core, weights_dtype)

    result, _ = run_auto_fused(device, topo, inp, wgt, out, add, M, K, N_per_core)
    output_torch = ttnn.to_torch(result)

    pcc_threshold = 0.97
    passing, pcc_msg = comp_pcc(golden, output_torch, pcc_threshold)
    logger.info(f"Auto-fused vs golden (full 13x10): {pcc_msg}")
    assert passing, f"Auto-fused PCC too low: {pcc_msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
