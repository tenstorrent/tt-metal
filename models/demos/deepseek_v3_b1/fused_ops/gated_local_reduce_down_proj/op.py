# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gated Local Reduce + Down Projection fused operation.

Fuses Input Gather + GatedLocalReduce + Mcast1 + Mcast2 + Matmul + ResidualAdd + Output Gather:
  1. Input gather: pull [1,32] tiles from 32 source cores (16 per group) to (12,9)
  2. Gated reduce on sender core (12,9):
     For each K-tile position:
       SiLU(reduce(group1[k])) * reduce(group2[k]) → mcast_src[k]
  3. Mcast1: broadcast [1, K] from (12,9) to 130-core grid
  4. Mcast2: broadcast add input [1, N] from (12,9) to 130-core grid
  5. Matmul: [1, K] x [K, N_per_core] on 112 matmul cores
  6. Residual add: matmul_out + shard(residual) on 112 matmul cores
  7. Output gather: collect [1, N_per_core] from 112 cores to [1, N] on (12,9)

CB Layout:
  CB 0:  group1 gather dest / reduce input (sender, tensor-backed)
  CB 1:  group2 gather dest / reduce input (sender, tensor-backed)
  CB 2:  intermediate (sender, 2 tiles, manual)
  CB 3:  mcast1 source (sender, k_num_tiles tiles, manual, TRISC-filled)
  CB 4:  mcast1 destination / matmul in0 (all 130 cores, manual)
  CB 5:  matmul weights (112 matmul cores, tensor-backed)
  CB 6:  matmul output (112 matmul cores, manual)
  CB 7:  output gather destination (sender, tensor-backed)
  CB 8:  input source group1 (g1 source cores, tensor-backed)
  CB 9:  input source group2 (g2 source cores, tensor-backed)
  CB 10: mcast2 source / add input (sender core, tensor-backed)
  CB 11: mcast2 destination (all 130 cores, manual)
  CB 12: residual add output (112 matmul cores, manual)
"""

from dataclasses import dataclass
from typing import Any

import torch.nn.functional as F

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj
from models.demos.deepseek_v3_b1.fused_ops.face_view_utils import FACE_HEIGHT, FACE_WIDTH, can_use_face_view
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


@dataclass
class _GatedReduceDownProjContext:
    """Holds all computed values needed by GatedLocalReduceDownProjOp helper methods."""

    # Device & format
    full_device_grid: Any
    data_format: Any
    input_tile: Any
    input_tile_size: int
    tile_1x32_size: int
    tile_1x32_descriptor: Any

    # Grids
    mcast_gather_core: Any
    mcast_gather_core_grid: Any
    mcast_grid: Any
    all_cores: Any
    matmul_core_grid: Any
    mcast_receiver_grid: Any
    g1_source_grid: Any
    g2_source_grid: Any
    g1_source_cores: list
    g2_source_cores: list
    num_g1_sources: int
    num_g2_sources: int
    num_mcast_cores: int

    # Dimensions
    out_w_per_core: int
    tiles_per_k: int
    k_num_tiles: int

    # Face-view
    use_face_view: bool
    face_tile_desc: Any
    face_tile_size: Any
    kernel_tiles_per_k: int
    kernel_k_num_tiles: int
    mcast_src_num_pages: int
    reduce_tile_size: int

    # CB indices
    group1_cb: int
    group2_cb: int
    intermed_cb: int
    mcast_src_cb: int
    mcast_dst_cb: int
    matmul_in1_cb: int
    matmul_out_cb: int
    gather_dst_cb: int
    input_src_g1_cb: int
    input_src_g2_cb: int
    residual_add_mcast_src_cb: int
    residual_add_mcast_dst_cb: int
    residual_add_out_cb: int

    # Derived sizes
    mcast_data_size_bytes: int
    mcast_is_part_of_receiver_grid: bool
    total_residual_add_tiles: int
    residual_add_mcast_data_size_bytes: int
    gather_data_size_bytes: int
    gather_src_num_pages: int
    gather_dst_num_pages: int
    gather_noc0_num_senders: int
    gather_noc1_num_senders: int
    ig_data_size_bytes: int
    total_input_tiles: int

    # Semaphore IDs
    mcast_data_sender_semaphore_id: int
    mcast_data_receiver_semaphore_id: int
    gather_noc0_receiver_semaphore_id: int
    gather_noc1_receiver_semaphore_id: int
    ig_g1_receiver_semaphore_id: int
    ig_g2_receiver_semaphore_id: int
    ig_g1_noc1_receiver_semaphore_id: int
    ig_g2_noc1_receiver_semaphore_id: int
    mcast2_data_sender_semaphore_id: int
    mcast2_data_receiver_semaphore_id: int
    num_semaphores: int

    # Addresses
    gather_receiver_data_addr: int
    ig_g1_receiver_data_addr: int
    ig_g2_receiver_data_addr: int

    # NOC coordinates
    gather_dest_noc_core: Any
    mcast_dest_noc_start: Any
    mcast_dest_noc_end: Any

    # Input tensors (needed for CB descriptors)
    group1_src_tensor: Any
    group2_src_tensor: Any
    group1_dst_tensor: Any
    group2_dst_tensor: Any
    weights_tensor: Any
    add_input_tensor: Any
    output_tensor: Any


class GatedLocalReduceDownProjOp:
    """
    Fused Input Gather + GatedLocalReduce + DownProj operation.

    Computes: SiLU(sum(group1)) * sum(group2) @ weights
    where group1/group2 inputs are gathered from remote source cores.
    """

    @staticmethod
    def golden(group1_inputs, group2_inputs, weights, add_input):
        """
        PyTorch reference implementation.

        Args:
            group1_inputs: List of E tensors, each [1, K]
            group2_inputs: List of E tensors, each [1, K]
            weights: [K, N] tensor
            add_input: [1, N] tensor (bias)

        Returns:
            [1, N] tensor
        """
        g1_sum = group1_inputs[0]
        for inp in group1_inputs[1:]:
            g1_sum = g1_sum + inp
        g1_silu = F.silu(g1_sum)

        g2_sum = group2_inputs[0]
        for inp in group2_inputs[1:]:
            g2_sum = g2_sum + inp

        hidden = g1_silu * g2_sum
        return hidden @ weights + add_input

    @staticmethod
    def _setup_dimensions(
        group1_src_tensor,
        group2_src_tensor,
        group1_dst_tensor,
        group2_dst_tensor,
        weights_tensor,
        add_input_tensor,
        output_tensor,
        tiles_per_k,
        k_num_tiles,
        use_face_view,
    ):
        """Compute all dimensions, grids, face-view params, CB indices, semaphores, and addresses."""
        device = group1_src_tensor.device()
        data_format = group1_src_tensor.dtype

        # Tile definitions
        TILE_1x32 = ttnn.Tile((1, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)
        tile_1x32_descriptor = ttnn.TileDescriptor(TILE_1x32)

        # Core grid configuration
        mcast_gather_core = DownProj.MCAST_GATHER_CORE
        mcast_gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_gather_core, mcast_gather_core)])

        mcast_grid = ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(DownProj.MCAST_GRID_X - 1, DownProj.MCAST_GRID_Y - 1),
        )
        mcast_grid_set = ttnn.CoreRangeSet([mcast_grid])
        num_mcast_cores = DownProj.MCAST_GRID_X * DownProj.MCAST_GRID_Y  # 130

        matmul_core_grid = DownProj.build_matmul_core_grid()
        mcast_receiver_grid = DownProj.build_mcast_receiver_grid()
        all_cores = mcast_grid_set

        # NOC coordinates
        gather_dest_noc_core = device.worker_core_from_logical_core(mcast_gather_core)
        mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
        mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)

        # Input gather source core grids
        g1_source_grid = group1_src_tensor.memory_config().shard_spec.grid
        g2_source_grid = group2_src_tensor.memory_config().shard_spec.grid
        g1_source_cores = ttnn.corerange_to_cores(g1_source_grid)
        g2_source_cores = ttnn.corerange_to_cores(g2_source_grid)
        num_g1_sources = len(g1_source_cores)
        num_g2_sources = len(g2_source_cores)

        # Dimension parameters
        input_tile = group1_src_tensor.get_tile()
        tile_h, tile_w = input_tile.tile_shape
        input_tile_size = input_tile.get_tile_size(data_format)

        weights_shard_spec = weights_tensor.memory_config().shard_spec
        n_per_core = weights_shard_spec.shape[1]
        out_w_per_core = n_per_core // TILE_1x32.tile_shape[1]

        # Face-view optimization
        if use_face_view is None:
            use_face_view = can_use_face_view(tile_h, tile_w, tiles_per_k, k_num_tiles)

        if use_face_view:
            face_tile = ttnn.Tile([FACE_HEIGHT, FACE_WIDTH])
            face_tile_desc = ttnn.TileDescriptor(FACE_HEIGHT, FACE_WIDTH, False)
            face_tile_size = face_tile.get_tile_size(data_format)
            kernel_tiles_per_k = tiles_per_k
            kernel_k_num_tiles = 1
            mcast_src_num_pages = 1
            reduce_tile_size = face_tile_size
        else:
            face_tile_desc = None
            face_tile_size = None
            kernel_tiles_per_k = tiles_per_k
            kernel_k_num_tiles = k_num_tiles
            mcast_src_num_pages = k_num_tiles
            reduce_tile_size = input_tile_size

        # CB indices
        group1_cb = 0
        group2_cb = 1
        intermed_cb = 2
        mcast_src_cb = 3
        mcast_dst_cb = 4
        matmul_in1_cb = 5
        matmul_out_cb = 6
        gather_dst_cb = 7
        input_src_g1_cb = 8
        input_src_g2_cb = 9
        residual_add_mcast_src_cb = 10
        residual_add_mcast_dst_cb = 11
        residual_add_out_cb = 12

        # Derived sizes
        mcast_data_size_bytes = k_num_tiles * input_tile_size
        mcast_is_part_of_receiver_grid = mcast_grid.contains(mcast_gather_core)
        total_residual_add_tiles = DownProj.NUM_MATMUL_CORES * out_w_per_core
        residual_add_mcast_data_size_bytes = total_residual_add_tiles * tile_1x32_size
        gather_data_size_bytes = out_w_per_core * tile_1x32_size
        gather_src_num_pages = out_w_per_core
        gather_dst_num_pages = DownProj.NUM_MATMUL_CORES * out_w_per_core
        gather_noc0_num_senders = DownProj.NUM_MATMUL_CORES
        gather_noc1_num_senders = 0
        ig_data_size_bytes = input_tile_size
        total_input_tiles = tiles_per_k * k_num_tiles

        # Semaphore IDs
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1
        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3
        ig_g1_receiver_semaphore_id = 4
        ig_g2_receiver_semaphore_id = 5
        ig_g1_noc1_receiver_semaphore_id = 6
        ig_g2_noc1_receiver_semaphore_id = 7
        mcast2_data_receiver_semaphore_id = 8
        num_semaphores = 9

        # Buffer addresses
        gather_receiver_data_addr = output_tensor.buffer_address()
        ig_g1_receiver_data_addr = group1_dst_tensor.buffer_address()
        ig_g2_receiver_data_addr = group2_dst_tensor.buffer_address()

        # Pre-compute full device grid so we don't need to store device reference
        full_device_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        device.compute_with_storage_grid_size().x - 1,
                        device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            ]
        )

        return _GatedReduceDownProjContext(
            full_device_grid=full_device_grid,
            data_format=data_format,
            input_tile=input_tile,
            input_tile_size=input_tile_size,
            tile_1x32_size=tile_1x32_size,
            tile_1x32_descriptor=tile_1x32_descriptor,
            mcast_gather_core=mcast_gather_core,
            mcast_gather_core_grid=mcast_gather_core_grid,
            mcast_grid=mcast_grid,
            all_cores=all_cores,
            matmul_core_grid=matmul_core_grid,
            mcast_receiver_grid=mcast_receiver_grid,
            g1_source_grid=g1_source_grid,
            g2_source_grid=g2_source_grid,
            g1_source_cores=g1_source_cores,
            g2_source_cores=g2_source_cores,
            num_g1_sources=num_g1_sources,
            num_g2_sources=num_g2_sources,
            num_mcast_cores=num_mcast_cores,
            out_w_per_core=out_w_per_core,
            tiles_per_k=tiles_per_k,
            k_num_tiles=k_num_tiles,
            use_face_view=use_face_view,
            face_tile_desc=face_tile_desc,
            face_tile_size=face_tile_size,
            kernel_tiles_per_k=kernel_tiles_per_k,
            kernel_k_num_tiles=kernel_k_num_tiles,
            mcast_src_num_pages=mcast_src_num_pages,
            reduce_tile_size=reduce_tile_size,
            group1_cb=group1_cb,
            group2_cb=group2_cb,
            intermed_cb=intermed_cb,
            mcast_src_cb=mcast_src_cb,
            mcast_dst_cb=mcast_dst_cb,
            matmul_in1_cb=matmul_in1_cb,
            matmul_out_cb=matmul_out_cb,
            gather_dst_cb=gather_dst_cb,
            input_src_g1_cb=input_src_g1_cb,
            input_src_g2_cb=input_src_g2_cb,
            residual_add_mcast_src_cb=residual_add_mcast_src_cb,
            residual_add_mcast_dst_cb=residual_add_mcast_dst_cb,
            residual_add_out_cb=residual_add_out_cb,
            mcast_data_size_bytes=mcast_data_size_bytes,
            mcast_is_part_of_receiver_grid=mcast_is_part_of_receiver_grid,
            total_residual_add_tiles=total_residual_add_tiles,
            residual_add_mcast_data_size_bytes=residual_add_mcast_data_size_bytes,
            gather_data_size_bytes=gather_data_size_bytes,
            gather_src_num_pages=gather_src_num_pages,
            gather_dst_num_pages=gather_dst_num_pages,
            gather_noc0_num_senders=gather_noc0_num_senders,
            gather_noc1_num_senders=gather_noc1_num_senders,
            ig_data_size_bytes=ig_data_size_bytes,
            total_input_tiles=total_input_tiles,
            mcast_data_sender_semaphore_id=mcast_data_sender_semaphore_id,
            mcast_data_receiver_semaphore_id=mcast_data_receiver_semaphore_id,
            gather_noc0_receiver_semaphore_id=gather_noc0_receiver_semaphore_id,
            gather_noc1_receiver_semaphore_id=gather_noc1_receiver_semaphore_id,
            ig_g1_receiver_semaphore_id=ig_g1_receiver_semaphore_id,
            ig_g2_receiver_semaphore_id=ig_g2_receiver_semaphore_id,
            ig_g1_noc1_receiver_semaphore_id=ig_g1_noc1_receiver_semaphore_id,
            ig_g2_noc1_receiver_semaphore_id=ig_g2_noc1_receiver_semaphore_id,
            mcast2_data_sender_semaphore_id=mcast_data_sender_semaphore_id,
            mcast2_data_receiver_semaphore_id=mcast2_data_receiver_semaphore_id,
            num_semaphores=num_semaphores,
            gather_receiver_data_addr=gather_receiver_data_addr,
            ig_g1_receiver_data_addr=ig_g1_receiver_data_addr,
            ig_g2_receiver_data_addr=ig_g2_receiver_data_addr,
            gather_dest_noc_core=gather_dest_noc_core,
            mcast_dest_noc_start=mcast_dest_noc_start,
            mcast_dest_noc_end=mcast_dest_noc_end,
            group1_src_tensor=group1_src_tensor,
            group2_src_tensor=group2_src_tensor,
            group1_dst_tensor=group1_dst_tensor,
            group2_dst_tensor=group2_dst_tensor,
            weights_tensor=weights_tensor,
            add_input_tensor=add_input_tensor,
            output_tensor=output_tensor,
        )

    @staticmethod
    def _build_compile_time_args(ctx):
        """Build NCRISC, BRISC, and TRISC compile-time arg lists."""
        ncrisc_named_compile_time_args = [
            # Input gather g1 sender
            ("ig_g1_dest_noc_x", ctx.gather_dest_noc_core.x),
            ("ig_g1_dest_noc_y", ctx.gather_dest_noc_core.y),
            ("ig_g1_data_size_bytes", ctx.ig_data_size_bytes),
            ("ig_g1_receiver_semaphore_id", ctx.ig_g1_receiver_semaphore_id),
            ("ig_g1_src_cb", ctx.input_src_g1_cb),
            ("ig_g1_src_num_pages", 1),
            ("ig_g1_receiver_data_addr", ctx.ig_g1_receiver_data_addr),
            # Input gather g2 sender
            ("ig_g2_dest_noc_x", ctx.gather_dest_noc_core.x),
            ("ig_g2_dest_noc_y", ctx.gather_dest_noc_core.y),
            ("ig_g2_data_size_bytes", ctx.ig_data_size_bytes),
            ("ig_g2_receiver_semaphore_id", ctx.ig_g2_receiver_semaphore_id),
            ("ig_g2_src_cb", ctx.input_src_g2_cb),
            ("ig_g2_src_num_pages", 1),
            ("ig_g2_receiver_data_addr", ctx.ig_g2_receiver_data_addr),
            # Mcast1 receiver
            ("mcast_data_receiver_semaphore", ctx.mcast_data_receiver_semaphore_id),
            ("mcast_dst_cb", ctx.mcast_dst_cb),
            ("mcast_dst_num_pages", ctx.k_num_tiles),
            # Mcast2 source (for setup_sharded_buffer on sender core)
            ("mcast2_src_cb", ctx.residual_add_mcast_src_cb),
            ("mcast2_src_num_pages", ctx.total_residual_add_tiles),
            # Mcast2 receiver
            ("mcast2_data_receiver_semaphore", ctx.mcast2_data_receiver_semaphore_id),
            ("mcast2_dst_cb", ctx.residual_add_mcast_dst_cb),
            ("mcast2_dst_num_pages", ctx.total_residual_add_tiles),
            # Matmul
            ("matmul_in0", ctx.mcast_dst_cb),
            ("matmul_in1", ctx.matmul_in1_cb),
            ("matmul_out", ctx.matmul_out_cb),
            ("matmul_k_num_tiles", ctx.k_num_tiles),
            ("matmul_out_w_per_core", ctx.out_w_per_core),
            # Residual add (needed for ResidualAdd CTArgs template parameter)
            ("residual_add_out_w", ctx.out_w_per_core),
            # Output gather sender
            ("gather_dest_noc_x", ctx.gather_dest_noc_core.x),
            ("gather_dest_noc_y", ctx.gather_dest_noc_core.y),
            ("gather_data_size_bytes", ctx.gather_data_size_bytes),
            ("gather_receiver_semaphore_id", ctx.gather_noc0_receiver_semaphore_id),
            ("gather_src_cb", ctx.residual_add_out_cb),
            ("gather_src_num_pages", ctx.gather_src_num_pages),
            ("gather_sender_grid_start_x", 0),
            ("gather_sender_grid_start_y", 0),
            ("gather_sender_grid_end_x", 0),
            ("gather_sender_grid_end_y", 0),
            ("gather_row_major", 1),
            ("gather_receiver_data_addr", ctx.gather_receiver_data_addr),
        ]

        brisc_named_compile_time_args = [
            # Input gather g1 receiver
            ("ig_g1_noc0_num_senders", ctx.num_g1_sources),
            ("ig_g1_noc0_receiver_semaphore_id", ctx.ig_g1_receiver_semaphore_id),
            ("ig_g1_noc1_receiver_semaphore_id", ctx.ig_g1_noc1_receiver_semaphore_id),
            ("ig_g1_dst_cb", ctx.group1_cb),
            ("ig_g1_dst_num_pages", ctx.kernel_tiles_per_k if ctx.use_face_view else ctx.total_input_tiles),
            # Input gather g2 receiver
            ("ig_g2_noc0_num_senders", ctx.num_g2_sources),
            ("ig_g2_noc0_receiver_semaphore_id", ctx.ig_g2_receiver_semaphore_id),
            ("ig_g2_noc1_receiver_semaphore_id", ctx.ig_g2_noc1_receiver_semaphore_id),
            ("ig_g2_dst_cb", ctx.group2_cb),
            ("ig_g2_dst_num_pages", ctx.kernel_tiles_per_k if ctx.use_face_view else ctx.total_input_tiles),
            # Mcast sender
            ("mcast_dest_noc_start_x", ctx.mcast_dest_noc_start.x),
            ("mcast_dest_noc_start_y", ctx.mcast_dest_noc_start.y),
            ("mcast_dest_noc_end_x", ctx.mcast_dest_noc_end.x),
            ("mcast_dest_noc_end_y", ctx.mcast_dest_noc_end.y),
            ("mcast_num_cores", ctx.num_mcast_cores),
            ("mcast_data_sender_semaphore", ctx.mcast_data_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", ctx.mcast_data_receiver_semaphore_id),
            ("mcast_data_size_bytes", ctx.mcast_data_size_bytes),
            ("mcast_src_cb", ctx.mcast_src_cb),
            ("mcast_src_num_pages", ctx.mcast_src_num_pages),
            ("mcast_dst_cb", ctx.mcast_dst_cb),
            ("mcast_is_part_of_receiver_grid", ctx.mcast_is_part_of_receiver_grid),
            # Mcast2 sender
            ("mcast2_data_sender_semaphore", ctx.mcast2_data_sender_semaphore_id),
            ("mcast2_data_receiver_semaphore", ctx.mcast2_data_receiver_semaphore_id),
            ("mcast2_data_size_bytes", ctx.residual_add_mcast_data_size_bytes),
            ("mcast2_src_cb", ctx.residual_add_mcast_src_cb),
            ("mcast2_src_num_pages", ctx.total_residual_add_tiles),
            ("mcast2_dst_cb", ctx.residual_add_mcast_dst_cb),
            # Residual add (needed for ResidualAdd CTArgs template parameter)
            ("residual_add_out_w", ctx.out_w_per_core),
            # Output gather receiver
            ("gather_noc0_num_senders", ctx.gather_noc0_num_senders),
            ("gather_noc1_num_senders", ctx.gather_noc1_num_senders),
            ("gather_noc0_receiver_semaphore_id", ctx.gather_noc0_receiver_semaphore_id),
            ("gather_noc1_receiver_semaphore_id", ctx.gather_noc1_receiver_semaphore_id),
            ("gather_dst_cb", ctx.gather_dst_cb),
            ("gather_dst_num_pages", ctx.gather_dst_num_pages),
        ]

        trisc_named_compile_time_args = [
            # Gated reduce
            ("gated_reduce_group1_cb", ctx.group1_cb),
            ("gated_reduce_group2_cb", ctx.group2_cb),
            ("gated_reduce_intermed_cb", ctx.intermed_cb),
            ("gated_reduce_mcast_src_cb", ctx.mcast_src_cb),
            ("gated_reduce_tiles_per_k", ctx.kernel_tiles_per_k),
            ("gated_reduce_k_num_tiles", ctx.kernel_k_num_tiles),
            # Matmul
            ("matmul_in0", ctx.mcast_dst_cb),
            ("matmul_in1", ctx.matmul_in1_cb),
            ("matmul_out", ctx.matmul_out_cb),
            ("matmul_k_num_tiles", ctx.k_num_tiles),
            ("matmul_out_w_per_core", ctx.out_w_per_core),
            # Residual add step
            ("residual_add_in0", ctx.matmul_out_cb),
            ("residual_add_in1", ctx.residual_add_mcast_dst_cb),
            ("residual_add_out", ctx.residual_add_out_cb),
            ("residual_add_out_w", ctx.out_w_per_core),
            ("residual_add_total_in1_tiles", ctx.total_residual_add_tiles),
        ]

        return ncrisc_named_compile_time_args, brisc_named_compile_time_args, trisc_named_compile_time_args

    @staticmethod
    def _build_cb_descriptors(ctx):
        """Build all 13 circular buffer descriptors."""
        # CB 0: Group 1 gather dest — tensor-backed on sender core
        group1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ctx.group1_cb, ctx.group1_dst_tensor)

        # CB 1: Group 2 gather dest — tensor-backed on sender core
        group2_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ctx.group2_cb, ctx.group2_dst_tensor)

        # Face-view: override CB 0, 1 tile descriptors to [16,16] face tiles
        if ctx.use_face_view:
            group1_cb_descriptor.format_descriptors[0].tile = ctx.face_tile_desc
            group1_cb_descriptor.format_descriptors[0].page_size = ctx.face_tile_size
            group2_cb_descriptor.format_descriptors[0].tile = ctx.face_tile_desc
            group2_cb_descriptor.format_descriptors[0].page_size = ctx.face_tile_size

        tile_desc = ctx.face_tile_desc if ctx.use_face_view else ttnn.TileDescriptor(ctx.input_tile)

        # CB 2: Intermediate — 2 tiles on sender core, manual
        intermed_format = ttnn.CBFormatDescriptor(
            buffer_index=ctx.intermed_cb,
            data_format=ctx.data_format,
            page_size=ctx.reduce_tile_size,
            tile=tile_desc,
        )
        intermed_cb_descriptor = ttnn.CBDescriptor(
            total_size=2 * ctx.reduce_tile_size,
            core_ranges=ctx.mcast_gather_core_grid,
            format_descriptors=[intermed_format],
        )

        # CB 3: Mcast source — on sender core, manual
        mcast_src_format = ttnn.CBFormatDescriptor(
            buffer_index=ctx.mcast_src_cb,
            data_format=ctx.data_format,
            page_size=ctx.reduce_tile_size,
            tile=tile_desc,
        )
        mcast_src_cb_descriptor = ttnn.CBDescriptor(
            total_size=ctx.mcast_src_num_pages * ctx.reduce_tile_size,
            core_ranges=ctx.mcast_gather_core_grid,
            format_descriptors=[mcast_src_format],
        )

        input_tile_desc = ttnn.TileDescriptor(ctx.input_tile)

        # CB 4: Mcast destination / matmul in0 — all 130 cores
        mcast_dst_format = ttnn.CBFormatDescriptor(
            buffer_index=ctx.mcast_dst_cb,
            data_format=ctx.data_format,
            page_size=ctx.input_tile_size,
            tile=input_tile_desc,
        )
        mcast_dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=ctx.k_num_tiles * ctx.input_tile_size,
            core_ranges=ctx.all_cores,
            format_descriptors=[mcast_dst_format],
        )

        # CB 5: Matmul weights — tensor-backed on 112 matmul cores
        matmul_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ctx.matmul_in1_cb, ctx.weights_tensor)

        # CB 6: Matmul output — on 112 matmul cores
        matmul_out_format = ttnn.CBFormatDescriptor(
            buffer_index=ctx.matmul_out_cb,
            data_format=ctx.data_format,
            page_size=ctx.tile_1x32_size,
            tile=ctx.tile_1x32_descriptor,
        )
        matmul_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=ctx.out_w_per_core * ctx.tile_1x32_size,
            core_ranges=ctx.matmul_core_grid,
            format_descriptors=[matmul_out_format],
        )

        # CB 7: Output gather destination — tensor-backed on sender core
        gather_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ctx.gather_dst_cb, ctx.output_tensor)

        # CB 8: Input source group1 — tensor-backed on g1 source cores
        input_src_g1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ctx.input_src_g1_cb, ctx.group1_src_tensor)

        # CB 9: Input source group2 — tensor-backed on g2 source cores
        input_src_g2_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ctx.input_src_g2_cb, ctx.group2_src_tensor)

        # CB 10: Mcast2 source — residual input on (12,9), tensor-backed
        residual_add_mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            ctx.residual_add_mcast_src_cb, ctx.add_input_tensor
        )

        # CB 11: Mcast2 destination — on all 130 cores
        residual_add_mcast_dst_format = ttnn.CBFormatDescriptor(
            buffer_index=ctx.residual_add_mcast_dst_cb,
            data_format=ctx.data_format,
            page_size=ctx.tile_1x32_size,
            tile=ctx.tile_1x32_descriptor,
        )
        residual_add_mcast_dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=ctx.total_residual_add_tiles * ctx.tile_1x32_size,
            core_ranges=ctx.all_cores,
            format_descriptors=[residual_add_mcast_dst_format],
        )

        # CB 12: Residual add output — on 112 matmul cores
        residual_add_out_format = ttnn.CBFormatDescriptor(
            buffer_index=ctx.residual_add_out_cb,
            data_format=ctx.data_format,
            page_size=ctx.tile_1x32_size,
            tile=ctx.tile_1x32_descriptor,
        )
        residual_add_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=ctx.out_w_per_core * ctx.tile_1x32_size,
            core_ranges=ctx.matmul_core_grid,
            format_descriptors=[residual_add_out_format],
        )

        return [
            group1_cb_descriptor,
            group2_cb_descriptor,
            intermed_cb_descriptor,
            mcast_src_cb_descriptor,
            mcast_dst_cb_descriptor,
            matmul_in1_cb_descriptor,
            matmul_out_cb_descriptor,
            gather_dst_cb_descriptor,
            input_src_g1_cb_descriptor,
            input_src_g2_cb_descriptor,
            residual_add_mcast_src_cb_descriptor,
            residual_add_mcast_dst_cb_descriptor,
            residual_add_out_cb_descriptor,
        ]

    @staticmethod
    def _build_core_descriptors(ctx):
        """Build per-core and unified compile-time core descriptors."""
        # Per-core: output gather sender index
        matmul_cores_list = ttnn.corerange_to_cores(ctx.matmul_core_grid)
        per_core_gather_idx = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="gather_sender_idx",
            core_values=[(core, idx) for idx, core in enumerate(matmul_cores_list)],
            other_value=0,
        )

        # Per-core: input gather sender indices
        def compute_sender_indices(source_cores):
            indices = []
            for i, core in enumerate(source_cores):
                collection = i // ctx.k_num_tiles
                k = i % ctx.k_num_tiles
                if ctx.use_face_view:
                    sender_idx = collection * ctx.k_num_tiles + k
                else:
                    sender_idx = k * ctx.tiles_per_k + collection
                indices.append((core, sender_idx))
            return indices

        per_core_ig_g1_idx = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="ig_g1_sender_idx",
            core_values=compute_sender_indices(ctx.g1_source_cores),
            other_value=0,
        )
        per_core_ig_g2_idx = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="ig_g2_sender_idx",
            core_values=compute_sender_indices(ctx.g2_source_cores),
            other_value=0,
        )

        # Unified core descriptors (role flags)
        core_descs = [
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_input_gather_sender_g1",
                core_range=ctx.g1_source_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_input_gather_sender_g2",
                core_range=ctx.g2_source_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_gated_reduce_core",
                core_range=ctx.mcast_gather_core_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_mcast_sender_core",
                core_range=ctx.mcast_gather_core_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_mcast_receiver_core",
                core_range=ctx.mcast_receiver_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_matmul_core",
                core_range=ctx.matmul_core_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_gather_receiver_core",
                core_range=ctx.mcast_gather_core_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="gather_use_per_core_sender_idx",
                core_range=ctx.matmul_core_grid,
                value=1,
                other_value=0,
            ),
        ]

        per_core_descs = [per_core_gather_idx, per_core_ig_g1_idx, per_core_ig_g2_idx]

        return core_descs, per_core_descs

    @staticmethod
    def op(
        group1_src_tensor,
        group2_src_tensor,
        group1_dst_tensor,
        group2_dst_tensor,
        weights_tensor,
        add_input_tensor,
        output_tensor,
        tiles_per_k,
        k_num_tiles,
        use_face_view=None,
    ):
        """
        Execute fused input gather + gated local reduce + down projection.

        Args:
            group1_src_tensor: Source data for group1, HEIGHT_SHARDED on g1 source cores,
                               1 tile [1,32] per core
            group2_src_tensor: Source data for group2, same layout on g2 source cores
            group1_dst_tensor: Gather destination for group1, HEIGHT_SHARDED on (12,9),
                               pre-allocated (filled by input gather)
            group2_dst_tensor: Gather destination for group2, same layout on (12,9)
            weights_tensor: Weights [K, N] WIDTH_SHARDED on 112 matmul cores
            output_tensor: Output [1, N] HEIGHT_SHARDED on (12,9)
            tiles_per_k: Number of collections to reduce per K-position
            k_num_tiles: Number of K-tile positions (K / 32)

        Returns:
            Output tensor with result
        """
        ctx = GatedLocalReduceDownProjOp._setup_dimensions(
            group1_src_tensor,
            group2_src_tensor,
            group1_dst_tensor,
            group2_dst_tensor,
            weights_tensor,
            add_input_tensor,
            output_tensor,
            tiles_per_k,
            k_num_tiles,
            use_face_view,
        )

        ncrisc_args, brisc_args, trisc_args = GatedLocalReduceDownProjOp._build_compile_time_args(ctx)
        cb_descriptors = GatedLocalReduceDownProjOp._build_cb_descriptors(ctx)
        core_descs, per_core_descs = GatedLocalReduceDownProjOp._build_core_descriptors(ctx)

        # Semaphore descriptors
        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=ctx.full_device_grid, initial_value=0)
            for i in range(ctx.num_semaphores)
        ]

        # Kernel descriptor
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/fused_ops/gated_local_reduce_down_proj/kernels/gated_local_reduce_down_proj_kernel.cpp",
            core_ranges=ctx.all_cores,
            ncrisc_named_compile_time_args=ncrisc_args,
            brisc_named_compile_time_args=brisc_args,
            trisc_named_compile_time_args=trisc_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
            unified_compile_time_core_descriptors=core_descs,
            per_core_compile_time_descriptors=per_core_descs,
        )

        # Program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=cb_descriptors,
            semaphores=semaphore_descriptors,
        )

        # Execute generic op
        io_tensors = [
            group1_src_tensor,
            group2_src_tensor,
            group1_dst_tensor,
            group2_dst_tensor,
            weights_tensor,
            add_input_tensor,
            output_tensor,
        ]
        return ttnn.generic_op(io_tensors, program_descriptor)
