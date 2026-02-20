# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Shared expert operation.

Fuses: Activation Mcast + Gate/Up Matmul + Gather + GatedLocalReduce
       + Mcast1 + Mcast2 + Down Proj Matmul + ResidualAdd + Output Gather

Pipeline:
  1. Activation mcast: [1, K_gate] from (12,9) → all 130 cores
  2. Gate matmul on 64 A cores, Up matmul on 64 B cores
  3. Gather A→CB0, B→CB1 on (12,9)
  4. Gated reduce: SiLU(sum(A)) * sum(B) → [1, K_down]
  5. Mcast1: [1, K_down] from (12,9) → all 130 cores
  6. Mcast2: bias [1, N] from (12,9) → all 130 cores
  7. Down proj matmul on 112 cores
  8. Residual add on 112 cores
  9. Output gather → (12,9)

CB Layout (15 CBs: 0-14):
  CB 0:  A_gather_dst / reduce_g1 (sender, manual, face-view)
  CB 1:  B_gather_dst / reduce_g2 (sender, manual, face-view)
  CB 2:  intermediate (sender, 2 tiles, manual)
  CB 3:  mcast1 source (sender, manual, TRISC-filled)
  CB 4:  mcast1 dest / down_matmul_in0 (all 130 cores, manual)
  CB 5:  down_weights (112 matmul cores, tensor-backed)
  CB 6:  down_matmul_out (112 matmul cores, manual)
  CB 7:  output_gather_dst (sender, tensor-backed)
  CB 8:  act_mcast_src (sender, tensor-backed)
  CB 9:  act_mcast_recv (all 130 cores, manual)
  CB 10: residual_mcast_src (sender, tensor-backed)
  CB 11: residual_mcast_dst (all 130 cores, manual)
  CB 12: residual_add_out (112 matmul cores, manual)
  CB 13: gate_up_weights (128 compute cores, tensor-backed)
  CB 14: gate_up_matmul_out (128 compute cores, 1 tile, manual)
"""

from dataclasses import dataclass
from typing import Any

import torch
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
class _SharedExpertContext:
    """Holds all computed values needed by SharedExpertOp helper methods."""

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
    a_cores_list: list
    b_cores_list: list
    compute_cores_list: list
    a_compute_grid: Any
    b_compute_grid: Any
    num_mcast_cores: int

    # Dimensions
    K_gate_tiles: int
    k_per_core: int
    K_down_tiles: int
    out_w_per_core: int
    tiles_per_k: int
    k_num_tiles: int
    n_parallel: int
    k_parallel: int
    num_compute_cores: int

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
    act_mcast_src_cb: int
    act_mcast_dst_cb: int
    residual_mcast_src_cb: int
    residual_mcast_dst_cb: int
    residual_add_out_cb: int
    gu_weights_cb: int
    gu_out_cb: int

    # Derived sizes
    act_mcast_data_size_bytes: int
    act_mcast_is_part_of_receiver_grid: bool
    mcast_data_size_bytes: int
    mcast_is_part_of_receiver_grid: bool
    total_residual_add_tiles: int
    residual_mcast_data_size_bytes: int
    gather_data_size_bytes: int
    gather_src_num_pages: int
    gather_dst_num_pages: int
    gather_noc0_num_senders: int
    gather_noc1_num_senders: int
    gu_gather_data_size_bytes: int
    total_gather_tiles: int

    # Semaphore IDs
    mcast_data_sender_semaphore_id: int
    mcast_data_receiver_semaphore_id: int
    gather_noc0_receiver_semaphore_id: int
    gather_noc1_receiver_semaphore_id: int
    ag_receiver_semaphore_id: int
    bg_receiver_semaphore_id: int
    ag_noc1_receiver_semaphore_id: int
    bg_noc1_receiver_semaphore_id: int
    act_mcast_sender_semaphore_id: int
    act_mcast_receiver_semaphore_id: int
    num_semaphores: int

    # Addresses & tensors
    gather_receiver_data_addr: int
    ag_dummy_tensor: Any
    bg_dummy_tensor: Any
    ag_receiver_data_addr: int
    bg_receiver_data_addr: int

    # NOC coordinates
    gather_dest_noc_core: Any
    mcast_dest_noc_start: Any
    mcast_dest_noc_end: Any


class SharedExpertOp:
    """
    Shared expert operation.

    Computes: (SiLU(activation @ gate_weights) * (activation @ up_weights)) @ down_weights + bias
    """

    @staticmethod
    def build_ab_grids():
        """
        Build A (gate) / B (up) compute core grids for 64+64 dual matmul layout.

        Layout on 13x10 grid:
            Rows 0-3: A=cols{0-3,7-9}, B=cols{4-6,10-12}
            Rows 4-7: A=cols{0-2,7-9}, B=cols{3-6,10-12}
            Row 8:    A=cols{0-2,7-9}, B=cols{3-6,10-11}  (col 12 = idle phantom)
            Row 9:    A=cols{0-2,7-9}, B=cols{3-6,10-11}  (col 12 = sender M)

        Returns:
            (a_cores, b_cores): Lists of ttnn.CoreCoord, 64 each
        """
        a_cores = []
        b_cores = []
        for row in range(10):
            for col in range(13):
                if col == 12 and row == 9:  # sender core (12,9)
                    continue
                if col == 12 and row == 8:  # idle phantom
                    continue
                if row < 4:
                    is_a = col in {0, 1, 2, 3, 7, 8, 9}
                else:
                    is_a = col in {0, 1, 2, 7, 8, 9}
                if is_a:
                    a_cores.append(ttnn.CoreCoord(col, row))
                else:
                    b_cores.append(ttnn.CoreCoord(col, row))
        assert len(a_cores) == 64, f"Expected 64 A cores, got {len(a_cores)}"
        assert len(b_cores) == 64, f"Expected 64 B cores, got {len(b_cores)}"
        return a_cores, b_cores

    @staticmethod
    def golden(activation, gate_weights, up_weights, down_weights, bias):
        """
        PyTorch reference implementation.

        Args:
            activation: [1, K_gate] tensor
            gate_weights: [K_gate, K_down] tensor
            up_weights: [K_gate, K_down] tensor
            down_weights: [K_down, N] tensor
            bias: [1, N] tensor

        Returns:
            [1, N] tensor
        """
        gate_out = F.silu(activation @ gate_weights)
        up_out = activation @ up_weights
        hidden = gate_out * up_out
        return hidden @ down_weights + bias

    @staticmethod
    def _setup_dimensions(
        activation_tensor,
        gate_up_weights_tensor,
        down_weights_tensor,
        bias_tensor,
        output_tensor,
        k_parallel,
        n_parallel,
    ):
        """Compute all dimensions, grids, face-view params, CB indices, semaphores, and addresses."""
        device = activation_tensor.device()
        data_format = activation_tensor.dtype

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

        # A/B compute grids
        a_cores_list, b_cores_list = SharedExpertOp.build_ab_grids()
        a_compute_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in a_cores_list])
        b_compute_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in b_cores_list])
        compute_cores_list = a_cores_list + b_cores_list  # 128 cores

        mcast_receiver_grid = DownProj.build_mcast_receiver_grid()
        all_cores = mcast_grid_set

        # NOC coordinates
        gather_dest_noc_core = device.worker_core_from_logical_core(mcast_gather_core)
        mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
        mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)

        # Dimension parameters
        input_tile = activation_tensor.get_tile()
        tile_h, tile_w = input_tile.tile_shape
        input_tile_size = input_tile.get_tile_size(data_format)

        K_gate = activation_tensor.shape[1]
        K_gate_tiles = K_gate // tile_w

        num_compute_cores = 64
        assert (
            K_gate_tiles % k_parallel == 0
        ), f"K_gate_tiles ({K_gate_tiles}) must be divisible by k_parallel ({k_parallel})"
        assert k_parallel * n_parallel == num_compute_cores, f"k_parallel * n_parallel must equal {num_compute_cores}"
        k_per_core = K_gate_tiles // k_parallel
        K_down_tiles = n_parallel

        # Down proj dimensions
        down_weights_shard_spec = down_weights_tensor.memory_config().shard_spec
        n_per_core = down_weights_shard_spec.shape[1]
        out_w_per_core = n_per_core // TILE_1x32.tile_shape[1]

        # Face-view parameters
        tiles_per_k = k_parallel
        k_num_tiles = n_parallel

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
        act_mcast_src_cb = 8
        act_mcast_dst_cb = 9
        residual_mcast_src_cb = 10
        residual_mcast_dst_cb = 11
        residual_add_out_cb = 12
        gu_weights_cb = 13
        gu_out_cb = 14

        # Derived sizes
        act_mcast_data_size_bytes = K_gate_tiles * input_tile_size
        act_mcast_is_part_of_receiver_grid = mcast_grid.contains(mcast_gather_core)
        mcast_data_size_bytes = K_down_tiles * input_tile_size
        mcast_is_part_of_receiver_grid = mcast_grid.contains(mcast_gather_core)
        total_residual_add_tiles = DownProj.NUM_MATMUL_CORES * out_w_per_core
        residual_mcast_data_size_bytes = total_residual_add_tiles * tile_1x32_size
        gather_data_size_bytes = out_w_per_core * tile_1x32_size
        gather_src_num_pages = out_w_per_core
        gather_dst_num_pages = DownProj.NUM_MATMUL_CORES * out_w_per_core
        gather_noc0_num_senders = DownProj.NUM_MATMUL_CORES
        gather_noc1_num_senders = 0
        gu_gather_data_size_bytes = input_tile_size
        total_gather_tiles = num_compute_cores

        # Semaphore IDs
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1
        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3
        ag_receiver_semaphore_id = 4
        bg_receiver_semaphore_id = 5
        ag_noc1_receiver_semaphore_id = 6
        bg_noc1_receiver_semaphore_id = 7
        act_mcast_receiver_semaphore_id = 8
        num_semaphores = 9

        # Buffer addresses
        gather_receiver_data_addr = output_tensor.buffer_address()

        ag_dummy_shape = (total_gather_tiles, 32)
        ag_dummy_shard_spec = ttnn.ShardSpec(mcast_gather_core_grid, ag_dummy_shape, ttnn.ShardOrientation.ROW_MAJOR)
        ag_dummy_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ag_dummy_shard_spec
        )

        ag_dummy_tensor = ttnn.from_torch(
            torch.zeros(total_gather_tiles, 32, dtype=torch.bfloat16),
            dtype=data_format,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ag_dummy_mem,
            tile=input_tile,
        )
        bg_dummy_tensor = ttnn.from_torch(
            torch.zeros(total_gather_tiles, 32, dtype=torch.bfloat16),
            dtype=data_format,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ag_dummy_mem,
            tile=input_tile,
        )

        ag_receiver_data_addr = ag_dummy_tensor.buffer_address()
        bg_receiver_data_addr = bg_dummy_tensor.buffer_address()

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

        return _SharedExpertContext(
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
            a_cores_list=a_cores_list,
            b_cores_list=b_cores_list,
            compute_cores_list=compute_cores_list,
            a_compute_grid=a_compute_grid,
            b_compute_grid=b_compute_grid,
            num_mcast_cores=num_mcast_cores,
            K_gate_tiles=K_gate_tiles,
            k_per_core=k_per_core,
            K_down_tiles=K_down_tiles,
            out_w_per_core=out_w_per_core,
            tiles_per_k=tiles_per_k,
            k_num_tiles=k_num_tiles,
            n_parallel=n_parallel,
            k_parallel=k_parallel,
            num_compute_cores=num_compute_cores,
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
            act_mcast_src_cb=act_mcast_src_cb,
            act_mcast_dst_cb=act_mcast_dst_cb,
            residual_mcast_src_cb=residual_mcast_src_cb,
            residual_mcast_dst_cb=residual_mcast_dst_cb,
            residual_add_out_cb=residual_add_out_cb,
            gu_weights_cb=gu_weights_cb,
            gu_out_cb=gu_out_cb,
            act_mcast_data_size_bytes=act_mcast_data_size_bytes,
            act_mcast_is_part_of_receiver_grid=act_mcast_is_part_of_receiver_grid,
            mcast_data_size_bytes=mcast_data_size_bytes,
            mcast_is_part_of_receiver_grid=mcast_is_part_of_receiver_grid,
            total_residual_add_tiles=total_residual_add_tiles,
            residual_mcast_data_size_bytes=residual_mcast_data_size_bytes,
            gather_data_size_bytes=gather_data_size_bytes,
            gather_src_num_pages=gather_src_num_pages,
            gather_dst_num_pages=gather_dst_num_pages,
            gather_noc0_num_senders=gather_noc0_num_senders,
            gather_noc1_num_senders=gather_noc1_num_senders,
            gu_gather_data_size_bytes=gu_gather_data_size_bytes,
            total_gather_tiles=total_gather_tiles,
            mcast_data_sender_semaphore_id=mcast_data_sender_semaphore_id,
            mcast_data_receiver_semaphore_id=mcast_data_receiver_semaphore_id,
            gather_noc0_receiver_semaphore_id=gather_noc0_receiver_semaphore_id,
            gather_noc1_receiver_semaphore_id=gather_noc1_receiver_semaphore_id,
            ag_receiver_semaphore_id=ag_receiver_semaphore_id,
            bg_receiver_semaphore_id=bg_receiver_semaphore_id,
            ag_noc1_receiver_semaphore_id=ag_noc1_receiver_semaphore_id,
            bg_noc1_receiver_semaphore_id=bg_noc1_receiver_semaphore_id,
            act_mcast_sender_semaphore_id=mcast_data_sender_semaphore_id,
            act_mcast_receiver_semaphore_id=act_mcast_receiver_semaphore_id,
            num_semaphores=num_semaphores,
            gather_receiver_data_addr=gather_receiver_data_addr,
            ag_dummy_tensor=ag_dummy_tensor,
            bg_dummy_tensor=bg_dummy_tensor,
            ag_receiver_data_addr=ag_receiver_data_addr,
            bg_receiver_data_addr=bg_receiver_data_addr,
            gather_dest_noc_core=gather_dest_noc_core,
            mcast_dest_noc_start=mcast_dest_noc_start,
            mcast_dest_noc_end=mcast_dest_noc_end,
        )

    @staticmethod
    def _build_compile_time_args(ctx):
        """Build NCRISC, BRISC, and TRISC compile-time arg lists."""
        ncrisc_named_compile_time_args = [
            # Activation mcast source (setup_sharded_buffer on sender)
            ("act_mcast_src_cb", ctx.act_mcast_src_cb),
            ("act_mcast_src_num_pages", ctx.K_gate_tiles),
            # Activation mcast receiver
            ("act_mcast_data_receiver_semaphore", ctx.act_mcast_receiver_semaphore_id),
            ("act_mcast_dst_cb", ctx.act_mcast_dst_cb),
            ("act_mcast_dst_num_pages", ctx.K_gate_tiles),
            # Gate/Up weights (setup_sharded_buffer on compute cores)
            ("gu_weights_cb", ctx.gu_weights_cb),
            ("gu_weights_num_pages", ctx.k_per_core),
            # Gate gather (A) sender
            ("ag_dest_noc_x", ctx.gather_dest_noc_core.x),
            ("ag_dest_noc_y", ctx.gather_dest_noc_core.y),
            ("ag_data_size_bytes", ctx.gu_gather_data_size_bytes),
            ("ag_receiver_semaphore_id", ctx.ag_receiver_semaphore_id),
            ("ag_src_cb", ctx.gu_out_cb),
            ("ag_src_num_pages", 1),
            ("ag_receiver_data_addr", ctx.ag_receiver_data_addr),
            # Up gather (B) sender
            ("bg_dest_noc_x", ctx.gather_dest_noc_core.x),
            ("bg_dest_noc_y", ctx.gather_dest_noc_core.y),
            ("bg_data_size_bytes", ctx.gu_gather_data_size_bytes),
            ("bg_receiver_semaphore_id", ctx.bg_receiver_semaphore_id),
            ("bg_src_cb", ctx.gu_out_cb),
            ("bg_src_num_pages", 1),
            ("bg_receiver_data_addr", ctx.bg_receiver_data_addr),
            # Mcast1 receiver (down proj activation)
            ("mcast_data_receiver_semaphore", ctx.mcast_data_receiver_semaphore_id),
            ("mcast_dst_cb", ctx.mcast_dst_cb),
            ("mcast_dst_num_pages", ctx.K_down_tiles),
            # Down proj weights (setup_sharded_buffer on matmul cores)
            ("matmul_in1", ctx.matmul_in1_cb),
            ("matmul_k_num_tiles", ctx.K_down_tiles),
            ("matmul_out_w_per_core", ctx.out_w_per_core),
            # Mcast2 source (residual on sender)
            ("mcast2_src_cb", ctx.residual_mcast_src_cb),
            ("mcast2_src_num_pages", ctx.total_residual_add_tiles),
            # Mcast2 receiver (residual, shares semaphores with mcast1)
            ("mcast2_data_receiver_semaphore", ctx.mcast_data_receiver_semaphore_id),
            ("mcast2_dst_cb", ctx.residual_mcast_dst_cb),
            ("mcast2_dst_num_pages", ctx.total_residual_add_tiles),
            # Down proj matmul (NCRISC no-op, but args needed for split kernel)
            ("matmul_in0", ctx.mcast_dst_cb),
            ("matmul_out", ctx.matmul_out_cb),
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
            # Activation mcast sender
            ("act_mcast_dest_noc_start_x", ctx.mcast_dest_noc_start.x),
            ("act_mcast_dest_noc_start_y", ctx.mcast_dest_noc_start.y),
            ("act_mcast_dest_noc_end_x", ctx.mcast_dest_noc_end.x),
            ("act_mcast_dest_noc_end_y", ctx.mcast_dest_noc_end.y),
            ("act_mcast_num_cores", ctx.num_mcast_cores),
            ("act_mcast_data_sender_semaphore", ctx.act_mcast_sender_semaphore_id),
            ("act_mcast_data_receiver_semaphore", ctx.act_mcast_receiver_semaphore_id),
            ("act_mcast_data_size_bytes", ctx.act_mcast_data_size_bytes),
            ("act_mcast_src_cb", ctx.act_mcast_src_cb),
            ("act_mcast_src_num_pages", ctx.K_gate_tiles),
            ("act_mcast_dst_cb", ctx.act_mcast_dst_cb),
            ("act_mcast_is_part_of_receiver_grid", ctx.act_mcast_is_part_of_receiver_grid),
            # Gate gather (A) receiver
            ("ag_noc0_num_senders", ctx.num_compute_cores),
            ("ag_noc0_receiver_semaphore_id", ctx.ag_receiver_semaphore_id),
            ("ag_noc1_receiver_semaphore_id", ctx.ag_noc1_receiver_semaphore_id),
            ("ag_dst_cb", ctx.group1_cb),
            ("ag_dst_num_pages", ctx.kernel_tiles_per_k if ctx.use_face_view else ctx.total_gather_tiles),
            # Up gather (B) receiver
            ("bg_noc0_num_senders", ctx.num_compute_cores),
            ("bg_noc0_receiver_semaphore_id", ctx.bg_receiver_semaphore_id),
            ("bg_noc1_receiver_semaphore_id", ctx.bg_noc1_receiver_semaphore_id),
            ("bg_dst_cb", ctx.group2_cb),
            ("bg_dst_num_pages", ctx.kernel_tiles_per_k if ctx.use_face_view else ctx.total_gather_tiles),
            # Mcast1 sender (down proj activation)
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
            # Mcast2 sender (bias, shares semaphores with mcast1)
            ("mcast2_data_sender_semaphore", ctx.mcast_data_sender_semaphore_id),
            ("mcast2_data_receiver_semaphore", ctx.mcast_data_receiver_semaphore_id),
            ("mcast2_data_size_bytes", ctx.residual_mcast_data_size_bytes),
            ("mcast2_src_cb", ctx.residual_mcast_src_cb),
            ("mcast2_src_num_pages", ctx.total_residual_add_tiles),
            ("mcast2_dst_cb", ctx.residual_mcast_dst_cb),
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
            # Gate/Up matmul
            ("gu_act_cb", ctx.act_mcast_dst_cb),
            ("gu_weights_cb", ctx.gu_weights_cb),
            ("gu_out_cb", ctx.gu_out_cb),
            ("gu_k_per_core", ctx.k_per_core),
            ("gu_act_total_tiles", ctx.K_gate_tiles),
            # Gated reduce
            ("gated_reduce_group1_cb", ctx.group1_cb),
            ("gated_reduce_group2_cb", ctx.group2_cb),
            ("gated_reduce_intermed_cb", ctx.intermed_cb),
            ("gated_reduce_mcast_src_cb", ctx.mcast_src_cb),
            ("gated_reduce_tiles_per_k", ctx.kernel_tiles_per_k),
            ("gated_reduce_k_num_tiles", ctx.kernel_k_num_tiles),
            # Down proj matmul
            ("matmul_in0", ctx.mcast_dst_cb),
            ("matmul_in1", ctx.matmul_in1_cb),
            ("matmul_out", ctx.matmul_out_cb),
            ("matmul_k_num_tiles", ctx.K_down_tiles),
            ("matmul_out_w_per_core", ctx.out_w_per_core),
            # Residual add step
            ("residual_add_in0", ctx.matmul_out_cb),
            ("residual_add_in1", ctx.residual_mcast_dst_cb),
            ("residual_add_out", ctx.residual_add_out_cb),
            ("residual_add_out_w", ctx.out_w_per_core),
            ("residual_add_total_in1_tiles", ctx.total_residual_add_tiles),
        ]

        return ncrisc_named_compile_time_args, brisc_named_compile_time_args, trisc_named_compile_time_args

    @staticmethod
    def _build_cb_descriptors(
        ctx, activation_tensor, down_weights_tensor, bias_tensor, output_tensor, gate_up_weights_tensor
    ):
        """Build all 15 circular buffer descriptors."""
        # CB 0: A_gather_dst — tensor-backed on sender (backed by dummy tensor)
        group1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ctx.group1_cb, ctx.ag_dummy_tensor)
        if ctx.use_face_view:
            group1_cb_descriptor.format_descriptors[0].tile = ctx.face_tile_desc
            group1_cb_descriptor.format_descriptors[0].page_size = ctx.face_tile_size

        # CB 1: B_gather_dst — tensor-backed on sender (backed by dummy tensor)
        group2_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ctx.group2_cb, ctx.bg_dummy_tensor)
        if ctx.use_face_view:
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

        # CB 3: Mcast1 source — on sender core, manual, TRISC-filled
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

        # CB 4: Mcast1 dest / down_matmul_in0 — all 130 cores
        mcast_dst_format = ttnn.CBFormatDescriptor(
            buffer_index=ctx.mcast_dst_cb,
            data_format=ctx.data_format,
            page_size=ctx.input_tile_size,
            tile=input_tile_desc,
        )
        mcast_dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=ctx.K_down_tiles * ctx.input_tile_size,
            core_ranges=ctx.all_cores,
            format_descriptors=[mcast_dst_format],
        )

        # CB 5: Down weights — tensor-backed on 112 matmul cores
        matmul_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ctx.matmul_in1_cb, down_weights_tensor)

        # CB 6: Down matmul output — on 112 matmul cores
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

        # CB 7: Output gather destination — tensor-backed on sender
        gather_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ctx.gather_dst_cb, output_tensor)

        # CB 8: Activation mcast source — tensor-backed on sender
        act_mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ctx.act_mcast_src_cb, activation_tensor)

        # CB 9: Activation mcast recv — manual on all 130 cores
        act_mcast_dst_format = ttnn.CBFormatDescriptor(
            buffer_index=ctx.act_mcast_dst_cb,
            data_format=ctx.data_format,
            page_size=ctx.input_tile_size,
            tile=input_tile_desc,
        )
        act_mcast_dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=ctx.K_gate_tiles * ctx.input_tile_size,
            core_ranges=ctx.all_cores,
            format_descriptors=[act_mcast_dst_format],
        )

        # CB 10: Residual mcast source — tensor-backed on sender
        residual_mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            ctx.residual_mcast_src_cb, bias_tensor
        )

        # CB 11: Residual mcast dest — all 130 cores, manual
        residual_mcast_dst_format = ttnn.CBFormatDescriptor(
            buffer_index=ctx.residual_mcast_dst_cb,
            data_format=ctx.data_format,
            page_size=ctx.tile_1x32_size,
            tile=ctx.tile_1x32_descriptor,
        )
        residual_mcast_dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=ctx.total_residual_add_tiles * ctx.tile_1x32_size,
            core_ranges=ctx.all_cores,
            format_descriptors=[residual_mcast_dst_format],
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

        # CB 13: Gate/Up weights — tensor-backed on 128 compute cores
        gu_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ctx.gu_weights_cb, gate_up_weights_tensor)

        # CB 14: Gate/Up matmul output — 1 tile on 128 compute cores, manual
        compute_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in ctx.compute_cores_list])
        gu_out_format = ttnn.CBFormatDescriptor(
            buffer_index=ctx.gu_out_cb,
            data_format=ctx.data_format,
            page_size=ctx.input_tile_size,
            tile=input_tile_desc,
        )
        gu_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=ctx.input_tile_size,
            core_ranges=compute_core_grid,
            format_descriptors=[gu_out_format],
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
            act_mcast_src_cb_descriptor,
            act_mcast_dst_cb_descriptor,
            residual_mcast_src_cb_descriptor,
            residual_mcast_dst_cb_descriptor,
            residual_add_out_cb_descriptor,
            gu_weights_cb_descriptor,
            gu_out_cb_descriptor,
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

        # Per-core: gate/up gather sender indices
        def compute_gu_sender_indices(cores_list):
            indices = []
            for linear_id, core in enumerate(cores_list):
                k_idx = linear_id // ctx.n_parallel
                n_idx = linear_id % ctx.n_parallel
                if ctx.use_face_view:
                    sender_idx = k_idx * ctx.n_parallel + n_idx
                else:
                    sender_idx = n_idx * ctx.tiles_per_k + k_idx
                indices.append((core, sender_idx))
            return indices

        per_core_ag_idx = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="ag_sender_idx",
            core_values=compute_gu_sender_indices(ctx.a_cores_list),
            other_value=0,
        )
        per_core_bg_idx = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="bg_sender_idx",
            core_values=compute_gu_sender_indices(ctx.b_cores_list),
            other_value=0,
        )

        # Per-core: gate/up matmul K-offset
        per_core_gu_k_offset = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="gu_k_offset",
            core_values=[(core, (i // ctx.n_parallel) * ctx.k_per_core) for i, core in enumerate(ctx.a_cores_list)]
            + [(core, (i // ctx.n_parallel) * ctx.k_per_core) for i, core in enumerate(ctx.b_cores_list)],
            other_value=0,
        )

        # Unified core descriptors (role flags)
        core_descs = [
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_gate_compute_core",
                core_range=ctx.a_compute_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_up_compute_core",
                core_range=ctx.b_compute_grid,
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

        per_core_descs = [per_core_gather_idx, per_core_ag_idx, per_core_bg_idx, per_core_gu_k_offset]

        return core_descs, per_core_descs

    @staticmethod
    def op(
        activation_tensor,
        gate_up_weights_tensor,
        down_weights_tensor,
        bias_tensor,
        output_tensor,
        k_parallel,
        n_parallel,
    ):
        """
        Execute shared expert operation.

        Args:
            activation_tensor: [1, K_gate] HEIGHT_SHARDED on (12,9)
            gate_up_weights_tensor: [128*k_per_core, 32] HEIGHT_SHARDED on 128 compute cores (row_wise)
            down_weights_tensor: [K_down, N_per_core] WIDTH_SHARDED on 112 matmul cores
            bias_tensor: [1, N] HEIGHT_SHARDED on (12,9)
            output_tensor: [1, N] HEIGHT_SHARDED on (12,9)
            k_parallel: Number of K-parallel partitions per branch (e.g., 8)
            n_parallel: Number of N-parallel partitions per branch (e.g., 8)

        Returns:
            Output tensor with result
        """
        ctx = SharedExpertOp._setup_dimensions(
            activation_tensor,
            gate_up_weights_tensor,
            down_weights_tensor,
            bias_tensor,
            output_tensor,
            k_parallel,
            n_parallel,
        )

        ncrisc_args, brisc_args, trisc_args = SharedExpertOp._build_compile_time_args(ctx)
        cb_descriptors = SharedExpertOp._build_cb_descriptors(
            ctx, activation_tensor, down_weights_tensor, bias_tensor, output_tensor, gate_up_weights_tensor
        )
        core_descs, per_core_descs = SharedExpertOp._build_core_descriptors(ctx)

        # Semaphore descriptors
        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=ctx.full_device_grid, initial_value=0)
            for i in range(ctx.num_semaphores)
        ]

        # Kernel descriptor
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/fused_ops/shared_expert/kernels/shared_expert_kernel.cpp",
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
            ctx.ag_dummy_tensor,
            ctx.bg_dummy_tensor,
            down_weights_tensor,
            output_tensor,
            activation_tensor,
            bias_tensor,
            gate_up_weights_tensor,
        ]
        ttnn.generic_op(io_tensors, program_descriptor)

        # Release dummy tensors to avoid holding device memory
        ttnn.deallocate(ctx.ag_dummy_tensor)
        ttnn.deallocate(ctx.bg_dummy_tensor)

        return output_tensor
