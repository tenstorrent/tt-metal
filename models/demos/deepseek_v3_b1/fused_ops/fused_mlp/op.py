# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Fused MLP operation.

Fuses: Activation Mcast + Gate/Up Matmul + Gather + GatedLocalReduce
       + Mcast1 + Mcast2 + Down Proj Matmul + Add + Output Gather

Pipeline:
  1. Activation mcast: [1, K_gate] from (12,9) → all 130 cores
  2. Gate matmul on 64 A cores, Up matmul on 64 B cores
  3. Gather A→CB0, B→CB1 on (12,9)
  4. Gated reduce: SiLU(sum(A)) * sum(B) → [1, K_down]
  5. Mcast1: [1, K_down] from (12,9) → all 130 cores
  6. Mcast2: bias [1, N] from (12,9) → all 130 cores
  7. Down proj matmul on 112 cores
  8. Add bias on 112 cores
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
  CB 10: bias_mcast_src (sender, tensor-backed)
  CB 11: bias_mcast_dst (all 130 cores, manual)
  CB 12: add_out (112 matmul cores, manual)
  CB 13: gate_up_weights (128 compute cores, tensor-backed)
  CB 14: gate_up_matmul_out (128 compute cores, 1 tile, manual)
"""

import torch
import torch.nn.functional as F

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

# Face dimensions (hardware constant)
FACE_HEIGHT = 16
FACE_WIDTH = 16
FACE_ELEMENTS = FACE_HEIGHT * FACE_WIDTH  # 256


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


class FusedMLPOp:
    """
    Fused MLP operation.

    Computes: (SiLU(activation @ gate_weights) * (activation @ up_weights)) @ down_weights + bias
    """

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
    def can_use_face_view(tile_h, tile_w, tiles_per_k, k_num_tiles):
        """
        Check if face-view optimization can be applied.

        Face-view requires:
        1. Tiles smaller than a face (tile_h * tile_w < 256)
        2. Each collection's k_num_tiles tiles exactly fill one face (k_num_tiles * tile_w = 256)
        3. tiles_per_k >= 2 (need at least 2 faces to reduce)
        4. tiles_per_k is even (for pairwise addition)
        """
        elements_per_tile = tile_h * tile_w
        if elements_per_tile >= FACE_ELEMENTS:
            return False

        if elements_per_tile * k_num_tiles != FACE_ELEMENTS:
            return False

        if tiles_per_k < 2:
            return False

        if tiles_per_k % 2 != 0:
            return False

        return True

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
        Execute fused MLP operation.

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
        device = activation_tensor.device()
        data_format = activation_tensor.dtype

        # ====================================================================
        # Tile definitions
        # ====================================================================
        TILE_1x32 = ttnn.Tile((1, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)
        tile_1x32_descriptor = ttnn.TileDescriptor(TILE_1x32)

        # ====================================================================
        # Core grid configuration (reuse DownProj layout)
        # ====================================================================
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
        a_cores_list, b_cores_list = build_ab_grids()
        a_compute_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in a_cores_list])
        b_compute_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in b_cores_list])
        compute_cores_list = a_cores_list + b_cores_list  # 128 cores

        # Mcast receiver grid = all 130 minus sender (12,9)
        mcast_receiver_grid = DownProj.build_mcast_receiver_grid()

        all_cores = mcast_grid_set

        # NOC coordinates
        gather_dest_noc_core = device.worker_core_from_logical_core(mcast_gather_core)
        mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
        mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)

        # ====================================================================
        # Dimension parameters
        # ====================================================================
        input_tile = activation_tensor.get_tile()
        tile_h, tile_w = input_tile.tile_shape
        input_tile_size = input_tile.get_tile_size(data_format)

        K_gate = activation_tensor.shape[1]
        K_gate_tiles = K_gate // tile_w  # total activation tiles

        # Gate/Up matmul: 64 cores per branch = k_parallel * n_parallel
        # Each produces 1 [1,32] tile. Gated reduce sums k_parallel K-partials
        # per N-position → n_parallel output tiles → K_down = n_parallel * 32
        num_compute_cores = 64  # per branch
        k_per_core = K_gate_tiles // k_parallel
        K_down_tiles = n_parallel
        K_down = K_down_tiles * 32

        # Down proj dimensions
        down_weights_shard_spec = down_weights_tensor.memory_config().shard_spec
        n_per_core = down_weights_shard_spec.shape[1]
        out_w_per_core = n_per_core // TILE_1x32.tile_shape[1]
        N = DownProj.NUM_MATMUL_CORES * n_per_core

        # ====================================================================
        # Face-view parameters for gated reduce
        # tiles_per_k = k_parallel (number of K-partials to sum per N-position)
        # k_num_tiles = n_parallel (number of N-positions)
        # ====================================================================
        tiles_per_k = k_parallel
        k_num_tiles = n_parallel

        use_face_view = FusedMLPOp.can_use_face_view(tile_h, tile_w, tiles_per_k, k_num_tiles)

        if use_face_view:
            face_tile = ttnn.Tile([FACE_HEIGHT, FACE_WIDTH])
            face_tile_desc = ttnn.TileDescriptor(FACE_HEIGHT, FACE_WIDTH, False)
            face_tile_size = face_tile.get_tile_size(data_format)
            kernel_tiles_per_k = tiles_per_k  # face tiles to reduce
            kernel_k_num_tiles = 1  # single iteration
            mcast_src_num_pages = 1  # 1 face tile output
            reduce_tile_size = face_tile_size
        else:
            face_tile_desc = None
            face_tile_size = None
            kernel_tiles_per_k = tiles_per_k
            kernel_k_num_tiles = k_num_tiles
            mcast_src_num_pages = k_num_tiles
            reduce_tile_size = input_tile_size

        # ====================================================================
        # CB indices
        # ====================================================================
        group1_cb = 0  # A_gather_dst / reduce_g1
        group2_cb = 1  # B_gather_dst / reduce_g2
        intermed_cb = 2  # intermediate (gated reduce)
        mcast_src_cb = 3  # mcast1 source (TRISC-filled)
        mcast_dst_cb = 4  # mcast1 dest / down_matmul_in0
        matmul_in1_cb = 5  # down_weights (tensor-backed)
        matmul_out_cb = 6  # down_matmul_out
        gather_dst_cb = 7  # output_gather_dst (tensor-backed)
        act_mcast_src_cb = 8  # activation mcast source (tensor-backed)
        act_mcast_dst_cb = 9  # activation mcast recv (manual)
        bias_mcast_src_cb = 10  # bias mcast source (tensor-backed)
        bias_mcast_dst_cb = 11  # bias mcast dest
        add_out_cb = 12  # add output
        gu_weights_cb = 13  # gate/up weights (tensor-backed)
        gu_out_cb = 14  # gate/up matmul output

        # ====================================================================
        # Activation mcast parameters
        # ====================================================================
        act_mcast_data_size_bytes = K_gate_tiles * input_tile_size
        act_mcast_is_part_of_receiver_grid = mcast_grid.contains(mcast_gather_core)

        # ====================================================================
        # Mcast1 parameters (down proj activation)
        # ====================================================================
        mcast_data_size_bytes = K_down_tiles * input_tile_size
        mcast_is_part_of_receiver_grid = mcast_grid.contains(mcast_gather_core)

        # ====================================================================
        # Mcast2 parameters (bias)
        # ====================================================================
        total_add_tiles = DownProj.NUM_MATMUL_CORES * out_w_per_core  # = N / 32
        bias_mcast_data_size_bytes = total_add_tiles * tile_1x32_size

        # ====================================================================
        # Output gather parameters
        # ====================================================================
        gather_data_size_bytes = out_w_per_core * tile_1x32_size
        gather_src_num_pages = out_w_per_core
        gather_dst_num_pages = DownProj.NUM_MATMUL_CORES * out_w_per_core
        gather_noc0_num_senders = DownProj.NUM_MATMUL_CORES
        gather_noc1_num_senders = 0

        # ====================================================================
        # Gate/Up gather parameters
        # Each of 64 compute cores sends 1 tile to (12,9)
        # ====================================================================
        gu_gather_data_size_bytes = input_tile_size  # 1 [1,32] tile per sender
        total_gather_tiles = num_compute_cores  # 64 tiles per group

        # ====================================================================
        # Semaphore IDs (10 total: 0-9)
        # mcast1 and mcast2 share semaphores 0/1 (same grid, sequential sends)
        # ====================================================================
        mcast_data_sender_semaphore_id = 0  # mcast1 + mcast2 sender
        mcast_data_receiver_semaphore_id = 1  # mcast1 + mcast2 receiver
        gather_noc0_receiver_semaphore_id = 2  # output gather
        gather_noc1_receiver_semaphore_id = 3  # output gather (unused NOC1 dummy)
        ag_receiver_semaphore_id = 4  # gate gather (A)
        bg_receiver_semaphore_id = 5  # up gather (B)
        ag_noc1_receiver_semaphore_id = 6  # gate gather NOC1 dummy
        bg_noc1_receiver_semaphore_id = 7  # up gather NOC1 dummy
        act_mcast_sender_semaphore_id = 8  # activation mcast sender
        act_mcast_receiver_semaphore_id = 9  # activation mcast receiver

        # ====================================================================
        # Buffer addresses
        # ====================================================================
        gather_receiver_data_addr = output_tensor.buffer_address()

        # Gate/Up gather destinations (CB0/CB1): backed by dummy tensors on sender
        # to get a known L1 address for NOC writes (same pattern as gated_local_reduce_down_proj)
        ag_dummy_shape = (total_gather_tiles, 32)
        ag_dummy_shard_spec = ttnn.ShardSpec(mcast_gather_core_grid, ag_dummy_shape, ttnn.ShardOrientation.ROW_MAJOR)
        ag_dummy_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ag_dummy_shard_spec
        )

        ag_dummy_tensor = ttnn.from_torch(
            torch.zeros(total_gather_tiles, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ag_dummy_mem,
            tile=input_tile,
        )
        bg_dummy_tensor = ttnn.from_torch(
            torch.zeros(total_gather_tiles, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ag_dummy_mem,
            tile=input_tile,
        )

        ag_receiver_data_addr = ag_dummy_tensor.buffer_address()
        bg_receiver_data_addr = bg_dummy_tensor.buffer_address()

        # ====================================================================
        # Per-core compile-time descriptors
        # ====================================================================
        # Output gather: 112 matmul cores → per-core sender index
        matmul_cores_list = ttnn.corerange_to_cores(matmul_core_grid)
        per_core_gather_idx = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="gather_sender_idx",
            core_values=[(core, idx) for idx, core in enumerate(matmul_cores_list)],
            other_value=0,
        )

        # Gate gather (A): 64 A cores → per-core sender index
        # Standard K-major ordering: sender_idx = linear_id in A grid
        # With face-view: collection-major (collection * k_num_tiles + k)
        # Without face-view: K-interleaved (k * tiles_per_k + collection)
        def compute_gu_sender_indices(cores_list):
            """Compute sender indices for gate/up gather.

            Core linear_id=i: k_idx = i // n_parallel, n_idx = i % n_parallel
            Face-view (collection-major): sender_idx = k_idx * n_parallel + n_idx = i
            Non-face-view (K-interleaved): sender_idx = n_idx * k_parallel + k_idx
            """
            indices = []
            for linear_id, core in enumerate(cores_list):
                k_idx = linear_id // n_parallel
                n_idx = linear_id % n_parallel
                if use_face_view:
                    sender_idx = k_idx * n_parallel + n_idx
                else:
                    sender_idx = n_idx * tiles_per_k + k_idx
                indices.append((core, sender_idx))
            return indices

        per_core_ag_idx = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="ag_sender_idx",
            core_values=compute_gu_sender_indices(a_cores_list),
            other_value=0,
        )
        per_core_bg_idx = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="bg_sender_idx",
            core_values=compute_gu_sender_indices(b_cores_list),
            other_value=0,
        )

        # Gate/Up matmul: per-core K-offset for activation indexing
        # Core with linear_id=i: k_idx = i // n_parallel, k_offset = k_idx * k_per_core
        per_core_gu_k_offset = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="gu_k_offset",
            core_values=[(core, (i // n_parallel) * k_per_core) for i, core in enumerate(a_cores_list)]
            + [(core, (i // n_parallel) * k_per_core) for i, core in enumerate(b_cores_list)],
            other_value=0,
        )

        # ====================================================================
        # NCRISC compile-time args
        # ====================================================================
        ncrisc_named_compile_time_args = [
            # Activation mcast source (setup_sharded_buffer on sender)
            ("act_mcast_src_cb", act_mcast_src_cb),
            ("act_mcast_src_num_pages", K_gate_tiles),
            # Activation mcast receiver
            ("act_mcast_data_receiver_semaphore", act_mcast_receiver_semaphore_id),
            ("act_mcast_dst_cb", act_mcast_dst_cb),
            ("act_mcast_dst_num_pages", K_gate_tiles),
            # Gate/Up weights (setup_sharded_buffer on compute cores)
            ("gu_weights_cb", gu_weights_cb),
            ("gu_weights_num_pages", k_per_core),  # k_per_core [32,32] weight tiles per core
            # Gate gather (A) sender
            ("ag_dest_noc_x", gather_dest_noc_core.x),
            ("ag_dest_noc_y", gather_dest_noc_core.y),
            ("ag_data_size_bytes", gu_gather_data_size_bytes),
            ("ag_receiver_semaphore_id", ag_receiver_semaphore_id),
            ("ag_src_cb", gu_out_cb),
            ("ag_src_num_pages", 1),
            ("ag_receiver_data_addr", ag_receiver_data_addr),
            # Up gather (B) sender
            ("bg_dest_noc_x", gather_dest_noc_core.x),
            ("bg_dest_noc_y", gather_dest_noc_core.y),
            ("bg_data_size_bytes", gu_gather_data_size_bytes),
            ("bg_receiver_semaphore_id", bg_receiver_semaphore_id),
            ("bg_src_cb", gu_out_cb),
            ("bg_src_num_pages", 1),
            ("bg_receiver_data_addr", bg_receiver_data_addr),
            # Mcast1 receiver (down proj activation)
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_dst_cb", mcast_dst_cb),
            ("mcast_dst_num_pages", K_down_tiles),
            # Down proj weights (setup_sharded_buffer on matmul cores)
            ("matmul_in1", matmul_in1_cb),
            ("matmul_k_num_tiles", K_down_tiles),
            ("matmul_out_w_per_core", out_w_per_core),
            # Mcast2 source (bias on sender)
            ("mcast2_src_cb", bias_mcast_src_cb),
            ("mcast2_src_num_pages", total_add_tiles),
            # Mcast2 receiver (bias, shares semaphores with mcast1)
            ("mcast2_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast2_dst_cb", bias_mcast_dst_cb),
            ("mcast2_dst_num_pages", total_add_tiles),
            # Down proj matmul (NCRISC no-op, but args needed for split kernel)
            ("matmul_in0", mcast_dst_cb),
            ("matmul_out", matmul_out_cb),
            # Output gather sender
            ("gather_dest_noc_x", gather_dest_noc_core.x),
            ("gather_dest_noc_y", gather_dest_noc_core.y),
            ("gather_data_size_bytes", gather_data_size_bytes),
            ("gather_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("gather_src_cb", add_out_cb),
            ("gather_src_num_pages", gather_src_num_pages),
            ("gather_sender_grid_start_x", 0),
            ("gather_sender_grid_start_y", 0),
            ("gather_sender_grid_end_x", 0),
            ("gather_sender_grid_end_y", 0),
            ("gather_row_major", 1),
            ("gather_receiver_data_addr", gather_receiver_data_addr),
        ]

        # ====================================================================
        # BRISC compile-time args
        # ====================================================================
        brisc_named_compile_time_args = [
            # Activation mcast sender
            ("act_mcast_dest_noc_start_x", mcast_dest_noc_start.x),
            ("act_mcast_dest_noc_start_y", mcast_dest_noc_start.y),
            ("act_mcast_dest_noc_end_x", mcast_dest_noc_end.x),
            ("act_mcast_dest_noc_end_y", mcast_dest_noc_end.y),
            ("act_mcast_num_cores", num_mcast_cores),
            ("act_mcast_data_sender_semaphore", act_mcast_sender_semaphore_id),
            ("act_mcast_data_receiver_semaphore", act_mcast_receiver_semaphore_id),
            ("act_mcast_data_size_bytes", act_mcast_data_size_bytes),
            ("act_mcast_src_cb", act_mcast_src_cb),
            ("act_mcast_src_num_pages", K_gate_tiles),
            ("act_mcast_dst_cb", act_mcast_dst_cb),
            ("act_mcast_is_part_of_receiver_grid", act_mcast_is_part_of_receiver_grid),
            # Gate gather (A) receiver
            ("ag_noc0_num_senders", num_compute_cores),
            ("ag_noc0_receiver_semaphore_id", ag_receiver_semaphore_id),
            ("ag_noc1_receiver_semaphore_id", ag_noc1_receiver_semaphore_id),
            ("ag_dst_cb", group1_cb),
            ("ag_dst_num_pages", kernel_tiles_per_k if use_face_view else total_gather_tiles),
            # Up gather (B) receiver
            ("bg_noc0_num_senders", num_compute_cores),
            ("bg_noc0_receiver_semaphore_id", bg_receiver_semaphore_id),
            ("bg_noc1_receiver_semaphore_id", bg_noc1_receiver_semaphore_id),
            ("bg_dst_cb", group2_cb),
            ("bg_dst_num_pages", kernel_tiles_per_k if use_face_view else total_gather_tiles),
            # Mcast1 sender (down proj activation)
            ("mcast_dest_noc_start_x", mcast_dest_noc_start.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end.y),
            ("mcast_num_cores", num_mcast_cores),
            ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", mcast_src_cb),
            ("mcast_src_num_pages", mcast_src_num_pages),
            ("mcast_dst_cb", mcast_dst_cb),
            ("mcast_is_part_of_receiver_grid", mcast_is_part_of_receiver_grid),
            # Mcast2 sender (bias, shares semaphores with mcast1)
            ("mcast2_data_sender_semaphore", mcast_data_sender_semaphore_id),
            ("mcast2_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast2_data_size_bytes", bias_mcast_data_size_bytes),
            ("mcast2_src_cb", bias_mcast_src_cb),
            ("mcast2_src_num_pages", total_add_tiles),
            ("mcast2_dst_cb", bias_mcast_dst_cb),
            # Output gather receiver
            ("gather_noc0_num_senders", gather_noc0_num_senders),
            ("gather_noc1_num_senders", gather_noc1_num_senders),
            ("gather_noc0_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("gather_noc1_receiver_semaphore_id", gather_noc1_receiver_semaphore_id),
            ("gather_dst_cb", gather_dst_cb),
            ("gather_dst_num_pages", gather_dst_num_pages),
        ]

        # ====================================================================
        # TRISC compile-time args
        # ====================================================================
        trisc_named_compile_time_args = [
            # Gate/Up matmul
            ("gu_act_cb", act_mcast_dst_cb),
            ("gu_weights_cb", gu_weights_cb),
            ("gu_out_cb", gu_out_cb),
            ("gu_k_per_core", k_per_core),
            ("gu_act_total_tiles", K_gate_tiles),
            # Gated reduce
            ("gated_reduce_group1_cb", group1_cb),
            ("gated_reduce_group2_cb", group2_cb),
            ("gated_reduce_intermed_cb", intermed_cb),
            ("gated_reduce_mcast_src_cb", mcast_src_cb),
            ("gated_reduce_tiles_per_k", kernel_tiles_per_k),
            ("gated_reduce_k_num_tiles", kernel_k_num_tiles),
            # Down proj matmul
            ("matmul_in0", mcast_dst_cb),
            ("matmul_in1", matmul_in1_cb),
            ("matmul_out", matmul_out_cb),
            ("matmul_k_num_tiles", K_down_tiles),
            ("matmul_out_w_per_core", out_w_per_core),
            # Add step
            ("add_in0", matmul_out_cb),
            ("add_in1", bias_mcast_dst_cb),
            ("add_out", add_out_cb),
            ("add_out_w", out_w_per_core),
            ("add_total_in1_tiles", total_add_tiles),
        ]

        # ====================================================================
        # Circular buffer descriptors
        # ====================================================================
        # CB 0: A_gather_dst — tensor-backed on sender (backed by dummy tensor)
        group1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(group1_cb, ag_dummy_tensor)
        if use_face_view:
            group1_cb_descriptor.format_descriptors[0].tile = face_tile_desc
            group1_cb_descriptor.format_descriptors[0].page_size = face_tile_size

        # CB 1: B_gather_dst — tensor-backed on sender (backed by dummy tensor)
        group2_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(group2_cb, bg_dummy_tensor)
        if use_face_view:
            group2_cb_descriptor.format_descriptors[0].tile = face_tile_desc
            group2_cb_descriptor.format_descriptors[0].page_size = face_tile_size

        # CB 2: Intermediate — 2 tiles on sender core, manual
        intermed_format = ttnn.CBFormatDescriptor(
            buffer_index=intermed_cb,
            data_format=data_format,
            page_size=reduce_tile_size,
            tile=face_tile_desc if use_face_view else ttnn.TileDescriptor(input_tile),
        )
        intermed_cb_descriptor = ttnn.CBDescriptor(
            total_size=2 * reduce_tile_size,
            core_ranges=mcast_gather_core_grid,
            format_descriptors=[intermed_format],
        )

        # CB 3: Mcast1 source — on sender core, manual, TRISC-filled
        mcast_src_format = ttnn.CBFormatDescriptor(
            buffer_index=mcast_src_cb,
            data_format=data_format,
            page_size=reduce_tile_size,
            tile=face_tile_desc if use_face_view else ttnn.TileDescriptor(input_tile),
        )
        mcast_src_cb_descriptor = ttnn.CBDescriptor(
            total_size=mcast_src_num_pages * reduce_tile_size,
            core_ranges=mcast_gather_core_grid,
            format_descriptors=[mcast_src_format],
        )

        # CB 4: Mcast1 dest / down_matmul_in0 — all 130 cores
        mcast_dst_format = ttnn.CBFormatDescriptor(
            buffer_index=mcast_dst_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=ttnn.TileDescriptor(input_tile),
        )
        mcast_dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=K_down_tiles * input_tile_size,
            core_ranges=all_cores,
            format_descriptors=[mcast_dst_format],
        )

        # CB 5: Down weights — tensor-backed on 112 matmul cores
        matmul_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_in1_cb, down_weights_tensor)

        # CB 6: Down matmul output — on 112 matmul cores
        matmul_out_format = ttnn.CBFormatDescriptor(
            buffer_index=matmul_out_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=tile_1x32_descriptor,
        )
        matmul_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_w_per_core * tile_1x32_size,
            core_ranges=matmul_core_grid,
            format_descriptors=[matmul_out_format],
        )

        # CB 7: Output gather destination — tensor-backed on sender
        gather_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gather_dst_cb, output_tensor)

        # CB 8: Activation mcast source — tensor-backed on sender
        act_mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(act_mcast_src_cb, activation_tensor)

        # CB 9: Activation mcast recv — manual on all 130 cores
        # Must be manual (not tensor-backed) so all cores share the same L1 address for mcast
        act_mcast_dst_format = ttnn.CBFormatDescriptor(
            buffer_index=act_mcast_dst_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=ttnn.TileDescriptor(input_tile),
        )
        act_mcast_dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=K_gate_tiles * input_tile_size,
            core_ranges=all_cores,
            format_descriptors=[act_mcast_dst_format],
        )

        # CB 10: Bias mcast source — tensor-backed on sender
        bias_mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(bias_mcast_src_cb, bias_tensor)

        # CB 11: Bias mcast dest — all 130 cores, manual
        bias_mcast_dst_format = ttnn.CBFormatDescriptor(
            buffer_index=bias_mcast_dst_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=tile_1x32_descriptor,
        )
        bias_mcast_dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=total_add_tiles * tile_1x32_size,
            core_ranges=all_cores,
            format_descriptors=[bias_mcast_dst_format],
        )

        # CB 12: Add output — on 112 matmul cores
        add_out_format = ttnn.CBFormatDescriptor(
            buffer_index=add_out_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=tile_1x32_descriptor,
        )
        add_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_w_per_core * tile_1x32_size,
            core_ranges=matmul_core_grid,
            format_descriptors=[add_out_format],
        )

        # CB 13: Gate/Up weights — tensor-backed on 128 compute cores
        gu_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gu_weights_cb, gate_up_weights_tensor)

        # CB 14: Gate/Up matmul output — 1 tile on 128 compute cores, manual
        compute_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in compute_cores_list])
        gu_out_format = ttnn.CBFormatDescriptor(
            buffer_index=gu_out_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=ttnn.TileDescriptor(input_tile),
        )
        gu_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=input_tile_size,
            core_ranges=compute_core_grid,
            format_descriptors=[gu_out_format],
        )

        # ====================================================================
        # Semaphore descriptors
        # ====================================================================
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

        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=full_device_grid, initial_value=0) for i in range(10)
        ]

        # ====================================================================
        # Kernel descriptor
        # ====================================================================
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/fused_ops/fused_mlp/kernels/fused_mlp_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_gate_compute_core",
                    core_range=a_compute_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_up_compute_core",
                    core_range=b_compute_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_gated_reduce_core",
                    core_range=mcast_gather_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_mcast_sender_core",
                    core_range=mcast_gather_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_mcast_receiver_core",
                    core_range=mcast_receiver_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_matmul_core",
                    core_range=matmul_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_gather_receiver_core",
                    core_range=mcast_gather_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="gather_use_per_core_sender_idx",
                    core_range=matmul_core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
            per_core_compile_time_descriptors=[
                per_core_gather_idx,
                per_core_ag_idx,
                per_core_bg_idx,
                per_core_gu_k_offset,
            ],
        )

        # ====================================================================
        # Program descriptor
        # ====================================================================
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[
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
                bias_mcast_src_cb_descriptor,
                bias_mcast_dst_cb_descriptor,
                add_out_cb_descriptor,
                gu_weights_cb_descriptor,
                gu_out_cb_descriptor,
            ],
            semaphores=semaphore_descriptors,
        )

        # Execute generic op
        # Order must match tensor-backed CBs: CB0(ag_dummy), CB1(bg_dummy), CB5(down_weights),
        # CB7(output), CB8(activation), CB10(bias), CB13(gate_up_weights)
        # Note: CB9 is now manual (not tensor-backed), so no tensor for it
        io_tensors = [
            ag_dummy_tensor,
            bg_dummy_tensor,
            down_weights_tensor,
            output_tensor,
            activation_tensor,
            bias_tensor,
            gate_up_weights_tensor,
        ]
        ttnn.generic_op(io_tensors, program_descriptor)
        return output_tensor
