# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Post SDPA fused operation with CCL All-Reduce.

This implements Matmul1 + Gather1 + Mcast + Matmul2 + Gather2 + CCL All-Reduce as a fused execution:
- Matmul1: [1, 512] x [512, 128] -> [1, 128] distributed across 64 cores (8x8 grid)
- Gather1: Collect results from all 64 cores to [1, 8192] on gather core (12, 9)
- Mcast: Broadcast [1, 8192] to 130 cores (13x10 grid, rectangular for efficient mcast)
- Matmul2: [1, 8192] x [8192, 64] -> [1, 64] on 112 active cores (rows 0-8 full 12 + row 9 cols 0-3)
- Gather2: Collect results from all 112 active cores to [1, 7168] on gather core (12, 9)
- CCL All-Reduce: Exchange [1, 7168] between devices and reduce (local + remote + residual)

The 13x10 mcast grid contains 130 cores, but only 112 are active for matmul2.
The 8 inactive cores (row 9 cols 4-11) receive mcast data but skip matmul via is_matmul2_core=false.

CCL All-Reduce uses:
- CCL Receiver core = Gather core (12, 9) - already has local data after Gather2
- CCL Sender core = Adjacent core (11, 9) - reads from gather core, sends via fabric

CB Layout:
- CB 0: matmul1_in0 (8x8 grid)
- CB 1: matmul1_in1 (8x8 grid)
- CB 2: matmul1_out (8x8 grid)
- CB 3: gather1_dst = mcast_src (gather core)
- CB 4: mcast_dst = matmul2_in0 (13x10 grid)
- CB 5: matmul2_in1 (112 active matmul2 cores)
- CB 6: matmul2_out (112 active matmul2 cores)
- CB 7: gather2_dst = ccl_local_data (gather core)
- CB 8: ccl_sender_in (ccl sender core)
- CB 9: ccl_remote_data (gather/receiver core - intermediate tensor)
- CB 10: ccl_residual (gather/receiver core - optional)
- CB 11: ccl_temp (gather/receiver core - for compute)
- CB 12: ccl_output (gather/receiver core - final output)
- CB 13: ccl_packet_header (sender + receiver cores)
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _get_element_size_bytes(dtype):
    if dtype == ttnn.bfloat16:
        return 2
    if dtype == ttnn.float32:
        return 4
    raise ValueError(f"Unsupported dtype for sdpa: {dtype}")


class PostSDPA:
    """
    Post SDPA fused operation implementation using ttnn.generic_op.

    Implements the full post_sdpa fusion with CCL all-reduce:
    - Matmul1 + Gather1 + Mcast + Matmul2 + Gather2 + CCL All-Reduce
    """

    @staticmethod
    def golden(input_tensors, weights1_tensor, weights2_tensor, residual_tensor=None):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensors: List of input tensors (torch.Tensor) [1, 512], one per device
            weights1_tensor: First weights tensor (torch.Tensor) [512, 8192]
            weights2_tensor: Second weights tensor (torch.Tensor) [8192, 7168]
            residual_tensor: Optional residual tensor to add after reduction [1, 7168]

        Returns:
            Output tensor [1, 7168] - result of all-reduce across devices
        """
        # Compute matmul chain for each device
        device_results = []
        for input_tensor in input_tensors:
            intermediate = input_tensor @ weights1_tensor  # [1, 8192]
            result = intermediate @ weights2_tensor  # [1, 7168]
            device_results.append(result)

        # All-reduce: sum across all devices
        reduced = torch.sum(torch.stack(device_results), dim=0)

        # Add residual if provided
        if residual_tensor is not None:
            reduced = reduced + residual_tensor

        return reduced

    @staticmethod
    def op(
        input_tensor_mesh,
        weights1_tensor,
        weights2_tensor,
        gather1_output_tensor,
        gather2_output_tensor,
        intermediate_tensor=None,
        output_tensor=None,
        semaphores=None,
        cluster_axis=0,
        residual_tensor_mesh=None,
        fp32_dest_acc_en=False,
        ccl_enabled=True,
        sdpa_kv_cache_buffer=None,
        # SDPA Reduce-to-All parameters (optional - when None, SDPA phase is skipped)
        sdpa_input_l_mesh=None,
        sdpa_input_ms_mesh=None,
        sdpa_output_l_mesh=None,
        sdpa_r1_recv_mesh=None,
        sdpa_r2_recv_mesh=None,
        sdpa_forwarder_scratch_mesh=None,
        sdpa_semaphores=None,
        sdpa_scale_fp32=1.0,
        sdpa_forwarder_cores=None,
        sdpa_cluster_axis=0,
    ):
        """
        Execute post_sdpa fused operation with optional SDPA reduce-to-all and CCL all-reduce.

        Args:
            input_tensor_mesh: Input tensor mesh [1, 512] (height-sharded across 8x8 matmul1 cores)
                               When sdpa_enabled, this receives scatter output from SDPA phase
            weights1_tensor: First weights tensor [512, 8192] (width-sharded across 8x8)
            weights2_tensor: Second weights tensor [8192, 7168] (width-sharded across 112 cores)
            gather1_output_tensor: Intermediate tensor [1, 8192] for gather1/mcast (on gather core)
            gather2_output_tensor: Intermediate tensor mesh [1, 7168] for gather2/CCL (on gather core per device)
            intermediate_tensor: CCL intermediate tensor mesh for receiving remote data (None when ccl_enabled=False)
            output_tensor: Final output tensor mesh [1, 7168] (None when ccl_enabled=False)
            semaphores: List of two global semaphores for CCL synchronization (None when ccl_enabled=False)
            cluster_axis: Axis for TP all-reduce at the end (default 0, typically 1 for column-wise reduction)
            residual_tensor_mesh: Optional tensor mesh for residuals [1, 7168] (ignored when ccl_enabled=False)
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel
            ccl_enabled: Whether to enable CCL all-reduce after gather2 (default True)
            sdpa_kv_cache_buffer: Optional SDPA kv-cache buffer for CB overlap (height-sharded on full
                device grid, 156672 B/core). When provided, non-tensor-backed CBs (2, 4, 6, 8, 11, 13)
                are overlapped into the kv-cache L1 buffer to save memory.

            SDPA Reduce-to-All parameters (all None to skip SDPA phase):
            sdpa_input_l_mesh: L input tensor mesh [8, 4096] (width-sharded across 8 SDPA workers)
            sdpa_input_ms_mesh: MS input tensor mesh [8, 256] (width-sharded across 8 SDPA workers)
            sdpa_output_l_mesh: L output tensor mesh [8, 4096] (width-sharded across 8 SDPA workers)
            sdpa_r1_recv_mesh: R1 receive buffer mesh for SDPA CCL
            sdpa_r2_recv_mesh: R2 receive buffer mesh for SDPA CCL
            sdpa_forwarder_scratch_mesh: Forwarder scratch buffer mesh
            sdpa_semaphores: List of two global semaphores for SDPA CCL
            sdpa_scale_fp32: Scale value for SDPA reduction (default 1.0)
            sdpa_forwarder_cores: List of 2 CoreCoord for SDPA forwarders
            sdpa_cluster_axis: Axis for SDPA all-reduce (default 0, typically reduces across rows)

        Returns:
            When ccl_enabled=True: output_tensor mesh with all-reduced result
            When ccl_enabled=False: gather2_output_tensor mesh with per-device gather2 result
        """
        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        gather2_output_tensors_per_device = ttnn.get_device_tensors(gather2_output_tensor)
        if ccl_enabled:
            intermediate_tensors_per_device = ttnn.get_device_tensors(intermediate_tensor)
            output_tensors_per_device = ttnn.get_device_tensors(output_tensor)
        if ccl_enabled and residual_tensor_mesh is not None:
            residual_tensors_per_device = ttnn.get_device_tensors(residual_tensor_mesh)
        sdpa_kv_cache_buffers_per_device = (
            ttnn.get_device_tensors(sdpa_kv_cache_buffer) if sdpa_kv_cache_buffer is not None else None
        )

        # CCL semaphores (only when CCL is enabled)
        if ccl_enabled:
            ccl_semaphore1 = semaphores[0]
            ccl_semaphore2 = semaphores[1]
            ccl_semaphore1_addr = ttnn.get_global_semaphore_address(ccl_semaphore1)
            ccl_semaphore2_addr = ttnn.get_global_semaphore_address(ccl_semaphore2)

        # Get tensor properties from first device
        input_tensor_sample = input_tensors_per_device[0]
        data_format = input_tensor_sample.dtype
        element_size = 2  # bfloat16

        # Tile definitions
        TILE_1x32 = ttnn.Tile((1, 32))
        TILE_32x32 = ttnn.Tile((32, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)
        tile_32x32_size = TILE_32x32.get_tile_size(data_format)

        # CCL intermediate tensor info (32x32 tiles, only when CCL is enabled)
        if ccl_enabled:
            intermediate_tensor_sample = intermediate_tensors_per_device[0]
            intermediate_tile = intermediate_tensor_sample.tile
            intermediate_tile_height, intermediate_tile_width = intermediate_tile.tile_shape
            standard_tile_size_bytes = intermediate_tile_height * intermediate_tile_width * element_size

        # ========================================================================
        # Core grid configuration (same for all devices)
        # ========================================================================
        # Matmul1 grid: 8x8 = 64 cores
        MATMUL1_GRID_START_X = 0
        MATMUL1_GRID_START_Y = 0
        MATMUL1_GRID_END_X = 7  # 8 columns (0-7)
        MATMUL1_GRID_END_Y = 7  # 8 rows (0-7)
        matmul1_grid = ttnn.CoreRange(
            ttnn.CoreCoord(MATMUL1_GRID_START_X, MATMUL1_GRID_START_Y),
            ttnn.CoreCoord(MATMUL1_GRID_END_X, MATMUL1_GRID_END_Y),
        )
        matmul1_core_grid = ttnn.CoreRangeSet([matmul1_grid])
        num_matmul1_cores = matmul1_grid.grid_size().x * matmul1_grid.grid_size().y  # 64

        # Gather/CCL receiver core: (12, 9)
        gather_core = ttnn.CoreCoord(12, 9)
        gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

        # CCL sender core: (11, 9) - adjacent to gather core
        ccl_sender_core = ttnn.CoreCoord(11, 9)
        ccl_sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ccl_sender_core, ccl_sender_core)])

        # Mcast grid: 13x10 = 130 cores (full rectangular for efficient mcast)
        MCAST_GRID_START_X = 0
        MCAST_GRID_START_Y = 0
        MCAST_GRID_END_X = 12  # 13 columns (0-12)
        MCAST_GRID_END_Y = 9  # 10 rows (0-9)
        mcast_grid = ttnn.CoreRange(
            ttnn.CoreCoord(MCAST_GRID_START_X, MCAST_GRID_START_Y),
            ttnn.CoreCoord(MCAST_GRID_END_X, MCAST_GRID_END_Y),
        )
        mcast_core_grid = ttnn.CoreRangeSet([mcast_grid])
        num_mcast_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y  # 130

        # Active Matmul2 cores: 112 cores (rows 0-8 full 12 cols + row 9 cols 0-3)
        matmul2_grid_main = ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(11, 8),  # 12 columns x 9 rows = 108 cores
        )
        matmul2_grid_extra = ttnn.CoreRange(
            ttnn.CoreCoord(0, 9),
            ttnn.CoreCoord(3, 9),  # 4 columns x 1 row = 4 cores
        )
        matmul2_active_core_grid = ttnn.CoreRangeSet([matmul2_grid_main, matmul2_grid_extra])
        num_matmul2_cores = 112  # 108 + 4 active cores

        # Gather2 sender grid bounds (for offset calculation, use bounding box)
        MATMUL2_GRID_START_X = 0
        MATMUL2_GRID_START_Y = 0
        MATMUL2_GRID_END_X = 11  # Same as mcast grid for offset calculation
        MATMUL2_GRID_END_Y = 9

        # Full grid (union of all cores for semaphore allocation)
        full_grid = matmul1_core_grid.merge(gather_core_grid).merge(mcast_core_grid).merge(ccl_sender_core_grid)

        # ========================================================================
        # SDPA Reduce-to-All configuration (optional)
        # ========================================================================
        sdpa_enabled = sdpa_input_l_mesh is not None
        if sdpa_enabled:
            # SDPA worker grid: 8 cores at (2,8)-(5,8), (2,9)-(5,9)
            sdpa_worker_grid = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(2, 8), ttnn.CoreCoord(5, 8)),  # 4 cores
                    ttnn.CoreRange(ttnn.CoreCoord(2, 9), ttnn.CoreCoord(5, 9)),  # 4 cores
                ]
            )
            num_sdpa_workers = 8

            # SDPA forwarder cores: (6,9), (7,9) - provided by caller or default
            # Both must be on row 9 with x > 3 to be outside matmul2 grid (same compile-time args)
            if sdpa_forwarder_cores is None:
                sdpa_forwarder_cores = [ttnn.CoreCoord(6, 9), ttnn.CoreCoord(7, 9)]
            sdpa_forwarder_grid = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(sdpa_forwarder_cores[0], sdpa_forwarder_cores[0]),
                    ttnn.CoreRange(sdpa_forwarder_cores[1], sdpa_forwarder_cores[1]),
                ]
            )

            # Add SDPA cores to full grid (workers and forwarders both part of unified kernel)
            full_grid = full_grid.merge(sdpa_worker_grid).merge(sdpa_forwarder_grid)

            # SDPA tensor properties - use same calculation as original sdpa_reduce_to_all op
            sdpa_input_l_sample = ttnn.get_device_tensors(sdpa_input_l_mesh)[0]
            sdpa_input_ms_sample = ttnn.get_device_tensors(sdpa_input_ms_mesh)[0]

            # Get actual tile dimensions from input tensor (matches original op)
            sdpa_tile = sdpa_input_l_sample.tile
            sdpa_tile_height, sdpa_tile_width = sdpa_tile.tile_shape
            sdpa_element_size_bytes = _get_element_size_bytes(sdpa_input_l_sample.dtype)
            sdpa_input_page_size_bytes = sdpa_element_size_bytes * sdpa_tile_height * sdpa_tile_width
            sdpa_l1_alignment = 16  # L1 alignment for SDPA (matches original op)

            # Get shard spec to calculate tile counts (matches original op)
            sdpa_shard_spec = sdpa_input_l_sample.memory_config().shard_spec
            sdpa_input_l_num_pages = (sdpa_shard_spec.shape[0] // sdpa_tile_height) * (
                sdpa_shard_spec.shape[1] // sdpa_tile_width
            )

            # Calculate out_tiles using same formula as original op
            PNH = 8
            DH = sdpa_input_l_num_pages * sdpa_tile_width
            DHt = DH // sdpa_tile_width
            PNHt = PNH // sdpa_tile_height
            Sq_chunk_t = PNHt
            sdpa_out_tiles = Sq_chunk_t * DHt

            # Chunking formula (identical to original op)
            sdpa_max_tiles_per_chunk = 8
            sdpa_min_num_l_chunks = (sdpa_out_tiles + sdpa_max_tiles_per_chunk - 1) // sdpa_max_tiles_per_chunk
            sdpa_num_l_chunks = max(sdpa_min_num_l_chunks, 4)
            if sdpa_out_tiles % sdpa_num_l_chunks != 0:
                raise ValueError(
                    f"sdpa_out_tiles ({sdpa_out_tiles}) must be divisible by sdpa_num_l_chunks ({sdpa_num_l_chunks})"
                )

            sdpa_tiles_per_l_chunk = sdpa_out_tiles // sdpa_num_l_chunks
            sdpa_l_chunk_size_bytes = sdpa_tiles_per_l_chunk * sdpa_input_page_size_bytes

            # Alias for backward compatibility with CB descriptor code
            sdpa_l_tiles_per_worker = sdpa_out_tiles

            # SDPA tile sizes (get from actual tensor, not hardcoded)
            sdpa_l_tile_size = sdpa_input_page_size_bytes  # Actual tile size from input
            sdpa_ms_tile_size = _round_up(sdpa_input_page_size_bytes, sdpa_l1_alignment)  # Aligned for MS

            # SDPA scatter parameters (scatter output to matmul1 cores)
            # Each SDPA worker scatters rows to matmul1 cores (one row per tile when using 8x32 tiles)
            sdpa_scatter_num_rows = sdpa_tile_height  # Rows per tile (8 for 8x32 tiles)
            sdpa_scatter_num_tiles = sdpa_l_tiles_per_worker  # Tiles per worker
            sdpa_scatter_src_tile_size = sdpa_l_tile_size  # Source tile size
            sdpa_scatter_dst_tile_size = tile_1x32_size  # 1x32 destination (row-extracted)
            # Face layout: each 8x32 tile has 2 faces of tile_height × (tile_width/2) = 8 × 16.
            # face_size = stride from Face 0 to Face 1 in source tile
            # row_face_size = one row within a face (also = dest face size for 1×32 dest tiles)
            sdpa_scatter_face_size = sdpa_tile_height * (sdpa_tile_width // 2) * sdpa_element_size_bytes
            sdpa_scatter_row_face_size = 1 * (sdpa_tile_width // 2) * sdpa_element_size_bytes  # dest_h=1 for 1x32

            # SDPA CB indices (14-24, after existing CBs 0-13)
            sdpa_cb_local_l = 14
            sdpa_cb_local_ms = 15
            sdpa_cb_r1_neighbor_l = 16
            sdpa_cb_r1_neighbor_ms = 17
            sdpa_cb_r1_result_l = 18
            sdpa_cb_r1_result_ms = 19
            sdpa_cb_r2_neighbor_l = 20
            sdpa_cb_r2_neighbor_ms = 21
            sdpa_cb_l_out = 22
            sdpa_cb_ms_out = 23
            sdpa_cb_packet_slot = 24

            # SDPA forwarder parameters (using Type A/B worker split like original op)
            sdpa_num_workers = 8
            sdpa_num_forwarders = 2
            sdpa_workers_per_forwarder = sdpa_num_workers // sdpa_num_forwarders  # 4
            sdpa_workers_per_type = sdpa_workers_per_forwarder // 2  # 2 (Type A and Type B alternate)
            sdpa_slots_per_worker = 1 + sdpa_num_l_chunks  # MS + L chunks = 5 slots
            sdpa_fwd_slots_per_round = sdpa_workers_per_type * sdpa_slots_per_worker  # 2 * 5 = 10 slots per direction
            sdpa_fwd_slot_size = 256 + sdpa_l_chunk_size_bytes  # Header + max payload
            sdpa_fwd_r2_buffer_offset = sdpa_fwd_slots_per_round * sdpa_fwd_slot_size

            # SDPA semaphore ID for scatter arrival (new semaphore)
            scatter_arrival_semaphore_id = 7  # After existing semaphores 0-6

            # SDPA forwarder semaphore IDs (on forwarder cores, signaled by workers)
            sdpa_fwd_r1_sem_id = 8
            sdpa_fwd_r2_sem_id = 9
            sdpa_bwd_r1_sem_id = 10
            sdpa_bwd_r2_sem_id = 11

            # SDPA global semaphores
            if sdpa_semaphores is not None:
                sdpa_semaphore1 = sdpa_semaphores[0]
                sdpa_semaphore2 = sdpa_semaphores[1]
                sdpa_semaphore1_addr = ttnn.get_global_semaphore_address(sdpa_semaphore1)
                sdpa_semaphore2_addr = ttnn.get_global_semaphore_address(sdpa_semaphore2)

            # Convert scale to FP32 bits
            import struct

            sdpa_scale_fp32_bits = struct.unpack(">I", struct.pack(">f", sdpa_scale_fp32))[0]

        # ========================================================================
        # Matmul1 parameters: [1, 512] x [512, 128] -> [1, 128]
        # ========================================================================
        matmul1_k_num_tiles = 16  # 512 / 32 = 16 tiles
        matmul1_out_w_per_core = 4  # 128 / 32 = 4 tiles per core

        # ========================================================================
        # Matmul2 parameters: [1, 8192] x [8192, 64] -> [1, 64]
        # ========================================================================
        matmul2_k_num_tiles = 256  # 8192 / 32 = 256 tiles
        matmul2_out_w_per_core = 2  # 64 / 32 = 2 tiles per core

        # ========================================================================
        # CB indices
        # ========================================================================
        matmul1_in0_cb = 0  # Matmul1 input (8x8)
        matmul1_in1_cb = 1  # Matmul1 weights (8x8)
        matmul1_out_cb = 2  # Matmul1 output (8x8)
        gather1_dst_cb = 3  # Gather1 output = Mcast source (gather core)
        matmul2_in0_cb = 4  # Mcast dst = Matmul2 input (13x10 mcast grid)
        matmul2_in1_cb = 5  # Matmul2 weights (112 active cores)
        matmul2_out_cb = 6  # Matmul2 output (112 active cores)
        gather2_dst_cb = 7  # Gather2 output = CCL local data (gather core)
        ccl_sender_in_cb = 8  # CCL sender reads gather2 output (sender core)
        ccl_remote_data_cb = 9  # CCL received remote data (receiver core)
        ccl_residual_cb = 10  # CCL residual (receiver core)
        ccl_temp_cb = 11  # CCL temp for compute (receiver core)
        ccl_output_cb = 12  # CCL output (receiver core)
        ccl_packet_header_cb = 13  # CCL packet headers (sender + receiver cores)

        # ========================================================================
        # Gather1 parameters: 64 cores -> [1, 8192]
        # ========================================================================
        gather1_data_size_bytes = matmul1_out_w_per_core * tile_1x32_size
        gather1_src_num_pages = matmul1_out_w_per_core  # 4 pages per sender
        gather1_dst_num_pages = num_matmul1_cores * matmul1_out_w_per_core  # 64 * 4 = 256 pages
        gather1_noc0_num_senders = num_matmul1_cores
        gather1_noc1_num_senders = 0

        # ========================================================================
        # Mcast parameters: [1, 8192] to 130 cores (13x10 grid)
        # ========================================================================
        mcast_data_size_bytes = gather1_dst_num_pages * tile_1x32_size  # 256 * 64 = 16384 bytes
        mcast_src_num_pages = gather1_dst_num_pages  # 256 pages
        mcast_dst_num_pages = gather1_dst_num_pages  # 256 pages per receiver

        # ========================================================================
        # Gather2 parameters: 112 cores -> [1, 7168]
        # ========================================================================
        gather2_data_size_bytes = matmul2_out_w_per_core * tile_1x32_size
        gather2_src_num_pages = matmul2_out_w_per_core  # 2 pages per sender
        gather2_dst_num_pages = num_matmul2_cores * matmul2_out_w_per_core  # 112 * 2 = 224 pages
        gather2_noc0_num_senders = num_matmul2_cores
        gather2_noc1_num_senders = 0

        # ========================================================================
        # CCL parameters: [1, 7168] all-reduce
        # ========================================================================
        # Using 1x32 tiles to match gather2 output format (for tile-compatible reduction)
        # 7168 elements = 224 tiles of 1x32 (32 elements each)
        ccl_num_tiles = gather2_dst_num_pages  # 224 tiles of 1x32
        ccl_page_size_bytes = tile_1x32_size  # 1x32 tile size
        ccl_num_pages = gather2_dst_num_pages  # 224 pages of 1x32
        ccl_payload_size_bytes = ccl_num_pages * ccl_page_size_bytes  # 224 * 64 = 14336 bytes
        ccl_packet_header_size_bytes = 32
        l1_alignment = 16

        has_residual = 1 if residual_tensor_mesh is not None else 0

        # ========================================================================
        # Semaphore IDs
        # ========================================================================
        gather1_noc0_receiver_semaphore_id = 0
        gather1_noc1_receiver_semaphore_id = 1
        mcast_data_sender_semaphore_id = 2
        mcast_data_receiver_semaphore_id = 3
        gather2_noc0_receiver_semaphore_id = 4
        gather2_noc1_receiver_semaphore_id = 5
        gather2_completion_semaphore_id = 6  # Gather2 signals, CCL sender waits
        # CCL uses global semaphores (passed via runtime args)

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # For each device in the mesh, create appropriate program
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col

                # Ring index along the cluster axis
                ring_index = row if cluster_axis == 0 else col
                is_first_chip = ring_index == 0

                # Get the device's tensors
                input_tensor_device = input_tensors_per_device[device_idx]
                gather2_output_tensor_device = gather2_output_tensors_per_device[device_idx]
                if ccl_enabled:
                    output_tensor_device = output_tensors_per_device[device_idx]
                    intermediate_tensor_device = intermediate_tensors_per_device[device_idx]
                sdpa_kv_cache_buffer_device = (
                    sdpa_kv_cache_buffers_per_device[device_idx]
                    if sdpa_kv_cache_buffers_per_device is not None
                    else None
                )

                device = input_tensor_device.device()

                # Get NOC coordinates for this device
                gather_dest_noc_core = device.worker_core_from_logical_core(gather_core)
                mcast_dest_noc_start_core = device.worker_core_from_logical_core(mcast_grid.start)
                mcast_dest_noc_end_core = device.worker_core_from_logical_core(mcast_grid.end)
                ccl_sender_noc_core = device.worker_core_from_logical_core(ccl_sender_core)
                ccl_receiver_noc_core = gather_dest_noc_core  # Same as gather core

                # Determine CCL neighbor and semaphores based on position (only when CCL is enabled)
                if ccl_enabled:
                    ccl_sender_link = 0 if is_first_chip else 1
                    ccl_receiver_link = 1 if is_first_chip else 0
                    ccl_sender_semaphore_addr = ccl_semaphore1_addr if is_first_chip else ccl_semaphore2_addr
                    ccl_receiver_semaphore_addr = ccl_semaphore2_addr if is_first_chip else ccl_semaphore1_addr

                    # Calculate neighbor coordinate
                    if is_first_chip:
                        neighbor_row = row + 1 if cluster_axis == 0 else row
                        neighbor_col = col if cluster_axis == 0 else col + 1
                    else:
                        neighbor_row = row - 1 if cluster_axis == 0 else row
                        neighbor_col = col if cluster_axis == 0 else col - 1

                # Buffer addresses
                gather1_receiver_data_addr = gather1_output_tensor.buffer_address()
                # Gather2 writes to gather2_output_tensor, CCL reads from there and writes to output_tensor
                gather2_receiver_data_addr = gather2_output_tensor_device.buffer_address()

                mcast_is_part_of_receiver_grid = mcast_grid.contains(gather_core)

                # ========================================================================
                # NCRISC compile-time args
                # ========================================================================
                ncrisc_named_compile_time_args = [
                    # Matmul1
                    ("matmul1_in0", matmul1_in0_cb),
                    ("matmul1_in1", matmul1_in1_cb),
                    ("matmul1_out", matmul1_out_cb),
                    ("matmul1_k_num_tiles", matmul1_k_num_tiles),
                    ("matmul1_out_w_per_core", matmul1_out_w_per_core),
                    # Gather1 sender
                    ("gather1_dest_noc_x", gather_dest_noc_core.x),
                    ("gather1_dest_noc_y", gather_dest_noc_core.y),
                    ("gather1_data_size_bytes", gather1_data_size_bytes),
                    ("gather1_receiver_semaphore_id", gather1_noc0_receiver_semaphore_id),
                    ("gather1_src_cb", matmul1_out_cb),
                    ("gather1_src_num_pages", gather1_src_num_pages),
                    ("gather1_sender_grid_start_x", MATMUL1_GRID_START_X),
                    ("gather1_sender_grid_start_y", MATMUL1_GRID_START_Y),
                    ("gather1_sender_grid_end_x", MATMUL1_GRID_END_X),
                    ("gather1_sender_grid_end_y", MATMUL1_GRID_END_Y),
                    ("gather1_row_major", 1),
                    ("gather1_receiver_data_addr", gather1_receiver_data_addr),
                    # Mcast receiver
                    ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
                    ("mcast_dst_cb", matmul2_in0_cb),
                    ("mcast_dst_num_pages", mcast_dst_num_pages),
                    # Matmul2
                    ("matmul2_in0", matmul2_in0_cb),
                    ("matmul2_in1", matmul2_in1_cb),
                    ("matmul2_out", matmul2_out_cb),
                    ("matmul2_k_num_tiles", matmul2_k_num_tiles),
                    ("matmul2_out_w_per_core", matmul2_out_w_per_core),
                    # Gather2 sender
                    ("gather2_dest_noc_x", gather_dest_noc_core.x),
                    ("gather2_dest_noc_y", gather_dest_noc_core.y),
                    ("gather2_data_size_bytes", gather2_data_size_bytes),
                    ("gather2_receiver_semaphore_id", gather2_noc0_receiver_semaphore_id),
                    ("gather2_src_cb", matmul2_out_cb),
                    ("gather2_src_num_pages", gather2_src_num_pages),
                    ("gather2_sender_grid_start_x", MATMUL2_GRID_START_X),
                    ("gather2_sender_grid_start_y", MATMUL2_GRID_START_Y),
                    ("gather2_sender_grid_end_x", MATMUL2_GRID_END_X),
                    ("gather2_sender_grid_end_y", MATMUL2_GRID_END_Y),
                    ("gather2_row_major", 1),
                    ("gather2_receiver_data_addr", gather2_receiver_data_addr),
                    # CCL sender (NCRISC reads from gather core)
                    ("ccl_sender_cb0_id", ccl_sender_in_cb),
                    ("ccl_sender_num_tiles", ccl_num_pages),
                    ("ccl_sender_tensor_page_size", ccl_page_size_bytes),
                    ("ccl_sender_data_noc_x", ccl_receiver_noc_core.x),
                    ("ccl_sender_data_noc_y", ccl_receiver_noc_core.y),
                    ("ccl_sender_gather2_completion_semaphore_id", gather2_completion_semaphore_id),
                    # CCL receiver (NCRISC waits for remote data)
                    ("ccl_receiver_packet_header_cb_id", ccl_packet_header_cb),
                    ("ccl_receiver_cb_in1", ccl_remote_data_cb),
                    ("ccl_receiver_l1_alignment", l1_alignment),
                    ("ccl_receiver_cb_in2", gather2_dst_cb),  # Local data from gather2
                    ("ccl_receiver_remote_sender_noc_x", ccl_sender_noc_core.x),
                    ("ccl_receiver_remote_sender_noc_y", ccl_sender_noc_core.y),
                    ("ccl_receiver_num_standard_tiles", ccl_num_tiles),
                    ("ccl_receiver_cb_residual", ccl_residual_cb),
                    ("ccl_receiver_has_residual", has_residual),
                    ("ccl_receiver_skip_local_push", 1),  # Skip local push since gather2 already pushed to CB7
                ]

                # Add SDPA NCRISC compile-time args when enabled
                if sdpa_enabled:
                    ncrisc_named_compile_time_args.extend(
                        [
                            # SDPA CB indices
                            ("sdpa_cb_local_l", sdpa_cb_local_l),
                            ("sdpa_cb_local_ms", sdpa_cb_local_ms),
                            ("sdpa_cb_r1_neighbor_l", sdpa_cb_r1_neighbor_l),
                            ("sdpa_cb_r1_neighbor_ms", sdpa_cb_r1_neighbor_ms),
                            ("sdpa_cb_r2_neighbor_l", sdpa_cb_r2_neighbor_l),
                            ("sdpa_cb_r2_neighbor_ms", sdpa_cb_r2_neighbor_ms),
                            # SDPA tile/chunk sizes
                            ("sdpa_ms_tile_size_bytes", sdpa_ms_tile_size),
                            ("sdpa_l_chunk_size_bytes", sdpa_l_chunk_size_bytes),
                            ("sdpa_num_l_chunks", sdpa_num_l_chunks),
                            ("sdpa_tiles_per_l_chunk", sdpa_tiles_per_l_chunk),
                            # SDPA forwarder params
                            ("sdpa_fwd_slots_per_round", sdpa_fwd_slots_per_round),
                            ("sdpa_fwd_slot_size", sdpa_fwd_slot_size),
                            ("sdpa_fwd_r2_buffer_offset", sdpa_fwd_r2_buffer_offset),
                            # Scatter arrival semaphore
                            ("scatter_arrival_semaphore_id", scatter_arrival_semaphore_id),
                        ]
                    )

                # ========================================================================
                # BRISC compile-time args
                # ========================================================================
                brisc_named_compile_time_args = [
                    # Matmul1/2 (no-op)
                    ("matmul1_out", matmul1_out_cb),
                    ("matmul2_out", matmul2_out_cb),
                    # Gather1 receiver
                    ("gather1_noc0_num_senders", gather1_noc0_num_senders),
                    ("gather1_noc1_num_senders", gather1_noc1_num_senders),
                    ("gather1_noc0_receiver_semaphore_id", gather1_noc0_receiver_semaphore_id),
                    ("gather1_noc1_receiver_semaphore_id", gather1_noc1_receiver_semaphore_id),
                    ("gather1_dst_cb", gather1_dst_cb),
                    ("gather1_dst_num_pages", gather1_dst_num_pages),
                    # Mcast sender
                    ("mcast_dest_noc_start_x", mcast_dest_noc_start_core.x),
                    ("mcast_dest_noc_start_y", mcast_dest_noc_start_core.y),
                    ("mcast_dest_noc_end_x", mcast_dest_noc_end_core.x),
                    ("mcast_dest_noc_end_y", mcast_dest_noc_end_core.y),
                    ("mcast_num_cores", num_mcast_cores),
                    ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
                    ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
                    ("mcast_data_size_bytes", mcast_data_size_bytes),
                    ("mcast_src_cb", gather1_dst_cb),
                    ("mcast_src_num_pages", mcast_src_num_pages),
                    ("mcast_dst_cb", matmul2_in0_cb),
                    ("mcast_is_part_of_receiver_grid", mcast_is_part_of_receiver_grid),
                    # Gather2 receiver
                    ("gather2_noc0_num_senders", gather2_noc0_num_senders),
                    ("gather2_noc1_num_senders", gather2_noc1_num_senders),
                    ("gather2_noc0_receiver_semaphore_id", gather2_noc0_receiver_semaphore_id),
                    ("gather2_noc1_receiver_semaphore_id", gather2_noc1_receiver_semaphore_id),
                    ("gather2_dst_cb", gather2_dst_cb),
                    ("gather2_dst_num_pages", gather2_dst_num_pages),
                    # Gather2 completion signal for CCL sender synchronization
                    ("gather2_completion_semaphore_id", gather2_completion_semaphore_id),
                    ("ccl_sender_noc_x", ccl_sender_noc_core.x),
                    ("ccl_sender_noc_y", ccl_sender_noc_core.y),
                    # CCL sender (BRISC sends via fabric)
                    ("ccl_sender_packet_header_cb_id", ccl_packet_header_cb),
                    ("ccl_sender_packet_cb_id", ccl_sender_in_cb),
                    ("ccl_sender_l1_alignment", l1_alignment),
                    ("ccl_sender_input_num_tiles", ccl_num_pages),
                    ("ccl_sender_page_size_bytes", ccl_page_size_bytes),
                    ("ccl_sender_payload_size_bytes", ccl_payload_size_bytes),
                    ("ccl_sender_data_noc_x", ccl_receiver_noc_core.x),
                    ("ccl_sender_data_noc_y", ccl_receiver_noc_core.y),
                    ("ccl_sender_remote_receiver_noc_x", ccl_receiver_noc_core.x),
                    ("ccl_sender_remote_receiver_noc_y", ccl_receiver_noc_core.y),
                    ("ccl_sender_dst_num_hops", 1),
                    ("ccl_sender_num_connections", 1),
                ]

                # Add SDPA BRISC compile-time args when enabled
                if sdpa_enabled:
                    brisc_named_compile_time_args.extend(
                        [
                            # SDPA CB indices
                            ("sdpa_cb_local_l", sdpa_cb_local_l),
                            ("sdpa_cb_local_ms", sdpa_cb_local_ms),
                            ("sdpa_cb_r1_result_l", sdpa_cb_r1_result_l),
                            ("sdpa_cb_r1_result_ms", sdpa_cb_r1_result_ms),
                            ("sdpa_cb_packet_slot", sdpa_cb_packet_slot),
                            ("sdpa_cb_l_out", sdpa_cb_l_out),
                            # SDPA tile/chunk sizes
                            ("sdpa_ms_tile_size_bytes", sdpa_ms_tile_size),
                            ("sdpa_l_chunk_size_bytes", sdpa_l_chunk_size_bytes),
                            ("sdpa_num_l_chunks", sdpa_num_l_chunks),
                            ("sdpa_tiles_per_l_chunk", sdpa_tiles_per_l_chunk),
                            ("sdpa_l1_alignment", l1_alignment),
                            ("sdpa_page_size_bytes", sdpa_l_tile_size),
                            ("sdpa_slot_size", sdpa_fwd_slot_size),
                            # SDPA scatter params
                            ("sdpa_scatter_num_tiles", sdpa_scatter_num_tiles),
                            ("sdpa_scatter_src_tile_size", sdpa_scatter_src_tile_size),
                            ("sdpa_scatter_dst_tile_size", sdpa_scatter_dst_tile_size),
                            ("sdpa_scatter_face_size", sdpa_scatter_face_size),
                            ("sdpa_scatter_row_face_size", sdpa_scatter_row_face_size),
                            ("sdpa_scatter_num_rows", sdpa_scatter_num_rows),
                            ("scatter_arrival_semaphore_id", scatter_arrival_semaphore_id),
                            # SDPA forwarder params
                            ("sdpa_fwd_slots_per_round", sdpa_fwd_slots_per_round),
                            ("sdpa_fwd_slot_size", sdpa_fwd_slot_size),
                            ("sdpa_fwd_r2_buffer_offset", sdpa_fwd_r2_buffer_offset),
                        ]
                    )

                # ========================================================================
                # TRISC compile-time args
                # ========================================================================
                trisc_named_compile_time_args = [
                    # Matmul1
                    ("matmul1_in0", matmul1_in0_cb),
                    ("matmul1_in1", matmul1_in1_cb),
                    ("matmul1_out", matmul1_out_cb),
                    ("matmul1_k_num_tiles", matmul1_k_num_tiles),
                    ("matmul1_out_w_per_core", matmul1_out_w_per_core),
                    # Matmul2
                    ("matmul2_in0", matmul2_in0_cb),
                    ("matmul2_in1", matmul2_in1_cb),
                    ("matmul2_out", matmul2_out_cb),
                    ("matmul2_k_num_tiles", matmul2_k_num_tiles),
                    ("matmul2_out_w_per_core", matmul2_out_w_per_core),
                    # CCL receiver compute (reduction)
                    ("ccl_receiver_cb_in0", ccl_remote_data_cb),
                    ("ccl_receiver_cb_in1", gather2_dst_cb),  # Local data
                    ("ccl_receiver_cb_out0", ccl_output_cb),
                    ("ccl_receiver_cb_residual", ccl_residual_cb),
                    ("ccl_receiver_cb_temp", ccl_temp_cb),
                    ("ccl_receiver_has_residual", has_residual),
                    ("ccl_receiver_num_tiles", ccl_num_tiles),
                ]

                # Add SDPA TRISC compile-time args when enabled
                if sdpa_enabled:
                    trisc_named_compile_time_args.extend(
                        [
                            # SDPA CB indices
                            ("sdpa_cb_local_l", sdpa_cb_local_l),
                            ("sdpa_cb_local_ms", sdpa_cb_local_ms),
                            ("sdpa_cb_r1_neighbor_l", sdpa_cb_r1_neighbor_l),
                            ("sdpa_cb_r1_neighbor_ms", sdpa_cb_r1_neighbor_ms),
                            ("sdpa_cb_r1_result_l", sdpa_cb_r1_result_l),
                            ("sdpa_cb_r1_result_ms", sdpa_cb_r1_result_ms),
                            ("sdpa_cb_r2_neighbor_l", sdpa_cb_r2_neighbor_l),
                            ("sdpa_cb_r2_neighbor_ms", sdpa_cb_r2_neighbor_ms),
                            ("sdpa_cb_l_out", sdpa_cb_l_out),
                            ("sdpa_cb_ms_out", sdpa_cb_ms_out),
                            # SDPA compute params
                            ("sdpa_scale_fp32", sdpa_scale_fp32_bits),
                            ("sdpa_tiles_per_l_chunk", sdpa_tiles_per_l_chunk),
                            ("sdpa_num_l_chunks", sdpa_num_l_chunks),
                        ]
                    )

                # ========================================================================
                # Circular buffer descriptors
                # ========================================================================
                running_address_offset = 0

                # CB 0: Matmul1 input (from sharded tensor, 8x8 grid)
                matmul1_in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul1_in0_cb, input_tensor_device)

                # CB 1: Matmul1 weights (from sharded tensor, 8x8 grid)
                matmul1_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul1_in1_cb, weights1_tensor)

                # CB 2: Matmul1 output (4 tiles of 1x32 per core, 8x8 grid)
                # When kv_cache buffer is available, overlap into it. Otherwise standalone.
                matmul1_out_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                matmul1_out_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul1_out_cb,
                    data_format=data_format,
                    page_size=tile_1x32_size,
                    tile=matmul1_out_tile_descriptor,
                )
                if sdpa_kv_cache_buffer_device is not None:
                    matmul1_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        matmul1_out_cb,
                        sdpa_kv_cache_buffer_device,
                        address_offset=running_address_offset,
                        total_size=matmul1_out_w_per_core * tile_1x32_size,
                    )
                    matmul1_out_cb_descriptor.format_descriptors = [matmul1_out_cb_format]
                    running_address_offset += matmul1_out_cb_descriptor.total_size
                else:
                    matmul1_out_cb_descriptor = ttnn.CBDescriptor(
                        total_size=matmul1_out_w_per_core * tile_1x32_size,
                        core_ranges=matmul1_core_grid,
                        format_descriptors=[matmul1_out_cb_format],
                    )

                # CB 3: Gather1 output = Mcast source (from sharded tensor, gather core)
                gather1_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    gather1_dst_cb, gather1_output_tensor
                )

                # CB 4: Mcast destination = Matmul2 input (256 tiles of 1x32 per core)
                matmul2_in0_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul2_in0_cb,
                    data_format=data_format,
                    page_size=tile_1x32_size,
                    tile=matmul1_out_tile_descriptor,
                )
                matmul2_in0_cb_grid = mcast_core_grid.merge(gather_core_grid)
                if sdpa_kv_cache_buffer_device is not None:
                    matmul2_in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        matmul2_in0_cb,
                        sdpa_kv_cache_buffer_device,
                        address_offset=running_address_offset,
                        total_size=mcast_dst_num_pages * tile_1x32_size,
                    )
                    matmul2_in0_cb_descriptor.format_descriptors = [matmul2_in0_cb_format]
                    running_address_offset += matmul2_in0_cb_descriptor.total_size
                else:
                    matmul2_in0_cb_descriptor = ttnn.CBDescriptor(
                        total_size=mcast_dst_num_pages * tile_1x32_size,
                        core_ranges=matmul2_in0_cb_grid,
                        format_descriptors=[matmul2_in0_cb_format],
                    )

                # CB 5: Matmul2 weights (from sharded tensor, 112 active matmul2 cores)
                matmul2_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul2_in1_cb, weights2_tensor)

                # CB 6: Matmul2 output (2 tiles of 1x32 per core, 112 active matmul2 cores)
                matmul2_out_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul2_out_cb,
                    data_format=data_format,
                    page_size=tile_1x32_size,
                    tile=matmul1_out_tile_descriptor,
                )
                if sdpa_kv_cache_buffer_device is not None:
                    matmul2_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        matmul2_out_cb,
                        sdpa_kv_cache_buffer_device,
                        address_offset=running_address_offset,
                        total_size=matmul2_out_w_per_core * tile_1x32_size,
                    )
                    matmul2_out_cb_descriptor.format_descriptors = [matmul2_out_cb_format]
                    running_address_offset += matmul2_out_cb_descriptor.total_size
                else:
                    matmul2_out_cb_descriptor = ttnn.CBDescriptor(
                        total_size=matmul2_out_w_per_core * tile_1x32_size,
                        core_ranges=matmul2_active_core_grid,
                        format_descriptors=[matmul2_out_cb_format],
                    )

                # CB 7: Gather2 output = CCL local data (backed by tensor on gather core)
                # CCL sender reads from this tensor via NOC, not from local CB
                gather2_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    gather2_dst_cb, gather2_output_tensor_device
                )

                cb_list = [
                    matmul1_in0_cb_descriptor,
                    matmul1_in1_cb_descriptor,
                    matmul1_out_cb_descriptor,
                    gather1_dst_cb_descriptor,
                    matmul2_in0_cb_descriptor,
                    matmul2_in1_cb_descriptor,
                    matmul2_out_cb_descriptor,
                    gather2_dst_cb_descriptor,
                ]

                # CCL CBs (8-13): only when CCL is enabled
                if ccl_enabled:
                    # CB 8: CCL sender input (reads from gather2 output via NOC)
                    ccl_sender_in_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=ccl_sender_in_cb,
                        data_format=data_format,
                        page_size=tile_1x32_size,
                        tile=matmul1_out_tile_descriptor,
                    )
                    if sdpa_kv_cache_buffer_device is not None:
                        ccl_sender_in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                            ccl_sender_in_cb,
                            sdpa_kv_cache_buffer_device,
                            address_offset=running_address_offset,
                            total_size=ccl_num_pages * tile_1x32_size,
                        )
                        ccl_sender_in_cb_descriptor.format_descriptors = [ccl_sender_in_cb_format]
                        running_address_offset += ccl_sender_in_cb_descriptor.total_size
                    else:
                        ccl_sender_in_cb_descriptor = ttnn.CBDescriptor(
                            total_size=ccl_num_pages * tile_1x32_size,
                            core_ranges=ccl_sender_core_grid,
                            format_descriptors=[ccl_sender_in_cb_format],
                        )
                    cb_list.append(ccl_sender_in_cb_descriptor)

                    # CB 9: CCL remote data (backed by intermediate tensor with 1x32 tiles)
                    # The intermediate tensor is where the CCL sender writes remote data
                    ccl_remote_data_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        ccl_remote_data_cb, intermediate_tensor_device
                    )
                    ccl_remote_data_cb_descriptor.core_ranges = gather_core_grid
                    cb_list.append(ccl_remote_data_cb_descriptor)

                    # CB 10: CCL residual (optional, from sharded tensor)
                    if residual_tensor_mesh is not None:
                        ccl_residual_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                            ccl_residual_cb, residual_tensors_per_device[device_idx]
                        )
                        ccl_residual_cb_descriptor.core_ranges = gather_core_grid
                        ccl_residual_cb_descriptor.total_size = ccl_num_tiles * tile_1x32_size
                        ccl_residual_cb_descriptor.format_descriptors = [
                            ttnn.CBFormatDescriptor(
                                buffer_index=ccl_residual_cb,
                                data_format=data_format,
                                page_size=tile_1x32_size,
                                tile=matmul1_out_tile_descriptor,  # 1x32 tiles to match gather2
                            )
                        ]
                        cb_list.append(ccl_residual_cb_descriptor)

                        # CB 11: CCL temp scratch buffer (not backed by tensor)
                        ccl_temp_cb_format = ttnn.CBFormatDescriptor(
                            buffer_index=ccl_temp_cb,
                            data_format=data_format,
                            page_size=tile_1x32_size,
                            tile=matmul1_out_tile_descriptor,
                        )
                        if sdpa_kv_cache_buffer_device is not None:
                            ccl_temp_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                                ccl_temp_cb,
                                sdpa_kv_cache_buffer_device,
                                address_offset=running_address_offset,
                                total_size=ccl_num_tiles * tile_1x32_size,
                            )
                            ccl_temp_cb_descriptor.format_descriptors = [ccl_temp_cb_format]
                            running_address_offset += ccl_temp_cb_descriptor.total_size
                        else:
                            ccl_temp_cb_descriptor = ttnn.CBDescriptor(
                                total_size=ccl_num_tiles * tile_1x32_size,
                                core_ranges=gather_core_grid,
                                format_descriptors=[ccl_temp_cb_format],
                            )
                        cb_list.append(ccl_temp_cb_descriptor)

                    # CB 12: CCL output (from sharded tensor)
                    ccl_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        ccl_output_cb, output_tensor_device
                    )
                    ccl_output_cb_descriptor.core_ranges = gather_core_grid
                    cb_list.append(ccl_output_cb_descriptor)

                    # CB 13: CCL packet headers
                    ccl_packet_header_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=ccl_packet_header_cb,
                        data_format=ttnn.uint32,
                        page_size=ccl_packet_header_size_bytes,
                    )
                    if sdpa_kv_cache_buffer_device is not None:
                        ccl_packet_header_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                            ccl_packet_header_cb,
                            sdpa_kv_cache_buffer_device,
                            address_offset=running_address_offset,
                            total_size=2 * ccl_packet_header_size_bytes,
                        )
                        ccl_packet_header_cb_descriptor.format_descriptors = [ccl_packet_header_cb_format]
                        running_address_offset += ccl_packet_header_cb_descriptor.total_size
                    else:
                        ccl_packet_header_cb_descriptor = ttnn.CBDescriptor(
                            total_size=2 * ccl_packet_header_size_bytes,
                            core_ranges=gather_core_grid.merge(ccl_sender_core_grid),
                            format_descriptors=[ccl_packet_header_cb_format],
                        )
                    cb_list.append(ccl_packet_header_cb_descriptor)
                # SDPA CBs (14-24): only when SDPA is enabled
                if sdpa_enabled:
                    # Get per-device SDPA tensors
                    sdpa_input_l_device = ttnn.get_device_tensors(sdpa_input_l_mesh)[device_idx]
                    sdpa_input_ms_device = ttnn.get_device_tensors(sdpa_input_ms_mesh)[device_idx]
                    sdpa_output_l_device = ttnn.get_device_tensors(sdpa_output_l_mesh)[device_idx]
                    sdpa_r1_recv_device = ttnn.get_device_tensors(sdpa_r1_recv_mesh)[device_idx]
                    sdpa_r2_recv_device = ttnn.get_device_tensors(sdpa_r2_recv_mesh)[device_idx]

                    # CB 14: SDPA local L (aliased to input tensor)
                    sdpa_local_l_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        sdpa_cb_local_l, sdpa_input_l_device
                    )
                    cb_list.append(sdpa_local_l_cb_descriptor)

                    # CB 15: SDPA local MS (aliased to input tensor)
                    sdpa_local_ms_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        sdpa_cb_local_ms, sdpa_input_ms_device
                    )
                    cb_list.append(sdpa_local_ms_cb_descriptor)

                    # CB 16: SDPA R1 neighbor L (aliased to R1 recv buffer)
                    # The recv buffer holds both L and MS data, but this CB should only
                    # cover the L portion. Override total_size like the original op does.
                    sdpa_r1_neighbor_l_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        sdpa_cb_r1_neighbor_l, sdpa_r1_recv_device
                    )
                    sdpa_r1_neighbor_l_cb_descriptor.total_size = sdpa_l_tiles_per_worker * sdpa_l_tile_size
                    cb_list.append(sdpa_r1_neighbor_l_cb_descriptor)

                    # CB 17: SDPA R1 neighbor MS (scratch, not backed by tensor)
                    # Must use sdpa_tile (e.g., 8x32) to match MS input tensor tile format
                    sdpa_r1_neighbor_ms_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=sdpa_cb_r1_neighbor_ms,
                        data_format=data_format,
                        page_size=sdpa_ms_tile_size,
                        tile=ttnn.TileDescriptor(sdpa_tile),
                    )
                    sdpa_r1_neighbor_ms_cb_descriptor = ttnn.CBDescriptor(
                        total_size=sdpa_ms_tile_size,
                        core_ranges=sdpa_worker_grid,
                        format_descriptors=[sdpa_r1_neighbor_ms_cb_format],
                    )
                    cb_list.append(sdpa_r1_neighbor_ms_cb_descriptor)

                    # CB 18: SDPA R1 result L (scratch, reused for scatter)
                    # Use actual tile from SDPA input tensor (e.g., 8x32), not hardcoded 32x32
                    sdpa_r1_result_l_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=sdpa_cb_r1_result_l,
                        data_format=data_format,
                        page_size=sdpa_l_tile_size,
                        tile=ttnn.TileDescriptor(sdpa_tile),
                    )
                    sdpa_r1_result_l_cb_descriptor = ttnn.CBDescriptor(
                        total_size=sdpa_l_tiles_per_worker * sdpa_l_tile_size,
                        core_ranges=sdpa_worker_grid,
                        format_descriptors=[sdpa_r1_result_l_cb_format],
                    )
                    cb_list.append(sdpa_r1_result_l_cb_descriptor)

                    # CB 19: SDPA R1 result MS (scratch)
                    # Must use sdpa_tile to match MS input tensor tile format
                    sdpa_r1_result_ms_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=sdpa_cb_r1_result_ms,
                        data_format=data_format,
                        page_size=sdpa_ms_tile_size,
                        tile=ttnn.TileDescriptor(sdpa_tile),
                    )
                    sdpa_r1_result_ms_cb_descriptor = ttnn.CBDescriptor(
                        total_size=sdpa_ms_tile_size,
                        core_ranges=sdpa_worker_grid,
                        format_descriptors=[sdpa_r1_result_ms_cb_format],
                    )
                    cb_list.append(sdpa_r1_result_ms_cb_descriptor)

                    # CB 20: SDPA R2 neighbor L (aliased to R2 recv buffer)
                    # Same total_size override as R1 neighbor L - only cover L portion.
                    sdpa_r2_neighbor_l_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        sdpa_cb_r2_neighbor_l, sdpa_r2_recv_device
                    )
                    sdpa_r2_neighbor_l_cb_descriptor.total_size = sdpa_l_tiles_per_worker * sdpa_l_tile_size
                    cb_list.append(sdpa_r2_neighbor_l_cb_descriptor)

                    # CB 21: SDPA R2 neighbor MS (scratch)
                    # Must use sdpa_tile to match MS input tensor tile format
                    sdpa_r2_neighbor_ms_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=sdpa_cb_r2_neighbor_ms,
                        data_format=data_format,
                        page_size=sdpa_ms_tile_size,
                        tile=ttnn.TileDescriptor(sdpa_tile),
                    )
                    sdpa_r2_neighbor_ms_cb_descriptor = ttnn.CBDescriptor(
                        total_size=sdpa_ms_tile_size,
                        core_ranges=sdpa_worker_grid,
                        format_descriptors=[sdpa_r2_neighbor_ms_cb_format],
                    )
                    cb_list.append(sdpa_r2_neighbor_ms_cb_descriptor)

                    # CB 22: SDPA L output (aliased to output tensor)
                    sdpa_l_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        sdpa_cb_l_out, sdpa_output_l_device
                    )
                    cb_list.append(sdpa_l_out_cb_descriptor)

                    # CB 23: SDPA MS output (scratch, only used for R1 intermediate)
                    # Must use sdpa_tile to match MS input tensor tile format
                    sdpa_ms_out_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=sdpa_cb_ms_out,
                        data_format=data_format,
                        page_size=sdpa_ms_tile_size,
                        tile=ttnn.TileDescriptor(sdpa_tile),
                    )
                    sdpa_ms_out_cb_descriptor = ttnn.CBDescriptor(
                        total_size=sdpa_ms_tile_size,
                        core_ranges=sdpa_worker_grid,
                        format_descriptors=[sdpa_ms_out_cb_format],
                    )
                    cb_list.append(sdpa_ms_out_cb_descriptor)

                    # CB 24: SDPA packet slot (for fabric packet headers)
                    sdpa_packet_slot_cb_format = ttnn.CBFormatDescriptor(
                        buffer_index=sdpa_cb_packet_slot,
                        data_format=ttnn.uint32,
                        page_size=256,  # Packet header size
                    )
                    sdpa_packet_slot_cb_descriptor = ttnn.CBDescriptor(
                        total_size=256,
                        core_ranges=sdpa_worker_grid,
                        format_descriptors=[sdpa_packet_slot_cb_format],
                    )
                    cb_list.append(sdpa_packet_slot_cb_descriptor)

                # ========================================================================
                # Semaphore descriptors
                # ========================================================================
                gather1_noc0_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather1_noc0_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                gather1_noc1_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather1_noc1_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                mcast_sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=mcast_data_sender_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                mcast_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=mcast_data_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                gather2_noc0_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather2_noc0_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                gather2_noc1_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather2_noc1_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                semaphore_list = [
                    gather1_noc0_semaphore_descriptor,
                    gather1_noc1_semaphore_descriptor,
                    mcast_sender_semaphore_descriptor,
                    mcast_receiver_semaphore_descriptor,
                    gather2_noc0_semaphore_descriptor,
                    gather2_noc1_semaphore_descriptor,
                ]
                if ccl_enabled:
                    gather2_completion_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                        id=gather2_completion_semaphore_id,
                        core_ranges=full_grid,
                        initial_value=0,
                    )
                    semaphore_list.append(gather2_completion_semaphore_descriptor)

                # SDPA scatter arrival semaphore (for matmul1 cores to wait for scatter data)
                if sdpa_enabled:
                    scatter_arrival_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                        id=scatter_arrival_semaphore_id,
                        core_ranges=full_grid,
                        initial_value=0,
                    )
                    semaphore_list.append(scatter_arrival_semaphore_descriptor)

                    # SDPA forwarder semaphores (workers signal these to forwarders)
                    sdpa_forwarder_grid = ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(sdpa_forwarder_cores[0], sdpa_forwarder_cores[0]),
                            ttnn.CoreRange(sdpa_forwarder_cores[1], sdpa_forwarder_cores[1]),
                        ]
                    )
                    sdpa_fwd_r1_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                        id=sdpa_fwd_r1_sem_id,
                        core_ranges=sdpa_forwarder_grid,
                        initial_value=0,
                    )
                    sdpa_fwd_r2_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                        id=sdpa_fwd_r2_sem_id,
                        core_ranges=sdpa_forwarder_grid,
                        initial_value=0,
                    )
                    sdpa_bwd_r1_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                        id=sdpa_bwd_r1_sem_id,
                        core_ranges=sdpa_forwarder_grid,
                        initial_value=0,
                    )
                    sdpa_bwd_r2_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                        id=sdpa_bwd_r2_sem_id,
                        core_ranges=sdpa_forwarder_grid,
                        initial_value=0,
                    )
                    semaphore_list.extend(
                        [
                            sdpa_fwd_r1_semaphore_descriptor,
                            sdpa_fwd_r2_semaphore_descriptor,
                            sdpa_bwd_r1_semaphore_descriptor,
                            sdpa_bwd_r2_semaphore_descriptor,
                        ]
                    )

                # ========================================================================
                # Kernel defines and unified compile-time core descriptors
                # ========================================================================
                kernel_defines = []
                if not ccl_enabled:
                    kernel_defines.append(("SKIP_CCL", "1"))
                if not sdpa_enabled:
                    kernel_defines.append(("SKIP_SDPA", "1"))

                unified_compile_time_core_descriptors = [
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_matmul1_core",
                        core_range=matmul1_core_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_gather_receiver_core",
                        core_range=gather_core_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_matmul2_core",
                        core_range=matmul2_active_core_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_mcast_receiver_core",
                        core_range=mcast_core_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_ccl_sender_core",
                        core_range=ccl_sender_core_grid,
                        value=1 if ccl_enabled else 0,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_ccl_receiver_core",
                        core_range=gather_core_grid,  # CCL receiver = gather core
                        value=1 if ccl_enabled else 0,
                        other_value=0,
                    ),
                ]

                # Add SDPA core descriptors when enabled (both workers and forwarders in unified kernel)
                if sdpa_enabled:
                    unified_compile_time_core_descriptors.extend(
                        [
                            UnifiedCompileTimeCoreDescriptor(
                                named_compile_time_arg="is_sdpa_worker_core",
                                core_range=sdpa_worker_grid,
                                value=1,
                                other_value=0,
                            ),
                            UnifiedCompileTimeCoreDescriptor(
                                named_compile_time_arg="is_sdpa_forwarder_core",
                                core_range=sdpa_forwarder_grid,
                                value=1,
                                other_value=0,
                            ),
                        ]
                    )

                # ========================================================================
                # Kernel descriptor
                # ========================================================================
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/post_sdpa/kernels/post_sdpa_kernel.cpp",
                    core_ranges=full_grid,
                    ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
                    brisc_named_compile_time_args=brisc_named_compile_time_args,
                    trisc_named_compile_time_args=trisc_named_compile_time_args,
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False,
                        fp32_dest_acc_en=fp32_dest_acc_en,
                        dst_full_sync_en=fp32_dest_acc_en,
                    ),
                    defines=kernel_defines,
                    unified_compile_time_core_descriptors=unified_compile_time_core_descriptors,
                )

                # Get kernel descriptors
                kernel_result = unified_kernel.get_kernel_descriptors()

                # ========================================================================
                # Program descriptor
                # ========================================================================
                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    cbs=cb_list,
                    semaphores=semaphore_list,
                )

                # ========================================================================
                # CCL runtime args and fabric (only when CCL is enabled)
                # ========================================================================
                if ccl_enabled:
                    # Get kernel indices for runtime args
                    ccl_sender_group = kernel_result.get_group_by_arg("is_ccl_sender_core", 1)
                    ccl_receiver_group = kernel_result.get_group_by_arg("is_ccl_receiver_core", 1)

                    # === COMMON RUNTIME ARGS ===
                    # Sender NCRISC common runtime args (arg 0)
                    ccl_sender_ncrisc_common_rt_args = [
                        gather2_receiver_data_addr,  # tensor_address
                    ]

                    # Sender BRISC common runtime args (args 0-1)
                    ccl_sender_brisc_common_rt_args = [
                        intermediate_tensor_device.buffer_address(),  # receiver_base_address
                        ccl_sender_semaphore_addr,  # receive_semaphore_addr
                    ]

                    # Receiver NCRISC common runtime args (arg 0)
                    ccl_receiver_ncrisc_common_rt_args = [
                        ccl_receiver_semaphore_addr,  # sender_semaphore_addr
                    ]

                    # === PER-CORE RUNTIME ARGS (empty, fabric args appended later) ===
                    ccl_sender_ncrisc_rt_args = ttnn.RuntimeArgs()
                    ccl_sender_ncrisc_rt_args[ccl_sender_core.x][ccl_sender_core.y] = []

                    ccl_sender_brisc_rt_args = ttnn.RuntimeArgs()
                    ccl_sender_brisc_rt_args[ccl_sender_core.x][ccl_sender_core.y] = []

                    ccl_receiver_ncrisc_rt_args = ttnn.RuntimeArgs()
                    ccl_receiver_ncrisc_rt_args[gather_core.x][gather_core.y] = []

                    # Set runtime args and common runtime args for CCL kernels
                    program.kernels[ccl_sender_group.ncrisc_kernel_index].runtime_args = ccl_sender_ncrisc_rt_args
                    program.kernels[
                        ccl_sender_group.ncrisc_kernel_index
                    ].common_runtime_args = ccl_sender_ncrisc_common_rt_args
                    program.kernels[ccl_sender_group.brisc_kernel_index].runtime_args = ccl_sender_brisc_rt_args
                    program.kernels[
                        ccl_sender_group.brisc_kernel_index
                    ].common_runtime_args = ccl_sender_brisc_common_rt_args
                    program.kernels[ccl_receiver_group.ncrisc_kernel_index].runtime_args = ccl_receiver_ncrisc_rt_args
                    program.kernels[
                        ccl_receiver_group.ncrisc_kernel_index
                    ].common_runtime_args = ccl_receiver_ncrisc_common_rt_args

                    # ========================================================================
                    # Fabric connection setup
                    # ========================================================================
                    fabric_node_id = mesh_device.get_fabric_node_id(coord)
                    neighbor_coord = ttnn.MeshCoordinate(neighbor_row, neighbor_col)
                    neighbor_fabric_node_id = mesh_device.get_fabric_node_id(neighbor_coord)
                    # Setup sender fabric connection
                    sender_brisc_kernel_idx = ccl_sender_group.brisc_kernel_index
                    sender_brisc_rt_args_ref = program.kernels[sender_brisc_kernel_idx].runtime_args[ccl_sender_core.x][
                        ccl_sender_core.y
                    ]
                    sender_fabric_args = ttnn.setup_routing_plane_connection(
                        fabric_node_id,
                        [neighbor_fabric_node_id],
                        [ccl_sender_link],
                        program,
                        sender_brisc_kernel_idx,
                        ccl_sender_core,
                    )
                    sender_brisc_rt_args_ref.extend(sender_fabric_args)

                    # Setup receiver fabric connection
                    receiver_ncrisc_kernel_idx = ccl_receiver_group.ncrisc_kernel_index
                    receiver_ncrisc_rt_args_ref = program.kernels[receiver_ncrisc_kernel_idx].runtime_args[
                        gather_core.x
                    ][gather_core.y]
                    receiver_fabric_args = ttnn.setup_routing_plane_connection(
                        fabric_node_id,
                        [neighbor_fabric_node_id],
                        [ccl_receiver_link],
                        program,
                        receiver_ncrisc_kernel_idx,
                        gather_core,
                    )
                    receiver_ncrisc_rt_args_ref.extend(receiver_fabric_args)

                # ========================================================================
                # SDPA runtime args and fabric (only when SDPA is enabled)
                # ========================================================================
                if sdpa_enabled:
                    # Get kernel groups for SDPA cores (both from unified kernel)
                    # sdpa_worker_group = kernel_result.get_group_by_arg("is_sdpa_worker_core", 1)
                    sdpa_forwarder_group = kernel_result.get_group_by_arg("is_sdpa_forwarder_core", 1)

                    # Get per-device SDPA tensors
                    sdpa_r1_recv_device = ttnn.get_device_tensors(sdpa_r1_recv_mesh)[device_idx]
                    sdpa_r2_recv_device = ttnn.get_device_tensors(sdpa_r2_recv_mesh)[device_idx]
                    sdpa_forwarder_scratch_device = ttnn.get_device_tensors(sdpa_forwarder_scratch_mesh)[device_idx]

                    # Get device for logical to NOC coordinate conversion
                    device = sdpa_r1_recv_device.device()

                    # Get fabric node IDs for SDPA CCL
                    sdpa_fabric_node_id = mesh_device.get_fabric_node_id(coord)

                    # Calculate neighbor coordinates for SDPA (forward and backward)
                    def get_sdpa_neighbor_coord(mesh_shape, row, col, direction, axis):
                        if axis == 0:
                            neighbor_row = (row + direction) % mesh_shape[0]
                            return neighbor_row, col
                        else:
                            neighbor_col = (col + direction) % mesh_shape[1]
                            return row, neighbor_col

                    fwd_row, fwd_col = get_sdpa_neighbor_coord(mesh_shape, row, col, +1, sdpa_cluster_axis)
                    bwd_row, bwd_col = get_sdpa_neighbor_coord(mesh_shape, row, col, -1, sdpa_cluster_axis)
                    fwd_coord = ttnn.MeshCoordinate(fwd_row, fwd_col)
                    bwd_coord = ttnn.MeshCoordinate(bwd_row, bwd_col)
                    fwd_fabric_node_id = mesh_device.get_fabric_node_id(fwd_coord)
                    bwd_fabric_node_id = mesh_device.get_fabric_node_id(bwd_coord)
                    # SDPA worker runtime args (per-core)
                    sdpa_worker_ncrisc_rt_args = ttnn.RuntimeArgs()
                    sdpa_worker_brisc_rt_args = ttnn.RuntimeArgs()

                    # Get matmul1 input buffer address for scatter destination
                    scatter_dest_l1_addr = input_tensor_device.buffer_address()

                    # Iterate over SDPA worker cores
                    sdpa_worker_cores = [
                        ttnn.CoreCoord(2, 8),
                        ttnn.CoreCoord(3, 8),
                        ttnn.CoreCoord(4, 8),
                        ttnn.CoreCoord(5, 8),
                        ttnn.CoreCoord(2, 9),
                        ttnn.CoreCoord(3, 9),
                        ttnn.CoreCoord(4, 9),
                        ttnn.CoreCoord(5, 9),
                    ]

                    # Type A/B worker split (like original sdpa_reduce_to_all op)
                    # This distributes R1/R2 traffic across both FWD and BWD forwarder instances
                    # Type A: R1 via FWD forwarder (BRISC), R2 via BWD forwarder (NCRISC)
                    # Type B: R1 via BWD forwarder (NCRISC), R2 via FWD forwarder (BRISC)
                    forwarder_buffer_base = sdpa_forwarder_scratch_device.buffer_address()
                    ncrisc_buffer_offset = sdpa_fwd_slots_per_round * sdpa_fwd_slot_size * 2  # After BRISC R1+R2

                    # Track slot assignments per direction per link
                    # Each link (forwarder pair) has FWD and BWD directions
                    # fwd_r1_count[link]: count of workers using FWD forwarder for R1
                    # etc.
                    fwd_r1_count = [0, 0]  # Per link
                    fwd_r2_count = [0, 0]
                    bwd_r1_count = [0, 0]
                    bwd_r2_count = [0, 0]

                    for worker_idx, worker_core in enumerate(sdpa_worker_cores):
                        # Convert worker core to NOC coordinates (like original sdpa_reduce_to_all op)
                        worker_core_noc = device.worker_core_from_logical_core(worker_core)

                        # NCRISC runtime args: semaphore addresses, recv buffer addresses
                        r1_neighbor_sem_addr = sdpa_semaphore1_addr
                        r2_neighbor_sem_addr = sdpa_semaphore2_addr
                        r1_recv_buffer_addr = sdpa_r1_recv_device.buffer_address()
                        r2_recv_buffer_addr = sdpa_r2_recv_device.buffer_address()

                        sdpa_worker_ncrisc_rt_args[worker_core.x][worker_core.y] = [
                            r1_neighbor_sem_addr,
                            r2_neighbor_sem_addr,
                            r1_recv_buffer_addr,
                            r2_recv_buffer_addr,
                        ]

                        # BRISC runtime args: fabric destinations, scatter destinations
                        # Determine which matmul1 cores this worker scatters to (8 rows per worker)
                        scatter_dest_noc_coords = []
                        for scatter_row in range(sdpa_scatter_num_rows):
                            # Map worker_idx and scatter_row to matmul1 core (logical coordinates)
                            matmul1_core_idx = worker_idx * sdpa_scatter_num_rows + scatter_row
                            matmul1_core_x = matmul1_core_idx % 8
                            matmul1_core_y = matmul1_core_idx // 8
                            # Convert to NOC coordinates
                            matmul1_core_logical = ttnn.CoreCoord(matmul1_core_x, matmul1_core_y)
                            matmul1_core_noc = device.worker_core_from_logical_core(matmul1_core_logical)
                            scatter_dest_noc_coords.append((matmul1_core_noc.x, matmul1_core_noc.y))

                        # Get forwarder core for this worker (link index)
                        link_idx = worker_idx // 4  # First 4 workers use forwarder 0, next 4 use forwarder 1
                        fwd_core = sdpa_forwarder_cores[link_idx]
                        # Convert forwarder core to NOC coordinates
                        fwd_core_noc = device.worker_core_from_logical_core(fwd_core)

                        # Type A/B determination (like original op)
                        is_type_a = ((row + worker_idx) % 2) == 0

                        # Type A: R1 via FWD (BRISC) to forward neighbor, R2 via BWD (NCRISC) to backward neighbor
                        # Type B: R1 via BWD (NCRISC) to backward neighbor, R2 via FWD (BRISC) to forward neighbor
                        if is_type_a:
                            # R1 config: FWD forwarder (BRISC buffer region) → forward neighbor
                            r1_fwd_buffer_base = forwarder_buffer_base
                            r1_fwd_sem_id = sdpa_fwd_r1_sem_id  # Forward R1 semaphore (ID 8)
                            r1_slot_idx = fwd_r1_count[link_idx] * sdpa_slots_per_worker
                            fwd_r1_count[link_idx] += 1
                            r1_fwd_slot_addr = r1_fwd_buffer_base + r1_slot_idx * sdpa_fwd_slot_size
                            r1_dst_fabric_node_id = fwd_fabric_node_id  # Type A sends R1 to forward neighbor
                            # R2 config: BWD forwarder (NCRISC buffer region) → backward neighbor
                            r2_fwd_buffer_base = forwarder_buffer_base + ncrisc_buffer_offset
                            r2_fwd_sem_id = sdpa_bwd_r2_sem_id  # Backward R2 semaphore (ID 11)
                            r2_slot_idx = bwd_r2_count[link_idx] * sdpa_slots_per_worker
                            bwd_r2_count[link_idx] += 1
                            r2_fwd_slot_addr = (
                                r2_fwd_buffer_base + sdpa_fwd_r2_buffer_offset + r2_slot_idx * sdpa_fwd_slot_size
                            )
                            r2_dst_fabric_node_id = bwd_fabric_node_id  # Type A sends R2 to backward neighbor
                        else:
                            # R1 config: BWD forwarder (NCRISC buffer region) → backward neighbor
                            r1_fwd_buffer_base = forwarder_buffer_base + ncrisc_buffer_offset
                            r1_fwd_sem_id = sdpa_bwd_r1_sem_id  # Backward R1 semaphore (ID 10)
                            r1_slot_idx = bwd_r1_count[link_idx] * sdpa_slots_per_worker
                            bwd_r1_count[link_idx] += 1
                            r1_fwd_slot_addr = r1_fwd_buffer_base + r1_slot_idx * sdpa_fwd_slot_size
                            r1_dst_fabric_node_id = bwd_fabric_node_id  # Type B sends R1 to backward neighbor
                            # R2 config: FWD forwarder (BRISC buffer region) → forward neighbor
                            r2_fwd_buffer_base = forwarder_buffer_base
                            r2_fwd_sem_id = sdpa_fwd_r2_sem_id  # Forward R2 semaphore (ID 9)
                            r2_slot_idx = fwd_r2_count[link_idx] * sdpa_slots_per_worker
                            fwd_r2_count[link_idx] += 1
                            r2_fwd_slot_addr = (
                                r2_fwd_buffer_base + sdpa_fwd_r2_buffer_offset + r2_slot_idx * sdpa_fwd_slot_size
                            )
                            r2_dst_fabric_node_id = fwd_fabric_node_id  # Type B sends R2 to forward neighbor

                        brisc_rt_args = [
                            int(r1_dst_fabric_node_id.mesh_id),  # r1_dst_mesh_id (varies by type!)
                            r1_dst_fabric_node_id.chip_id,  # r1_dst_chip_id
                            sdpa_r1_recv_device.buffer_address(),  # r1_neighbor_dst_addr
                            sdpa_semaphore1_addr,  # r1_neighbor_sem_addr
                            int(r2_dst_fabric_node_id.mesh_id),  # r2_dst_mesh_id (varies by type!)
                            r2_dst_fabric_node_id.chip_id,  # r2_dst_chip_id
                            sdpa_r2_recv_device.buffer_address(),  # r2_neighbor_dst_addr
                            sdpa_semaphore2_addr,  # r2_neighbor_sem_addr
                            worker_core_noc.x,  # current_core_x (NOC coordinates)
                            worker_core_noc.y,  # current_core_y (NOC coordinates)
                            fwd_core_noc.x,  # fwd_core_x (NOC coordinates)
                            fwd_core_noc.y,  # fwd_core_y (NOC coordinates)
                            r1_fwd_slot_addr,  # r1_fwd_slot_addr
                            r1_fwd_sem_id,  # r1_fwd_sem_id
                            r1_slot_idx,  # r1_base_slot_idx
                            r2_fwd_slot_addr,  # r2_fwd_slot_addr
                            r2_fwd_sem_id,  # r2_fwd_sem_id
                            r2_slot_idx,  # r2_base_slot_idx
                            scatter_dest_l1_addr,  # scatter_dest_l1_addr
                        ]
                        # Add scatter destination NOC coordinates
                        for noc_x, noc_y in scatter_dest_noc_coords:
                            brisc_rt_args.extend([noc_x, noc_y])

                        sdpa_worker_brisc_rt_args[worker_core.x][worker_core.y] = brisc_rt_args

                    # Set SDPA worker runtime args
                    # program.kernels[sdpa_worker_group.ncrisc_kernel_index].runtime_args = sdpa_worker_ncrisc_rt_args
                    # program.kernels[sdpa_worker_group.brisc_kernel_index].runtime_args = sdpa_worker_brisc_rt_args
                    for group in kernel_result.groups:
                        if group.compile_time_arg_values.get("is_sdpa_worker_core") == 1:
                            program.kernels[group.ncrisc_kernel_index].runtime_args = sdpa_worker_ncrisc_rt_args
                            program.kernels[group.brisc_kernel_index].runtime_args = sdpa_worker_brisc_rt_args

                    # SDPA forwarder runtime args (using setup_fabric_connection like original SDPA op)
                    # Key: Build RuntimeArgs completely FIRST, then assign to program.kernels AFTER
                    sdpa_forwarder_brisc_rt_args = ttnn.RuntimeArgs()
                    sdpa_forwarder_ncrisc_rt_args = ttnn.RuntimeArgs()

                    # forwarder_buffer_base and ncrisc_buffer_offset already defined above
                    # Use the new semaphore IDs (matching what we defined above)

                    for fwd_idx, fwd_core in enumerate(sdpa_forwarder_cores):
                        # BRISC handles FWD direction - set base args first
                        sdpa_forwarder_brisc_rt_args[fwd_core.x][fwd_core.y] = [
                            forwarder_buffer_base,
                            0,  # buffer_offset (BRISC uses first half)
                            sdpa_fwd_r1_sem_id,
                            sdpa_fwd_r2_sem_id,
                        ]
                        # Get fabric args using setup_fabric_connection (like original SDPA op)
                        brisc_fabric_args = ttnn.setup_fabric_connection(
                            src_fabric_node_id=sdpa_fabric_node_id,
                            dst_fabric_node_id=fwd_fabric_node_id,
                            link_idx=fwd_idx,
                            program_descriptor=program,
                            worker_core=fwd_core,
                        )
                        sdpa_forwarder_brisc_rt_args[fwd_core.x][fwd_core.y].extend(brisc_fabric_args)

                        # NCRISC handles BWD direction - set base args first
                        sdpa_forwarder_ncrisc_rt_args[fwd_core.x][fwd_core.y] = [
                            forwarder_buffer_base,
                            ncrisc_buffer_offset,  # buffer_offset (NCRISC uses second half)
                            sdpa_bwd_r1_sem_id,
                            sdpa_bwd_r2_sem_id,
                        ]
                        # Get fabric args using setup_fabric_connection (like original SDPA op)
                        ncrisc_fabric_args = ttnn.setup_fabric_connection(
                            src_fabric_node_id=sdpa_fabric_node_id,
                            dst_fabric_node_id=bwd_fabric_node_id,
                            link_idx=fwd_idx,
                            program_descriptor=program,
                            worker_core=fwd_core,
                        )
                        sdpa_forwarder_ncrisc_rt_args[fwd_core.x][fwd_core.y].extend(ncrisc_fabric_args)

                    # Assign to program.kernels AFTER all args are built (critical!)
                    program.kernels[sdpa_forwarder_group.brisc_kernel_index].runtime_args = sdpa_forwarder_brisc_rt_args
                    program.kernels[
                        sdpa_forwarder_group.ncrisc_kernel_index
                    ].runtime_args = sdpa_forwarder_ncrisc_rt_args
                    # Set for ALL kernel groups that have is_sdpa_forwarder_core=1
                    # for group in kernel_result.groups:
                    #    if group.compile_time_arg_values.get("is_sdpa_forwarder_core") == 1:
                    #        program.kernels[group.brisc_kernel_index].runtime_args = sdpa_forwarder_brisc_rt_args
                    #        program.kernels[group.ncrisc_kernel_index].runtime_args = sdpa_forwarder_ncrisc_rt_args

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute generic_op
        io_tensors = [
            input_tensor_mesh,
            weights1_tensor,
            weights2_tensor,
            gather1_output_tensor,
            gather2_output_tensor,
        ]
        if ccl_enabled:
            io_tensors.append(intermediate_tensor)
            io_tensors.append(output_tensor)
        if ccl_enabled and residual_tensor_mesh is not None:
            io_tensors.append(residual_tensor_mesh)
        if sdpa_kv_cache_buffer is not None:
            io_tensors.append(sdpa_kv_cache_buffer)
        if sdpa_enabled:
            io_tensors.extend(
                [
                    sdpa_input_l_mesh,
                    sdpa_input_ms_mesh,
                    sdpa_output_l_mesh,
                    sdpa_r1_recv_mesh,
                    sdpa_r2_recv_mesh,
                    sdpa_forwarder_scratch_mesh,
                ]
            )

        ttnn.generic_op(io_tensors, mesh_program_descriptor)

        return output_tensor if ccl_enabled else gather2_output_tensor
