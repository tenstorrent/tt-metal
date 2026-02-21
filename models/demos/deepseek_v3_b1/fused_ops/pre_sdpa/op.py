# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_uint32


class PreSDPA:
    """
    Pre-SDPA fused operation implementation using ttnn.generic_op.

    This class implements the pre-SDPA operations as a fused execution:
    - CCL Broadcast (optional, for mesh devices)
    - RMSNorm on a single core
    - Multicast of the result to a grid of cores
    - Matmul on grid of cores
    - Gather from grid to single core
    - RMSNorm2 on single core
    - Mcast2 to grid of cores
    - Matmul2 on grid of cores
    """

    @staticmethod
    def golden(
        input_tensor,
        gamma_tensor,
        matmul_weights_tensor,
        rmsnorm2_gamma_tensor,
        matmul2_weights_tensor,
        matmul3_weights_tensor,
        sin_tensor,
        cos_tensor,
        position_ids,
        dkv_matmul_weights_tensor,
        dkv_rmsnorm_gamma_tensor,
        epsilon=1e-6,
        num_qnope_heads=64,
        num_qrope_heads=64,
        qnope_head_dim=128,
        qrope_head_dim=64,
        heads_per_row=8,
        knope_dim=512,
        krope_dim=64,
    ):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) [1, K]
            gamma_tensor: Gamma/weight tensor (torch.Tensor) [1, K]
            matmul_weights_tensor: Matmul weights (torch.Tensor) [K, N]
            rmsnorm2_gamma_tensor: Gamma tensor for second RMSNorm (torch.Tensor) [1, N]
            matmul2_weights_tensor: Matmul2 weights (torch.Tensor) [N, M]
            matmul3_weights_tensor: Matmul3 weights (torch.Tensor) [num_qnope_heads, qnope_head_dim, qnope_out_dim]
                                    e.g., [64, 128, 512] for batched matmul on Qnope heads
            sin_tensor: Sin tensor (torch.Tensor) [max_seq_len, qrope_head_dim]
            cos_tensor: Cos tensor (torch.Tensor) [max_seq_len, qrope_head_dim]
            position_ids: Position indices (torch.Tensor) [batch] for decode mode
            epsilon: Small value to avoid division by zero
            num_qnope_heads: Number of Qnope heads (default 64)
            num_qrope_heads: Number of Qrope heads (default 64)
            qnope_head_dim: Dimension per Qnope head (default 128)
            qrope_head_dim: Dimension per Qrope head (default 64)
            heads_per_row: Number of heads per grid row (default 8)

        Returns:
            Tuple of (qnope_output, qrope_output, sdpa_interleaved):
            - qnope_output: [num_qnope_heads, 1, qnope_out_dim] after matmul3
            - qrope_output: [num_qrope_heads, 1, qrope_head_dim] after RoPE
            - sdpa_interleaved: [8, 8, 576] interleaved QNOPE/QROPE output for SDPA
        """
        from models.demos.deepseek_v3_b1.micro_ops.rope.op import RopeSingleCore

        def rmsnorm(x, gamma):
            variance = x.pow(2).mean(-1, keepdim=True)
            normalized = x * torch.rsqrt(variance + epsilon)
            return normalized * gamma

        # RMSNorm -> Matmul: [1, K] @ [K, N] -> [1, N]
        input_layernorm = rmsnorm(input_tensor, gamma_tensor)
        matmul_result = input_layernorm @ matmul_weights_tensor

        # RMSNorm2 -> Matmul2: [1, N] @ [N, M] -> [1, M]
        matmul2_result = rmsnorm(matmul_result, rmsnorm2_gamma_tensor) @ matmul2_weights_tensor

        qnope_heads = matmul2_result[:, : num_qnope_heads * qnope_head_dim].reshape(num_qnope_heads, 1, qnope_head_dim)
        qrope_heads = matmul2_result[:, num_qnope_heads * qnope_head_dim :].reshape(num_qrope_heads, 1, qrope_head_dim)

        # Matmul3: Batched matmul on Qnope heads
        # [64, 1, 128] @ [64, 128, 512] -> [64, 1, 512]
        qnope_output = torch.bmm(qnope_heads, matmul3_weights_tensor)

        # Apply RoPE to Qrope heads
        # qrope_heads: [num_qrope_heads, 1, qrope_head_dim] = [64, 1, 64]
        # Reshape for RopeSingleCore.golden: [batch, n_heads, seq_len, head_dim] = [1, 64, 1, 64]
        qrope_reshaped_for_rope = qrope_heads.permute(1, 0, 2).unsqueeze(0)  # [1, 64, 1, 64]
        # position_ids_expanded: [batch, seq_len] = [1, 1]
        position_ids_expanded = position_ids.unsqueeze(1)  # [batch, 1]
        # Apply RoPE
        qrope_output_reshaped = RopeSingleCore.golden(
            qrope_reshaped_for_rope, cos_tensor, sin_tensor, position_ids_expanded
        )
        # Reshape back: [1, 64, 1, 64] -> [64, 1, 64]
        qrope_output = qrope_output_reshaped.squeeze(0).permute(1, 0, 2)  # [64, 1, 64]

        # CreateQHeads: Combine QNOPE and QROPE outputs for SDPA
        # After 3-phase tilization, output is [64, 576] where each row is one head:
        #   row[i] = [qnope_first256[i], qnope_second256[i], qrope[i]] = [qnope[512], qrope[64]]
        # Heads are grouped by receiver core: 8 cores × 8 heads = 64 rows
        num_rows = num_qnope_heads // heads_per_row  # 8 receiver cores
        qnope_out_dim = qnope_output.shape[2]  # 512
        combined_head_dim = qnope_out_dim + qrope_head_dim  # 512 + 64 = 576

        # Reshape qnope_output: [64, 1, 512] -> [8, 8, 512]
        qnope_reshaped = qnope_output.squeeze(1).reshape(num_rows, heads_per_row, qnope_out_dim)

        # Reshape qrope_output: [64, 1, 64] -> [8, 8, 64]
        qrope_reshaped = qrope_output.squeeze(1).reshape(num_rows, heads_per_row, qrope_head_dim)

        # Build [8, 8, 576] then reshape to [64, 576] for tilized output format
        sdpa_interleaved = torch.zeros(num_rows, heads_per_row, combined_head_dim, dtype=qnope_output.dtype)
        sdpa_interleaved[:, :, :qnope_out_dim] = qnope_reshaped
        sdpa_interleaved[:, :, qnope_out_dim:] = qrope_reshaped
        # Reshape: [8, 8, 576] -> [64, 576] (each row = one head)
        sdpa_interleaved = sdpa_interleaved.reshape(num_rows * heads_per_row, combined_head_dim)

        # KV Cache Branch
        dkv = input_layernorm @ dkv_matmul_weights_tensor
        kv, k_rope = torch.split(dkv, [knope_dim, krope_dim], dim=-1)
        kv = rmsnorm(kv, dkv_rmsnorm_gamma_tensor)
        k_rope = RopeSingleCore.golden(k_rope, cos_tensor, sin_tensor, position_ids).squeeze(0)
        full_kv_cache_tensor = torch.cat([kv, k_rope], dim=-1)

        return qnope_output, qrope_output, sdpa_interleaved, full_kv_cache_tensor

    @staticmethod
    def op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        gamma_tensor,
        matmul_weights_tensor,
        rmsnorm2_gamma_tensor,
        matmul2_weights_tensor,
        matmul3_weights_tensor,
        qrope_sin_tensor,
        qrope_cos_tensor,
        trans_mat_tensor,
        krope_cos_tensor,
        krope_sin_tensor,
        dkv_matmul_weights_tensor,
        dkv_rmsnorm_gamma_tensor,
        kv_cache_tensor,
        position_id,
        position_ids_tensor,
        output_tensor,
        sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer,
        sender_coord,
        semaphores=None,
        cluster_axis=0,
        secondary_cluster_axis=1,
        num_links=1,
        epsilon=1e-6,
        fp32_dest_acc_en=False,
        skip_ccl=False,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
    ):
        """
        Execute pre-SDPA fused operation using generic_op.

        Args:
            input_tensor_mesh: Input mesh tensor (must be sharded on single core per device)
            intermediate_tensor_mesh: Intermediate mesh tensor for CCL broadcast destination
            gamma_tensor: Gamma/weight tensor (must be sharded, same shape as input)
            matmul_weights_tensor: Matmul weights tensor (must be width sharded)
            rmsnorm2_gamma_tensor: Gamma tensor for second RMSNorm (1536 elements = 3 tiles of 16x32)
            matmul2_weights_tensor: Matmul2 weights tensor (width sharded, shuffled for interleaved output)
            matmul3_weights_tensor: Matmul3 weights tensor (height sharded on Qnope grid, [128, 512] per core)
            qrope_sin_tensor: Sin tensor (sharded tensor for QRoPE)
            qrope_cos_tensor: Cos tensor (sharded tensor for QRoPE)
            trans_mat_tensor: Trans_mat tensor (sharded tensor for RoPE)
            position_ids_tensor: Position IDs tensor (sharded tensor for RoPE)
            output_tensor: Output tensor for pre-SDPA (sharded on SDPA grid, [8, 576] per core = 8 interleaved heads)
            sender_coord: Tuple (row, col) of sender device in mesh
            semaphores: List of global semaphores [out_ready, barrier, secondary_sync] for CCL
            cluster_axis: Primary axis for CCL broadcast (0=row, 1=col)
            secondary_cluster_axis: Secondary axis for CCL broadcast (optional)
            num_links: Number of fabric links for CCL
            epsilon: Small value to avoid division by zero
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel
            skip_ccl: If True, skip CCL broadcast (single-device mode)
            noc_mode: NOC mode for the kernel (dedicated or dynamic)

        Returns:
            output_tensor with pre-SDPA operations applied
        """
        sender_row = sender_coord[0]
        sender_col = sender_coord[1]

        # Get mesh/device info
        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        intermediate_tensors_per_device = ttnn.get_device_tensors(intermediate_tensor_mesh)
        gamma_tensors_per_device = ttnn.get_device_tensors(gamma_tensor)
        matmul_weights_tensors_per_device = ttnn.get_device_tensors(matmul_weights_tensor)
        rmsnorm2_gamma_tensors_per_device = ttnn.get_device_tensors(rmsnorm2_gamma_tensor)
        matmul2_weights_tensors_per_device = ttnn.get_device_tensors(matmul2_weights_tensor)
        matmul3_weights_tensors_per_device = ttnn.get_device_tensors(matmul3_weights_tensor)
        qrope_sin_tensors_per_device = ttnn.get_device_tensors(qrope_sin_tensor)
        qrope_cos_tensors_per_device = ttnn.get_device_tensors(qrope_cos_tensor)
        trans_mat_tensors_per_device = ttnn.get_device_tensors(trans_mat_tensor)
        krope_cos_tensors_per_device = ttnn.get_device_tensors(krope_cos_tensor)
        krope_sin_tensors_per_device = ttnn.get_device_tensors(krope_sin_tensor)
        position_ids_tensors_per_device = ttnn.get_device_tensors(position_ids_tensor)
        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)
        dkv_matmul_weights_tensors_per_device = ttnn.get_device_tensors(dkv_matmul_weights_tensor)
        dkv_rmsnorm_gamma_tensors_per_device = ttnn.get_device_tensors(dkv_rmsnorm_gamma_tensor)
        kv_cache_tensors_per_device = ttnn.get_device_tensors(kv_cache_tensor)
        sdpa_out_interm_buffers_per_device = ttnn.get_device_tensors(sdpa_out_interm_buffer)
        sdpa_kv_cache_buffers_per_device = ttnn.get_device_tensors(sdpa_kv_cache_buffer)

        # Semaphore addresses (only needed for CCL mode)
        out_ready_sem_addr = 0
        barrier_sem_addr = 0
        secondary_sync_sem_addr = 0
        if not skip_ccl and semaphores is not None:
            out_ready_semaphore = semaphores[0]
            barrier_semaphore = semaphores[1]
            secondary_sync_semaphore = semaphores[2]
            out_ready_sem_addr = ttnn.get_global_semaphore_address(out_ready_semaphore)
            barrier_sem_addr = ttnn.get_global_semaphore_address(barrier_semaphore)
            secondary_sync_sem_addr = ttnn.get_global_semaphore_address(secondary_sync_semaphore)

        # Calculate packet size and page info for CCL broadcast
        packet_size_bytes = 14336  # 14 KB packets for (1, 7168) input

        # Get tensor properties (use a sample device tensor)
        input_tensor_sample = input_tensors_per_device[0]
        input_shape = input_tensor_sample.shape
        data_format = input_tensor_sample.dtype

        # CCL broadcast page info
        element_size = 2
        tile_id_start = 0
        bcast_page_size_bytes = 32 * 32 * element_size  # interpret as 32x32 tile
        bcast_num_pages = input_shape[0] * input_shape[1] * element_size // bcast_page_size_bytes
        num_pages_per_packet = packet_size_bytes // bcast_page_size_bytes

        # Interpret N 1x32 tiles as full 32x32 or 16x32 tiles
        # eg. [1, 7168] = 7 full 32x32 tiles
        # eg. [1, 1536] = 3 half 16x32 tiles
        # eg. [1, 512] = 1 half 16x32 tile
        FULL_32x32_TILE = ttnn.Tile((32, 32))
        HALF_16x32_TILE = ttnn.Tile((16, 32))
        is_16x32_tile = (input_shape[1] // FULL_32x32_TILE.tile_shape[1]) % FULL_32x32_TILE.tile_shape[0] != 0
        interpreted_tile = HALF_16x32_TILE if is_16x32_tile else FULL_32x32_TILE
        tile_height, tile_width = interpreted_tile.tile_shape

        # Calculate single tile size in bytes (bfloat16 = 2 bytes per element)
        tile_size = interpreted_tile.get_tile_size(data_format)

        # Calculate num_tiles from tensor shape
        num_tiles = (input_shape[0] * input_shape[1]) // (tile_height * tile_width)

        # Get number of elements for RMS calculation
        numel = input_tensor_sample.logical_volume()

        # Get core grid from input tensor's memory config
        input_memory_config = input_tensor_sample.memory_config()
        input_core_grid = input_memory_config.shard_spec.grid
        input_core_ranges = list(input_core_grid.ranges())
        rmsnorm_core = input_core_ranges[0].start
        rmsnorm_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(rmsnorm_core, rmsnorm_core)])

        # Get full device grid
        device = input_tensor_sample.device()
        device_grid_size = device.compute_with_storage_grid_size()
        full_device_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )

        # Matmul1 grid from weight shard spec (single packed weight tensor)
        matmul_weights_sample = matmul_weights_tensors_per_device[0]
        matmul_weights_memory_config = matmul_weights_sample.memory_config()
        matmul_weights_core_grid = matmul_weights_memory_config.shard_spec.grid
        if len(list(matmul_weights_core_grid.ranges())) != 1:
            raise ValueError("matmul weights core grid must be a single rectangular range for packed K-split")
        if matmul_weights_memory_config.shard_spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
            raise ValueError("matmul weights shard orientation must be ROW_MAJOR for packed K-split")
        matmul_bbox = matmul_weights_core_grid.bounding_box()
        matmul_grid_size = matmul_bbox.grid_size()
        matmul_num_cores = matmul_grid_size.x * matmul_grid_size.y
        if matmul_weights_core_grid.num_cores() != matmul_num_cores:
            raise ValueError("matmul core grid must be a single rectangular range for this packed K-split path")
        if matmul_num_cores % 2 != 0:
            raise ValueError(f"matmul core grid must have an even number of cores, got {matmul_num_cores}")
        if matmul_num_cores != 96:
            raise ValueError(f"matmul core grid must have 96 cores for this K-split path, got {matmul_num_cores}")
        matmul_half_num_cores = matmul_num_cores // 2

        # Calculate per-core width in tiles for matmul1 (from shard spec)
        matmul_weights_tile = matmul_weights_sample.get_tile()
        matmul_weights_shard_shape = matmul_weights_memory_config.shard_spec.shape
        matmul_weights_shard_width = matmul_weights_shard_shape[1]  # Width dimension
        matmul_out_w = matmul_weights_shard_width // matmul_weights_tile.tile_shape[1]  # Per-core width in tiles

        # Calculate per-core width in tiles for matmul2 (from shard spec)
        matmul2_weights_sample = matmul2_weights_tensors_per_device[0]
        matmul2_weights_memory_config = matmul2_weights_sample.memory_config()
        matmul2_weights_core_grid = matmul2_weights_memory_config.shard_spec.grid
        matmul2_weights_tile = matmul2_weights_sample.get_tile()
        matmul2_weights_shard_shape = matmul2_weights_memory_config.shard_spec.shape
        matmul2_weights_shard_width = matmul2_weights_shard_shape[1]  # Width dimension
        matmul2_out_w = matmul2_weights_shard_width // matmul2_weights_tile.tile_shape[1]  # Per-core width in tiles

        # Extract matmul3 weights core grid (for inferring QNOPE grid dimensions)
        matmul3_weights_sample = matmul3_weights_tensors_per_device[0]
        matmul3_weights_memory_config = matmul3_weights_sample.memory_config()
        matmul3_weights_core_grid = matmul3_weights_memory_config.shard_spec.grid

        # ========================================================================
        # Qnope/Qrope grid configuration (for interleaved Q head layout)
        # With shuffled weights, matmul2 output is interleaved by row groups:
        # Each row has [8 Qnope heads (1024 elements)] [8 Qrope heads (512 elements)]
        # Qnope cores: columns 0-7 (8 cols), each core has 1 head × 128 elements
        # Qrope cores: columns 8-11 (4 cols), each core has 2 heads × 64 elements = 128 elements
        # Grid layout (8 rows × 12 cols = 96 cores for P150):
        #   Row 0: Qnope heads 0-7 (cols 0-7), Qrope heads 0-7 (cols 8-11)
        #   Row 1: Qnope heads 8-15 (cols 0-7), Qrope heads 8-15 (cols 8-11)
        #   ...
        #   Row 7: Qnope heads 56-63 (cols 0-7), Qrope heads 56-63 (cols 8-11)
        # ========================================================================
        # Get grid dimensions from first range (assuming contiguous rectangular grids)
        matmul2_grid_ranges = list(matmul2_weights_core_grid.ranges())
        matmul3_grid_ranges = list(matmul3_weights_core_grid.ranges())
        matmul2_grid_size = matmul2_grid_ranges[0].grid_size()
        matmul3_grid_size = matmul3_grid_ranges[0].grid_size()

        # Infer dimensions from grids
        HEAD_GRID_ROWS = matmul2_grid_size.y  # Number of rows (same for both grids)
        QNOPE_GRID_COLS = matmul3_grid_size.x  # QNOPE columns (from matmul3 grid width)
        QROPE_GRID_COLS = matmul2_grid_size.x - matmul3_grid_size.x  # QROPE columns (remaining columns)

        # Qnope grid: columns 0-7, rows 0-7 (64 cores total)
        qnope_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(QNOPE_GRID_COLS - 1, HEAD_GRID_ROWS - 1),
                )
            ]
        )

        # Qrope grid: columns 8-11, rows 0-7 (32 cores total for P150)
        # Note: For non-P150 with 11 columns, Qrope grid would be cols 8-10 (24 cores)
        qrope_grid_start_x = QNOPE_GRID_COLS  # Column 8
        qrope_grid_end_x = min(QNOPE_GRID_COLS + QROPE_GRID_COLS - 1, device_grid_size.x - 1)  # Column 11 for P150
        qrope_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(qrope_grid_start_x, 0),
                    ttnn.CoreCoord(qrope_grid_end_x, HEAD_GRID_ROWS - 1),
                )
            ]
        )

        # Krope grid: columns 8-9, rows 8-9 (2 cores total)
        krope_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(8, 8),
                    ttnn.CoreCoord(8, 9),
                )
            }
        )
        # Use the merged grids for certain shared CBs between Q rope and K rope
        qkv_grid = qrope_grid.merge(krope_grid)

        # ========================================================================
        # SDPA Input grid configuration (for receiving interleaved QNOPE/QROPE data)
        # SDPA Input cores: 4×2 grid (4 columns × 2 rows) at logical (0,1)-(3,2)
        # Each SDPA Input core receives 8 interleaved heads:
        #   - 8 QNOPE unicasts: [1, 512] each from the source row
        #   - 8 QROPE unicasts: [1, 64] each from the source row
        # Total per SDPA Input core: 8 × (512 + 64) = 8 × 576 = 4608 elements
        #
        # Mapping: source_row → target_core
        #   row 0 → (0, 1), row 1 → (1, 1), row 2 → (2, 1), row 3 → (3, 1)
        #   row 4 → (0, 2), row 5 → (1, 2), row 6 → (2, 2), row 7 → (3, 2)
        # Formula: target_x = row % 4, target_y = 1 + row // 4
        # ========================================================================
        SDPA_INPUT_GRID_START_X = 0
        SDPA_INPUT_GRID_START_Y = 1
        SDPA_INPUT_GRID_END_X = 3  # 4 columns: 0, 1, 2, 3
        SDPA_INPUT_GRID_END_Y = 2  # 2 rows: 1, 2
        sdpa_input_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(SDPA_INPUT_GRID_START_X, SDPA_INPUT_GRID_START_Y),
                    ttnn.CoreCoord(SDPA_INPUT_GRID_END_X, SDPA_INPUT_GRID_END_Y),
                )
            ]
        )

        # CreateQHeads parameters for 3-phase tilization layout
        COMBINED_HEAD_SIZE = 576  # 512 (QNOPE) + 64 (QROPE) elements per combined head
        QNOPE_DATA_SIZE = 512  # Elements per QNOPE head
        QROPE_HEAD_DIM = 64  # Elements per QROPE head
        QNOPE_COLS = 8  # Number of QNOPE sender columns
        QROPE_COLS = 4  # Number of QROPE sender columns

        # KV Cache Branch grid configuration
        # DKV Matmul (9x2)
        dkv_matmul_weights_sample = dkv_matmul_weights_tensors_per_device[0]
        dkv_matmul_weights_memory_config = dkv_matmul_weights_sample.memory_config()
        dkv_matmul_weights_core_grid = dkv_matmul_weights_memory_config.shard_spec.grid

        # Calculate per-core width in tiles for matmul (from shard spec)
        # Get shard width directly from shard_spec and divide by tile width from tensor
        dkv_matmul_weights_tile = dkv_matmul_weights_sample.get_tile()
        dkv_matmul_weights_shard_shape = dkv_matmul_weights_memory_config.shard_spec.shape
        dkv_matmul_weights_shard_width = dkv_matmul_weights_shard_shape[1]  # Width dimension
        dkv_matmul_out_w = (
            dkv_matmul_weights_shard_width // dkv_matmul_weights_tile.tile_shape[1]
        )  # Per-core width in tiles

        # ========================================================================
        # Mcast grid configuration (decoupled from matmul weights tensor)
        # Mcast to full logical grid; only matmul cores participate in receive
        # P150: (0,0)-(11,7), non-P150: (0,0)-(10,7)
        # ========================================================================
        MCAST_GRID_START_X = 0
        MCAST_GRID_START_Y = 0
        MCAST_GRID_END_X = device_grid_size.x - 1  # 11 for P150, 10 for non-P150
        MCAST_GRID_END_Y = 9
        main_grid = ttnn.CoreRange(
            ttnn.CoreCoord(MCAST_GRID_START_X, MCAST_GRID_START_Y),
            ttnn.CoreCoord(MCAST_GRID_END_X, MCAST_GRID_END_Y),
        )

        # Mcast setup: sender core (rmsnorm) -> full mcast grid
        # Only matmul cores (is_matmul_core=true) will actually participate in receive

        # Get NOC coordinates for mcast destination
        mcast_dest_noc_start_core = device.worker_core_from_logical_core(main_grid.start)
        mcast_dest_noc_end_core = device.worker_core_from_logical_core(main_grid.end)

        # Calculate number of mcast cores (full grid)
        mcast_num_cores = main_grid.grid_size().x * main_grid.grid_size().y
        mcast_is_part_of_receiver_grid = main_grid.contains(rmsnorm_core_grid)

        # Semaphore IDs for mcast synchronization
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1

        # Semaphore IDs for gather synchronization
        # Senders on NCRISC use NOC_0, receiver on BRISC uses NOC_1
        # Only use noc0 semaphore since senders are on NOC_0 (default for NCRISC)
        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3
        # Gather-reduce for matmul path reuses gather semaphore IDs
        gather_reduce_noc0_receiver_semaphore_id = gather_noc0_receiver_semaphore_id
        gather_reduce_noc1_receiver_semaphore_id = gather_noc1_receiver_semaphore_id

        # CreateQHeads 3-phase semaphore IDs (reuse existing IDs, safe since prior ops have completed)
        # Phase 1: QNOPE first halves, Phase 2: QNOPE second halves, Phase 3: QROPE
        nope_phase1_semaphore_id = gather_noc0_receiver_semaphore_id  # ID 2
        nope_phase2_semaphore_id = gather_noc1_receiver_semaphore_id  # ID 3
        rope_semaphore_id = mcast_data_sender_semaphore_id  # ID 0 (mcast completed before CreateQHeads)

        # Calculate mcast data size in bytes (RMSNorm output = num_tiles * tile_size)
        mcast_data_size_bytes = num_tiles * tile_size

        # Calculate runtime args
        epsilon_packed = float_to_uint32(epsilon)

        # Compute 1/sqrt(num_elements) for RMS reduction
        inv_sqrt_numel = 1.0 / math.sqrt(float(numel))
        scalar_packed = float_to_uint32(inv_sqrt_numel)
        kv_numel = 512
        kv_scalar_packed = float_to_uint32(1.0 / math.sqrt(float(kv_numel)))

        # Define circular buffer page size
        cb_page_size = tile_size

        # CB indices (grouped by stage)
        # CB indices for CCL broadcast (use separate CBs to avoid conflicts)
        bcast_pkt_cb = 35  # Packet buffer for CCL broadcast
        input_cb = 0
        gamma_cb = 1
        rmsnorm_output_cb = 2
        # Matmul1 + gather-reduce + RMSNorm2 path
        matmul_weights_cb = 3
        matmul_output_cb = 4
        matmul_input_cb = 5
        rmsnorm2_gamma_cb = 6  # Gamma for second RMSNorm (1536 elements = 3 tiles of 16x32)
        rmsnorm2_input_cb = 7  # Input CB for RMSNorm2
        gather_reduce_half1_scratch_cb = 8  # Dedicated half1 scratch CB for gather_reduce
        rmsnorm2_output_cb = 9  # Output CB for RMSNorm2
        # Matmul2 + Matmul3 + QRoPE/CreateQHeads path
        matmul2_input_cb = 10  # Input CB for second matmul (1x1536 with 1x32 tiles)
        matmul2_weights_cb = 11  # Weights CB for second matmul (width sharded, 4 tiles per core)
        matmul2_output_cb = 12  # Output CB for second matmul ([64, 1, 128] + [64, 1, 64])
        matmul3_weights_cb = 13  # Weights CB for third matmul (height sharded on Qnope grid)
        matmul3_output_cb = 14  # Output CB for third matmul (Qnope final output)
        qrope_output_cb = 15  # Output CB for Qrope (RoPE output)
        create_q_heads_out_cb = 16  # Output CB for CreateQHeads (linked to output tensor on receiver cores)
        qrope_cos_cb = 17  # Cos CB for RoPE
        qrope_sin_cb = 18  # Sin CB for RoPE
        qrope_trans_mat_cb = 19  # Trans_mat CB for RoPE
        qrope_rotated_input_interm_cb = 20  # Rotated input intermediate CB for RoPE
        qrope_cos_interm_cb = 21  # Cos intermediate CB for RoPE
        qrope_sin_interm_cb = 22  # Sin intermediate CB for RoPE
        # KV cache branch
        dkv_matmul_weights_cb = 23  # DKV Matmul weights CB
        dkv_matmul_output_cb = 24  # DKV Matmul output CB, 64 bytes (1 tile per core for rope input)
        kv_rmsnorm_input_cb = 25  # Input CB for KV Cache Branch RMSNorm
        kv_rmsnorm_gamma_cb = 26  # Gamma CB for KV Cache Branch RMSNorm
        kv_rmsnorm_output_cb = 27  # Output CB for KV Cache Branch RMSNorm
        krope_output_cb = 28  # Output CB for KV Cache Branch RoPE
        krope_cos_cb = 29  # Cos CB for RoPE
        krope_sin_cb = 30  # Sin CB for RoPE
        create_q_heads_receiver_in_cb = 31  # Intermediate CB for CreateQHeads (row-major data before tilization)

        kv_cache_output_cb = 32  # Output CB for KV Cache Branch
        kv_cache_intermed_cb = 33  # Intermed CB for KV Cache Branch
        kv_cache_input_cb = 34  # Input CB for KV Cache Branch

        # RMSNorm2 parameters (for 1536 element input using 16x32 tiles)
        rmsnorm2_numel = 1536
        rmsnorm2_num_tiles = 3  # 3 tiles of 16x32 = 3 * 512 = 1536 elements

        # Compute 1/sqrt(1536) for RMSNorm2 reduction
        inv_sqrt_rmsnorm2_numel = 1.0 / math.sqrt(float(rmsnorm2_numel))
        scalar2_packed = float_to_uint32(inv_sqrt_rmsnorm2_numel)

        # Matmul2 parameters
        # Input: RMSNorm2 output (1x1536 = 48 1x32 tiles)
        # Weights: width sharded with 4 tiles per core on the main grid
        # Grid: 8x12 = 96 cores (P150) or 8x11 = 88 cores (non-P150)
        matmul2_num_tiles_k = 48  # 1536 / 32 = 48 1x32 tiles

        # Mcast2 parameters (broadcasts rmsnorm2 output from input core to all matmul2 cores)
        # Reads from rmsnorm2_output_cb (3 tiles of 16x32), writes to matmul2_in0 (48 1x32 tiles) with loopback
        # Uses same grid and semaphores as first mcast
        mcast2_data_size_bytes = 1536 * 2  # 1536 bfloat16 elements = 3072 bytes
        mcast2_src_num_pages = rmsnorm2_num_tiles  # 3 tiles (rmsnorm2 output in 16x32 format)
        mcast2_dst_num_pages = matmul2_num_tiles_k  # 48 pages (destination uses 1x32 tiles)

        # Calculate mcast page counts for source and destination CBs
        # Source CB (rmsnorm_output): uses RMSNorm tile format (32x32 or 16x32)
        mcast_src_num_pages = num_tiles
        # Destination CB (matmul_input): uses 1x32 tile format
        TILE_1x32 = ttnn.Tile((1, 32))
        matmul_input_page_size = TILE_1x32.get_tile_size(data_format)
        matmul_input_total_size = num_tiles * cb_page_size  # Same total bytes as RMSNorm output
        mcast_dst_num_pages = matmul_input_total_size // matmul_input_page_size

        # KV Cache Branch parameters
        dkv_matmul_k_num_tiles = 7168 // 32
        dkv_matmul_input_page_size = TILE_1x32.get_tile_size(data_format)

        # RMSNorm reader compile-time args (named args for NCRISC)
        rmsnorm_reader_named_compile_time_args = [
            ("rmsnorm_input_cb", input_cb),
            ("rmsnorm_gamma_cb", gamma_cb),
            ("rmsnorm_num_tiles", num_tiles),
        ]

        # Mcast sender compile-time args (named args for BRISC)
        mcast_sender_named_compile_time_args = [
            ("mcast_dest_noc_start_x", mcast_dest_noc_start_core.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start_core.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end_core.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end_core.y),
            ("mcast_num_cores", mcast_num_cores),
            ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", rmsnorm_output_cb),
            ("mcast_dst_cb", matmul_input_cb),
            ("mcast_src_num_pages", mcast_src_num_pages),
            ("mcast_is_part_of_receiver_grid", mcast_is_part_of_receiver_grid),
        ]

        # Mcast receiver compile-time args (named args for NCRISC)
        mcast_receiver_named_compile_time_args = [
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_dst_cb", matmul_input_cb),
            ("mcast_dst_num_pages", mcast_dst_num_pages),
        ]

        # Calculate matmul1 K-split parameters
        # act_total_tiles = number of 1x32 tiles in full activation (same as mcast_dst_num_pages)
        matmul_act_total_tiles = mcast_dst_num_pages
        # Split K=7168 into 2 halves: each half has 112 tiles
        matmul_k_per_core = matmul_act_total_tiles // 2
        matmul_k_offset_half1 = matmul_k_per_core

        # Matmul compile-time args (different per RISC, only pass what's used)
        # NCRISC: in1, k_per_core (for setup_sharded_buffer)
        matmul_ncrisc_named_compile_time_args = [
            ("matmul_in1", matmul_weights_cb),
            ("matmul_k_per_core", matmul_k_per_core),
            ("matmul_out_w_per_core", matmul_out_w),
        ]
        # BRISC: out
        matmul_brisc_named_compile_time_args = [
            ("matmul_out", matmul_output_cb),
        ]
        # TRISC: KNSlicedMatmul args
        matmul_trisc_named_compile_time_args = [
            ("matmul_in0", matmul_input_cb),
            ("matmul_in1", matmul_weights_cb),
            ("matmul_out", matmul_output_cb),
            ("matmul_out_w_per_core", matmul_out_w),
            ("matmul_grid_start_x", matmul_bbox.start.x),
            ("matmul_grid_start_y", matmul_bbox.start.y),
            ("matmul_grid_end_x", matmul_bbox.end.x),
            ("matmul_grid_end_y", matmul_bbox.end.y),
            ("matmul_half_num_cores", matmul_half_num_cores),
            ("matmul_k_offset_half1", matmul_k_offset_half1),
            ("matmul_k_per_core", matmul_k_per_core),
            ("matmul_act_total_tiles", matmul_act_total_tiles),
        ]

        # Matmul2 compile-time args (different per RISC)
        # NCRISC: in1, num_tiles, rmsnorm2_output_cb (for copy to matmul2_input)
        matmul2_ncrisc_named_compile_time_args = [
            ("matmul2_in0", matmul2_input_cb),
            ("matmul2_in1", matmul2_weights_cb),
            ("matmul2_out", matmul2_output_cb),
            ("matmul2_k_num_tiles", matmul2_num_tiles_k),
            ("matmul2_out_w_per_core", matmul2_out_w),
        ]
        # BRISC: in0 (for mcast2 receiver), out, out_w (for Qrope copy)
        matmul2_brisc_named_compile_time_args = [
            ("matmul2_in0", matmul2_input_cb),
            ("matmul2_out", matmul2_output_cb),
            ("matmul2_out_w_per_core", matmul2_out_w),
        ]
        # TRISC: in0, in1, out, num_tiles, out_w_per_core
        matmul2_trisc_named_compile_time_args = [
            ("matmul2_in0", matmul2_input_cb),
            ("matmul2_in1", matmul2_weights_cb),
            ("matmul2_out", matmul2_output_cb),
            ("matmul2_k_num_tiles", matmul2_num_tiles_k),
            ("matmul2_out_w_per_core", matmul2_out_w),
        ]

        # ========================================================================
        # Matmul3 parameters (batched matmul on Qnope cores only)
        # Input: matmul2 output on Qnope cores [1, 128] = 4 tiles of 1x32 per core
        # Weights: [128, 512] per core, height sharded on Qnope grid (64 cores)
        # Output: [1, 512] = 16 tiles of 1x32 per core
        # ========================================================================
        matmul3_num_tiles_k = 4  # 128 / 32 = 4 tiles (input width)
        matmul3_weights_memory_config = matmul3_weights_sample.memory_config()
        matmul3_weights_tile = matmul3_weights_sample.get_tile()
        matmul3_weights_shard_shape = matmul3_weights_memory_config.shard_spec.shape
        matmul3_weights_shard_width = matmul3_weights_shard_shape[1]  # Width dimension (512)
        matmul3_out_w = matmul3_weights_shard_width // matmul3_weights_tile.tile_shape[1]  # 512/32 = 16 tiles

        # Matmul3 compile-time args (only on Qnope cores)
        # NCRISC: in1, num_tiles
        matmul3_ncrisc_named_compile_time_args = [
            ("matmul3_in0", matmul2_output_cb),  # Input from matmul2 output
            ("matmul3_in1", matmul3_weights_cb),
            ("matmul3_out", matmul3_output_cb),
            ("matmul3_k_num_tiles", matmul3_num_tiles_k),
            ("matmul3_out_w_per_core", matmul3_out_w),
        ]
        # BRISC: out
        matmul3_brisc_named_compile_time_args = [
            ("matmul3_in0", matmul2_output_cb),
            ("matmul3_out", matmul3_output_cb),
            ("qrope_output_cb", qrope_output_cb),
        ]
        # TRISC: in0, in1, out, num_tiles, out_w_per_core
        matmul3_trisc_named_compile_time_args = [
            ("matmul3_in0", matmul2_output_cb),  # Input from matmul2 output
            ("matmul3_in1", matmul3_weights_cb),
            ("matmul3_out", matmul3_output_cb),
            ("matmul3_k_num_tiles", matmul3_num_tiles_k),
            ("matmul3_out_w_per_core", matmul3_out_w),
        ]

        # Qrope head configuration: each qrope core processes 2 heads, each head is 64 elements (2 1x32 tiles)
        qrope_head_dim_per_core_t = 2  # head_dim (64) // TILE_SIZE (32) = 2 tiles per head
        qrope_num_heads_per_core = 2  # Each qrope core processes 2 heads

        # RoPE compile-time args (only on Qrope cores)
        qrope_rope_tile_size = TILE_1x32.get_tile_size(data_format)
        qrope_total_Wt = qrope_head_dim_per_core_t  # all cores read full head_dim, so total_Wt = Wt
        qrope_ncrisc_named_compile_time_args = [
            ("qrope_in_cb", matmul2_output_cb),
            ("qrope_cos_cb", qrope_cos_cb),
            ("qrope_sin_cb", qrope_sin_cb),
            ("qrope_trans_mat_cb", qrope_trans_mat_cb),
            ("qrope_Wt", qrope_head_dim_per_core_t),
            ("qrope_Ht", qrope_num_heads_per_core),
            ("qrope_cos_sin_page_size", qrope_rope_tile_size),
            ("qrope_total_Wt", qrope_total_Wt),
        ]
        # BRISC: no-op (empty args)
        qrope_brisc_named_compile_time_args = []
        # TRISC: in_cb, cos_cb, sin_cb, trans_mat_cb, rotated_in_interm_cb, cos_interm_cb, sin_interm_cb, out_cb, Wt, Ht
        qrope_trisc_named_compile_time_args = [
            ("qrope_in_cb", matmul2_output_cb),
            ("qrope_cos_cb", qrope_cos_cb),
            ("qrope_sin_cb", qrope_sin_cb),
            ("qrope_trans_mat_cb", qrope_trans_mat_cb),
            ("qrope_rotated_in_interm_cb", qrope_rotated_input_interm_cb),
            ("qrope_cos_interm_cb", qrope_cos_interm_cb),
            ("qrope_sin_interm_cb", qrope_sin_interm_cb),
            ("qrope_output_cb", qrope_output_cb),
            ("qrope_Wt", qrope_head_dim_per_core_t),
            ("qrope_Ht", qrope_num_heads_per_core),
        ]

        # ========================================================================
        # Unicast parameters (QNOPE/QROPE -> SDPA Input interleaved transfer)
        # QNOPE cores unicast [1, 512] = 16 tiles of 1x32 to SDPA Input
        # QROPE cores unicast [2, 64] = 2 heads × 2 tiles of 1x32 each to SDPA Input
        # Interleaved layout in SDPA Input: 8 × (512 + 64) = 8 × 576 = 4608 elements per core
        #
        # 4×2 grid mapping: source_row → target_core
        #   row 0 → (0, 1), row 1 → (1, 1), row 2 → (2, 1), row 3 → (3, 1)
        #   row 4 → (0, 2), row 5 → (1, 2), row 6 → (2, 2), row 7 → (3, 2)
        # ========================================================================
        # Get NOC coordinates for all SDPA Input cores (4×2 grid, indexed by source row)
        sdpa_input_noc_coords = []
        for src_row in range(HEAD_GRID_ROWS):
            # Mapping: target_x = row % 4, target_y = 1 + row // 4
            target_x = SDPA_INPUT_GRID_START_X + (src_row % 4)
            target_y = SDPA_INPUT_GRID_START_Y + (src_row // 4)
            sdpa_input_logical_core = ttnn.CoreCoord(target_x, target_y)
            sdpa_input_noc_core = device.worker_core_from_logical_core(sdpa_input_logical_core)
            sdpa_input_noc_coords.append((sdpa_input_noc_core.x, sdpa_input_noc_core.y))

        # Common unicast parameters
        head_stride_bytes = COMBINED_HEAD_SIZE * 2  # 576 * 2 = 1152 bytes (2 bytes per bfloat16 element)
        qnope_data_size_bytes = QNOPE_DATA_SIZE * 2  # 512 * 2 = 1024 bytes
        qrope_head_size_bytes = QROPE_HEAD_DIM * 2  # 64 * 2 = 128 bytes per head
        # Tilization parameters for 3-phase CreateQHeads
        nope_tiles = 8  # [8, 256] / [8, 32] = 8 tiles per NOPE phase
        rope_tiles = 2  # [8, 64] / [8, 32] = 2 tiles for ROPE phase

        # NCRISC sender compile-time args (QNOPE/QROPE -> SDPA Input) - matching gather pattern: NCRISC sender, BRISC receiver
        # 3-phase synchronization: senders write to intermediate CB, TRISC tilizes to output
        # Pack NOC coordinates for each row's target SDPA Input core (x in lower 16 bits, y in upper 16 bits)
        create_q_heads_ncrisc_named_compile_time_args = [
            # Packed coordinates (x | (y << 16)) for each source row's target
            ("cqh_target_noc_coords_row0", (sdpa_input_noc_coords[0][1] << 16 | sdpa_input_noc_coords[0][0])),
            ("cqh_target_noc_coords_row1", (sdpa_input_noc_coords[1][1] << 16 | sdpa_input_noc_coords[1][0])),
            ("cqh_target_noc_coords_row2", (sdpa_input_noc_coords[2][1] << 16 | sdpa_input_noc_coords[2][0])),
            ("cqh_target_noc_coords_row3", (sdpa_input_noc_coords[3][1] << 16 | sdpa_input_noc_coords[3][0])),
            ("cqh_target_noc_coords_row4", (sdpa_input_noc_coords[4][1] << 16 | sdpa_input_noc_coords[4][0])),
            ("cqh_target_noc_coords_row5", (sdpa_input_noc_coords[5][1] << 16 | sdpa_input_noc_coords[5][0])),
            ("cqh_target_noc_coords_row6", (sdpa_input_noc_coords[6][1] << 16 | sdpa_input_noc_coords[6][0])),
            ("cqh_target_noc_coords_row7", (sdpa_input_noc_coords[7][1] << 16 | sdpa_input_noc_coords[7][0])),
            ("cqh_head_stride_bytes", head_stride_bytes),
            ("cqh_qnope_data_size_bytes", qnope_data_size_bytes),
            ("cqh_qrope_head_size_bytes", qrope_head_size_bytes),
            # 3 semaphores for race-free synchronization
            ("cqh_nope_phase1_semaphore_id", nope_phase1_semaphore_id),
            ("cqh_nope_phase2_semaphore_id", nope_phase2_semaphore_id),
            ("cqh_rope_semaphore_id", rope_semaphore_id),
            ("cqh_qnope_src_cb", matmul3_output_cb),  # QNOPE sends from matmul3 output
            ("cqh_qrope_src_cb", qrope_output_cb),  # QROPE sends from qrope output
            ("cqh_qnope_src_num_pages", matmul3_out_w),  # 16 tiles of 1x32
            ("cqh_qrope_src_num_pages", matmul2_out_w),  # 4 tiles of 1x32 (2 heads × 2 tiles)
            ("cqh_qnope_cols", QNOPE_COLS),
            ("cqh_receiver_in_cb", create_q_heads_receiver_in_cb),  # Intermediate CB for row-major data
        ]

        # BRISC receiver compile-time args (SDPA Input cores) - matching gather pattern: NCRISC sender, BRISC receiver
        # 3-phase receiver: waits for each phase's semaphore, then marks pages in intermediate CB
        # Prefixed with "cqh_" to avoid name collisions with other BRISC args
        create_q_heads_brisc_named_compile_time_args = [
            ("cqh_nope_phase1_semaphore_id", nope_phase1_semaphore_id),
            ("cqh_nope_phase2_semaphore_id", nope_phase2_semaphore_id),
            ("cqh_rope_semaphore_id", rope_semaphore_id),
            ("cqh_num_nope_senders", QNOPE_COLS),  # 8 QNOPE senders per receiver
            ("cqh_num_rope_senders", QROPE_COLS),  # 4 QROPE senders per receiver
            ("cqh_receiver_in_cb", create_q_heads_receiver_in_cb),  # Intermediate CB
            ("cqh_out_cb", create_q_heads_out_cb),  # Output CB (backed by output tensor)
            ("cqh_nope_tiles", nope_tiles),  # 8 tiles per NOPE phase
            ("cqh_rope_tiles", rope_tiles),  # 2 tiles for ROPE phase
        ]

        # TRISC compute compile-time args (tilization on SDPA Input cores)
        # Prefixed with "cqh_" to avoid name collisions with other TRISC args (e.g., RoPE's "out_cb")
        create_q_heads_trisc_named_compile_time_args = [
            ("cqh_receiver_in_cb", create_q_heads_receiver_in_cb),
            ("cqh_out_cb", create_q_heads_out_cb),
            ("cqh_nope_tiles", nope_tiles),
            ("cqh_rope_tiles", rope_tiles),
        ]

        # RMSNorm compute compile-time args (named args for TRISC)
        rmsnorm_compute_named_compile_time_args = [
            ("rmsnorm_input_cb", input_cb),
            ("rmsnorm_gamma_cb", gamma_cb),
            ("rmsnorm_output_cb", rmsnorm_output_cb),
            ("rmsnorm_fp32_acc", 1 if fp32_dest_acc_en else 0),
            ("rmsnorm_num_tiles", num_tiles),
            ("rmsnorm_rsqrt_fast_approx", 0),
        ]

        # RMSNorm2 compile-time args (for second RMSNorm on gathered data)
        # Uses separate CBs with exact sizes for testing
        rmsnorm2_ncrisc_named_compile_time_args = [
            ("rmsnorm2_input_cb", rmsnorm2_input_cb),
            ("rmsnorm2_gamma_cb", rmsnorm2_gamma_cb),
            ("rmsnorm2_output_cb", rmsnorm2_output_cb),
            ("rmsnorm2_num_tiles", rmsnorm2_num_tiles),
        ]
        rmsnorm2_trisc_named_compile_time_args = [
            ("rmsnorm2_input_cb", rmsnorm2_input_cb),
            ("rmsnorm2_gamma_cb", rmsnorm2_gamma_cb),
            ("rmsnorm2_output_cb", rmsnorm2_output_cb),
            ("rmsnorm2_num_tiles", rmsnorm2_num_tiles),
        ]

        # ========================================================================
        # Gather-reduce setup: matmul cores (senders) -> rmsnorm core (receiver/reducer)
        # Sender runs on NCRISC (NOC_0 default), Receiver runs on BRISC (NOC_1 default)
        # ========================================================================
        gather_reduce_receiver_core = rmsnorm_core
        gather_reduce_sender_grid = matmul_weights_core_grid

        # Get NOC coordinates for gather destination (receiver core)
        gather_reduce_dest_noc_core = device.worker_core_from_logical_core(gather_reduce_receiver_core)

        # Calculate gather data size (matmul output size per core = 1 tile of 1x32)
        # Note: matmul_input_page_size == matmul_output_page_size (both are 1x32 tiles)
        gather_reduce_data_size_bytes = matmul_input_page_size

        # Get number of sender cores (matmul grid)
        gather_reduce_sender_cores_list = ttnn.corerange_to_cores(gather_reduce_sender_grid, row_wise=True)
        gather_reduce_num_senders = len(gather_reduce_sender_cores_list)

        # All senders use NOC_0 (default for NCRISC), so noc0_num_senders = all, noc1_num_senders = 0
        gather_reduce_noc0_num_senders = gather_reduce_num_senders
        gather_reduce_noc1_num_senders = 0

        # Gather sender compile-time args (named args for NCRISC on matmul cores)
        # SenderCTArgs: dest_noc_x, dest_noc_y, data_size_bytes, receiver_semaphore_id
        # Grid-based destination and sender index are computed in kernel from my_logical_x_/y_.
        gather_reduce_src_num_pages = 1  # Matmul output tiles per core (single 1x32 tile)
        gather_reduce_sender_named_compile_time_args = [
            ("gather_reduce_dest_noc_x", gather_reduce_dest_noc_core.x),
            ("gather_reduce_dest_noc_y", gather_reduce_dest_noc_core.y),
            ("gather_reduce_data_size_bytes", gather_reduce_data_size_bytes),
            ("gather_reduce_receiver_semaphore_id", gather_reduce_noc0_receiver_semaphore_id),
            ("gather_reduce_src_cb", matmul_output_cb),
            ("gather_reduce_src_num_pages", gather_reduce_src_num_pages),
            ("gather_reduce_grid_start_x", matmul_bbox.start.x),
            ("gather_reduce_grid_start_y", matmul_bbox.start.y),
            ("gather_reduce_grid_end_x", matmul_bbox.end.x),
            ("gather_reduce_grid_end_y", matmul_bbox.end.y),
            ("gather_reduce_half_num_cores", matmul_half_num_cores),
            ("gather_reduce_half0_cb_id", rmsnorm2_input_cb),
            ("gather_reduce_half1_cb_id", gather_reduce_half1_scratch_cb),
        ]

        # Gather receiver compile-time args (named args for BRISC on rmsnorm core)
        # ReceiverCTArgs: noc0_num_senders, noc1_num_senders, noc0_receiver_semaphore_id, noc1_receiver_semaphore_id
        # Plus destination CB info for reserve/push
        # Writes directly to rmsnorm2_input_cb (3 tiles of 16x32 = 3072 bytes)
        gather_reduce_receiver_named_compile_time_args = [
            ("gather_reduce_noc0_num_senders", gather_reduce_noc0_num_senders),
            ("gather_reduce_noc1_num_senders", gather_reduce_noc1_num_senders),
            ("gather_reduce_noc0_receiver_semaphore_id", gather_reduce_noc0_receiver_semaphore_id),
            ("gather_reduce_noc1_receiver_semaphore_id", gather_reduce_noc1_receiver_semaphore_id),
            ("gather_reduce_half0_dst_cb", rmsnorm2_input_cb),
            ("gather_reduce_half1_dst_cb", gather_reduce_half1_scratch_cb),
            ("gather_reduce_dst_num_tiles", rmsnorm2_num_tiles),
        ]
        # TRISC: compute-side gather-reduce destination CBs and tile count
        gather_reduce_trisc_named_compile_time_args = [
            ("gather_reduce_half0_dst_cb", rmsnorm2_input_cb),
            ("gather_reduce_half1_dst_cb", gather_reduce_half1_scratch_cb),
            ("gather_reduce_dst_num_tiles", rmsnorm2_num_tiles),
        ]

        # KV Cache Branch
        # DKV Matmul (9x2)
        dkv_matmul_ncrisc_named_compile_time_args = [
            ("dkv_matmul_in1", dkv_matmul_weights_cb),
            ("dkv_matmul_k_num_tiles", dkv_matmul_k_num_tiles),
            ("dkv_matmul_out_w_per_core", dkv_matmul_out_w),
        ]
        dkv_matmul_trisc_named_compile_time_args = [
            (
                "dkv_matmul_in0",
                matmul_input_cb,
            ),  # Inputs are multicasted from the main branch, same input as first matmul
            ("dkv_matmul_in1", dkv_matmul_weights_cb),
            ("dkv_matmul_out", dkv_matmul_output_cb),
            ("dkv_matmul_k_num_tiles", dkv_matmul_k_num_tiles),
            ("dkv_matmul_out_w_per_core", dkv_matmul_out_w),
        ]

        # KV Cache Branch: RMSNorm
        # RMSNorm compute compile-time args (named args for TRISC)
        kv_rmsnorm_num_tiles = kv_numel // (16 * 32)  # 512 / 512 = 1 tile (16x32)
        kv_rmsnorm_brisc_named_compile_time_args = [
            ("kv_rmsnorm_output_cb", kv_rmsnorm_output_cb),
            ("kv_rmsnorm_num_tiles", kv_rmsnorm_num_tiles),
        ]

        kv_rmsnorm_ncrisc_named_compile_time_args = [
            ("kv_rmsnorm_input_cb", kv_rmsnorm_input_cb),
            ("kv_rmsnorm_gamma_cb", kv_rmsnorm_gamma_cb),
            ("kv_rmsnorm_output_cb", kv_rmsnorm_output_cb),
            ("kv_rmsnorm_num_tiles", kv_rmsnorm_num_tiles),
        ]
        kv_rmsnorm_trisc_named_compile_time_args = [
            ("kv_rmsnorm_input_cb", kv_rmsnorm_input_cb),
            ("kv_rmsnorm_gamma_cb", kv_rmsnorm_gamma_cb),
            ("kv_rmsnorm_output_cb", kv_rmsnorm_output_cb),
            ("kv_rmsnorm_num_tiles", kv_rmsnorm_num_tiles),
        ]

        # ========================================================================
        # KV Cache Branch: Gather: dkv matmul cores (senders) -> rmsnorm core (receiver)
        # Sender runs on NCRISC (NOC_0 default), Receiver runs on BRISC (NOC_1 default)
        # ========================================================================
        dkv_rmsnorm_gamma_sample = dkv_rmsnorm_gamma_tensors_per_device[0]
        dkv_gather_receiver_core = dkv_rmsnorm_gamma_sample.memory_config().shard_spec.grid.ranges()[0].start
        dkv_gather_sender_grid = dkv_matmul_weights_core_grid.subtract(krope_grid)

        # Get NOC coordinates for gather destination (receiver core)
        dkv_gather_dest_noc_core = device.worker_core_from_logical_core(dkv_gather_receiver_core)

        # Get number of sender cores (matmul grid)
        dkv_gather_sender_cores_list = ttnn.corerange_to_cores(dkv_gather_sender_grid, row_wise=True)
        dkv_gather_num_senders = len(dkv_gather_sender_cores_list)

        # All senders use NOC_0 (default for NCRISC), so noc0_num_senders = all, noc1_num_senders = 0
        dkv_gather_noc0_num_senders = dkv_gather_num_senders
        dkv_gather_noc1_num_senders = 0

        # Get sender grid dimensions for computing per-core offset in kernel
        # Use logical coordinates since kernel uses UnifiedCoreDescriptor with my_logical_x_/y_
        dkv_gather_sender_grid_ranges = list(dkv_gather_sender_grid.ranges())
        dkv_gather_sender_grid_range = dkv_gather_sender_grid_ranges[0]
        dkv_gather_sender_grid_start_x = dkv_gather_sender_grid_range.start.x
        dkv_gather_sender_grid_start_y = dkv_gather_sender_grid_range.start.y
        dkv_gather_sender_grid_end_x = dkv_gather_sender_grid_range.end.x
        dkv_gather_sender_grid_end_y = dkv_gather_sender_grid_range.end.y

        # Gather sender compile-time args (named args for NCRISC on matmul cores)
        # SenderCTArgs: dest_noc_x, dest_noc_y, data_size_bytes, receiver_semaphore_id
        # Plus grid info for computing per-core offset
        dkv_gather_src_num_pages = dkv_matmul_out_w  # dkv matmul output tiles per core (must match matmul cb_push_back)
        dkv_gather_data_size_bytes = dkv_gather_src_num_pages * dkv_matmul_input_page_size
        dkv_gather_sender_named_compile_time_args = [
            ("dkv_gather_dest_noc_x", dkv_gather_dest_noc_core.x),
            ("dkv_gather_dest_noc_y", dkv_gather_dest_noc_core.y),
            ("dkv_gather_data_size_bytes", dkv_gather_data_size_bytes),
            ("dkv_gather_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("dkv_gather_src_cb", dkv_matmul_output_cb),  # Source CB for gather (dkv matmul output)
            ("dkv_gather_src_num_pages", dkv_gather_src_num_pages),
            ("dkv_gather_sender_grid_start_x", dkv_gather_sender_grid_start_x),
            ("dkv_gather_sender_grid_start_y", dkv_gather_sender_grid_start_y),
            ("dkv_gather_sender_grid_end_x", dkv_gather_sender_grid_end_x),
            ("dkv_gather_sender_grid_end_y", dkv_gather_sender_grid_end_y),
            ("dkv_gather_row_major", 1),  # 1 = row-major linearization
            ("dkv_gather_dst_cb", kv_rmsnorm_input_cb),  # Destination CB: write directly to kv_rmsnorm_input_cb
        ]

        # Gather receiver compile-time args (named args for BRISC on kv rmsnorm core)
        # ReceiverCTArgs: noc0_num_senders, noc1_num_senders, noc0_receiver_semaphore_id, noc1_receiver_semaphore_id
        # Plus destination CB info for reserve/push
        # Writes directly to kv_rmsnorm_input_cb
        dkv_gather_receiver_named_compile_time_args = [
            ("dkv_gather_noc0_num_senders", dkv_gather_noc0_num_senders),
            ("dkv_gather_noc1_num_senders", dkv_gather_noc1_num_senders),
            ("dkv_gather_noc0_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("dkv_gather_noc1_receiver_semaphore_id", gather_noc1_receiver_semaphore_id),
            ("dkv_gather_dst_cb", kv_rmsnorm_input_cb),
            ("dkv_gather_dst_num_pages", dkv_gather_src_num_pages),
        ]

        # KV Cache Branch: RoPE
        krope_rope_tile_size = TILE_1x32.get_tile_size(data_format)
        krope_Wt = 1
        krope_Ht = 1
        num_krope_cores = krope_grid.num_cores()
        krope_total_Wt = krope_Wt * num_krope_cores
        krope_ncrisc_named_compile_time_args = [
            ("krope_output_cb", krope_output_cb),
            ("krope_in_cb", dkv_matmul_output_cb),
            ("krope_cos_cb", krope_cos_cb),
            ("krope_sin_cb", krope_sin_cb),
            ("krope_trans_mat_cb", qrope_trans_mat_cb),
            ("krope_Wt", krope_Wt),
            ("krope_Ht", krope_Ht),
            ("krope_cos_sin_page_size", krope_rope_tile_size),
            ("krope_total_Wt", krope_total_Wt),
        ]
        krope_trisc_named_compile_time_args = [
            ("krope_in_cb", dkv_matmul_output_cb),
            ("krope_cos_cb", krope_cos_cb),
            ("krope_sin_cb", krope_sin_cb),
            ("krope_trans_mat_cb", qrope_trans_mat_cb),
            ("krope_rotated_in_interm_cb", qrope_rotated_input_interm_cb),
            ("krope_cos_interm_cb", qrope_cos_interm_cb),
            ("krope_sin_interm_cb", qrope_sin_interm_cb),
            ("krope_output_cb", krope_output_cb),
            ("krope_Wt", krope_Wt),
            ("krope_Ht", krope_Ht),
        ]

        # KVCacheUpdate CB indices and krope_Wt passed as runtime args (ReaderArgs/WriterArgs/ComputeArgs)
        kv_cache_brisc_named_compile_time_args = [
            ("krope_output_cb", krope_output_cb),
            ("kv_cache_output_cb", kv_cache_output_cb),
            ("kv_cache_input_cb", kv_cache_input_cb),
            ("kv_cache_intermed_cb", kv_cache_intermed_cb),
            ("kv_cache_grid_start_y", list(krope_grid.ranges())[0].start.y),
        ]
        kv_cache_trisc_named_compile_time_args = [
            ("kv_rmsnorm_output_cb", kv_rmsnorm_output_cb),
            ("kv_cache_output_cb", kv_cache_output_cb),
            ("kv_cache_input_cb", kv_cache_input_cb),
            ("kv_cache_intermed_cb", kv_cache_intermed_cb),
        ]

        # Create tile descriptor for proper tile dimensions
        tile_descriptor = ttnn.TileDescriptor(interpreted_tile)

        # RMSNorm2 uses separate CBs with exact sizes (16x32 tiles)
        TILE_16x32 = ttnn.Tile((16, 32))
        rmsnorm2_tile_descriptor = ttnn.TileDescriptor(TILE_16x32)
        rmsnorm2_page_size = TILE_16x32.get_tile_size(data_format)

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # ========================================================================
        # Kernel descriptors
        # ========================================================================

        for row in range(mesh_rows):
            # ("start of loop for device row {}".format(row))
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col

                # CCL role calculation (only matters if not skipping CCL)
                if skip_ccl:
                    is_sender = False
                    is_secondary_sender = False
                    is_receiver = False
                else:
                    is_sender = (row == sender_row) and (col == sender_col)
                    is_secondary_sender = (
                        secondary_cluster_axis is not None and (row == sender_row) and (col != sender_col)
                    )
                    is_receiver = not is_sender and not is_secondary_sender

                # Get the device's tensors
                input_tensor_device = input_tensors_per_device[device_idx]
                intermediate_tensor_device = intermediate_tensors_per_device[device_idx]
                gamma_tensor_device = gamma_tensors_per_device[device_idx]
                matmul_weights_tensor_device = matmul_weights_tensors_per_device[device_idx]
                rmsnorm2_gamma_tensor_device = rmsnorm2_gamma_tensors_per_device[device_idx]
                matmul2_weights_tensor_device = matmul2_weights_tensors_per_device[device_idx]
                matmul3_weights_tensor_device = matmul3_weights_tensors_per_device[device_idx]
                qrope_cos_tensor_device = qrope_cos_tensors_per_device[device_idx]
                qrope_sin_tensor_device = qrope_sin_tensors_per_device[device_idx]
                trans_mat_tensor_device = trans_mat_tensors_per_device[device_idx]
                output_tensor_device = output_tensors_per_device[device_idx]
                dkv_matmul_weights_tensor_device = dkv_matmul_weights_tensors_per_device[device_idx]
                dkv_rmsnorm_gamma_tensor_device = dkv_rmsnorm_gamma_tensors_per_device[device_idx]
                krope_cos_tensor_device = krope_cos_tensors_per_device[device_idx]
                krope_sin_tensor_device = krope_sin_tensors_per_device[device_idx]
                position_ids_tensor_device = position_ids_tensors_per_device[device_idx]
                kv_cache_tensor_device = kv_cache_tensors_per_device[device_idx]
                sdpa_kv_cache_buffer_device = sdpa_kv_cache_buffers_per_device[device_idx]
                sdpa_out_interm_buffer_device = sdpa_out_interm_buffers_per_device[device_idx]

                # Get worker core from per-device input tensor shard grid
                device_local = input_tensor_device.device()
                device_input_shard_grid = input_tensor_device.memory_config().shard_spec.grid
                device_shard_grid_start = device_input_shard_grid.bounding_box().start
                worker_core = ttnn.CoreCoord(device_shard_grid_start.x, device_shard_grid_start.y)
                worker_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(worker_core, worker_core)])
                assert rmsnorm_core_grid == worker_core_set, "RMSNorm core grid does not match worker core"

                # Get physical core for NOC addressing
                data_core_physical = device_local.worker_core_from_logical_core(worker_core)
                core_noc_x = data_core_physical.x
                core_noc_y = data_core_physical.y

                # Calculate ring index and targets for primary axis (column)
                ring_size = mesh_rows
                ring_index = row

                # For Linear topology, calculate forward and backward targets
                num_targets_forward = ring_size - ring_index - 1
                num_targets_backward = ring_index

                # Determine if this device has secondary axis connections
                has_secondary_target = is_sender and (mesh_cols > 1) and (secondary_cluster_axis is not None)

                # Calculate mcast distances
                start_distance_forward = 1 if num_targets_forward > 0 else 0
                range_hops_forward = num_targets_forward
                start_distance_backward = 1 if num_targets_backward > 0 else 0
                range_hops_backward = num_targets_backward
                bcast_num_pages_to_read = bcast_num_pages

                # ================================================================
                # CCL Broadcast compile-time args (per-device)
                # ================================================================
                bcast_brisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("bcast_cb0_id", bcast_pkt_cb if not skip_ccl else 0),
                    ("bcast_num_pages_to_read", bcast_num_pages_to_read if not skip_ccl else 0),
                    ("bcast_is_sender", int(is_sender) if not skip_ccl else 0),
                ]

                bcast_ncrisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("bcast_cb0_id", bcast_pkt_cb if not skip_ccl else 0),
                    ("bcast_num_pages_to_read", bcast_num_pages_to_read if not skip_ccl else 0),
                    ("bcast_tensor0_page_size", bcast_page_size_bytes if not skip_ccl else 0),
                    ("bcast_num_targets_forward_direction", num_targets_forward if not skip_ccl else 0),
                    ("bcast_num_targets_backward_direction", num_targets_backward if not skip_ccl else 0),
                    ("bcast_is_sender", int(is_sender) if not skip_ccl else 0),
                    ("bcast_core_noc_x", core_noc_x if not skip_ccl else 0),
                    ("bcast_core_noc_y", core_noc_y if not skip_ccl else 0),
                    ("bcast_is_secondary_sender", int(is_secondary_sender) if not skip_ccl else 0),
                    ("bcast_has_secondary_target", int(has_secondary_target) if not skip_ccl else 0),
                    ("bcast_start_distance_in_hops_forward", start_distance_forward if not skip_ccl else 0),
                    ("bcast_range_hops_forward", range_hops_forward if not skip_ccl else 0),
                    ("bcast_start_distance_in_hops_backward", start_distance_backward if not skip_ccl else 0),
                    ("bcast_range_hops_backward", range_hops_backward if not skip_ccl else 0),
                ]

                bcast_trisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                ]

                sdpa_out_interm_running_offset = 0

                # Create circular buffer descriptors
                # CB: Input (created from sharded tensor)
                cb0_backing_tensor = input_tensor_device if skip_ccl else intermediate_tensor_device
                in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, cb0_backing_tensor)
                # Update the tile descriptor in the format descriptor
                in_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                in_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # CB: Gamma (created from sharded tensor)
                gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gamma_cb, gamma_tensor_device)
                # Update the tile descriptor in the format descriptor
                gamma_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                gamma_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # CB: RMSNorm2 Gamma (created from sharded tensor, 3 tiles of 16x32)
                rmsnorm2_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    rmsnorm2_gamma_cb, rmsnorm2_gamma_tensor_device
                )
                # Update the tile descriptor in the format descriptor to match rmsnorm2 tile shape
                rmsnorm2_gamma_cb_descriptor.format_descriptors[0].tile = rmsnorm2_tile_descriptor
                rmsnorm2_gamma_cb_descriptor.format_descriptors[0].page_size = rmsnorm2_page_size

                # CBs overlapped with sdpa_kv_cache L1 buffer (consumed before SDPA runs)
                sdpa_kv_cache_running_offset = 0

                # CB: CCL broadcast packet buffer
                bcast_pkt_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(bcast_pkt_cb, input_tensor_device)
                sdpa_kv_cache_running_offset += bcast_pkt_cb_descriptor.total_size

                # CB: RMSNorm output buffer
                rmsnorm_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    rmsnorm_output_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=sdpa_kv_cache_running_offset,
                    total_size=num_tiles * cb_page_size,
                )
                rmsnorm_output_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=rmsnorm_output_cb,
                        data_format=data_format,
                        page_size=cb_page_size,
                        tile=tile_descriptor,
                    )
                ]

                # CB: Matmul weights (created from sharded tensor)
                matmul_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul_weights_cb, matmul_weights_tensor_device
                )

                # CB: Matmul input buffer (1x32 tiles, receives mcast data)
                # Senders will query the write pointer of this CB to get the receiver address.
                # Tensor-backed on full device grid (superset of sender/receiver grids) so senders
                # can use get_write_ptr to get receiver address. This CB is consumed before SDPA runs.
                # CB: Matmul input — overlap with kv_cache L1 buffer at offset 14336 B.
                matmul_input_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                matmul_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul_input_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 0 B
                    total_size=matmul_input_total_size,
                )
                matmul_input_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=matmul_input_cb,
                        data_format=data_format,
                        page_size=matmul_input_page_size,
                        tile=matmul_input_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += matmul_input_cb_descriptor.total_size  # +14336 B

                # CB: Matmul output buffer (single tile) — overlap with sdpa_out_interm L1 buffer
                # at offset 0 B. This CB is consumed before SDPA runs.
                matmul_output_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                matmul_output_page_size = TILE_1x32.get_tile_size(data_format)
                matmul_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul_output_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 14336 B
                    total_size=matmul_output_page_size,
                )
                matmul_output_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=matmul_output_cb,
                        data_format=data_format,
                        page_size=matmul_output_page_size,
                        tile=matmul_output_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += matmul_output_cb_descriptor.total_size  # +64 B

                # CB 7: RMSNorm2 input buffer (3 tiles) — overlap with sdpa_out_interm L1 buffer
                # at offset 64 B. This CB is consumed before SDPA runs.
                rmsnorm2_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    rmsnorm2_input_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 14400 B
                    total_size=rmsnorm2_num_tiles * rmsnorm2_page_size,
                )
                rmsnorm2_input_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=rmsnorm2_input_cb,
                        data_format=data_format,
                        page_size=rmsnorm2_page_size,
                        tile=rmsnorm2_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += rmsnorm2_input_cb_descriptor.total_size  # +3072 B

                # CB9 lifecycle:
                # 1) RMSNorm2 writes normalized output here
                # 2) Mcast2 reads from CB9 and writes to matmul2 input CB

                # CB 8: gather_reduce half1 scratch buffer (3 tiles) — overlap with sdpa_out_interm L1 buffer
                # at offset 3136 B. This CB is consumed before SDPA runs.
                gather_reduce_half1_scratch_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    gather_reduce_half1_scratch_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 17472 B
                    total_size=rmsnorm2_num_tiles * rmsnorm2_page_size,
                )
                gather_reduce_half1_scratch_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=gather_reduce_half1_scratch_cb,
                        data_format=data_format,
                        page_size=rmsnorm2_page_size,
                        tile=rmsnorm2_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += gather_reduce_half1_scratch_cb_descriptor.total_size  # +3072 B

                # CB: RMSNorm2 output buffer (3 tiles)
                rmsnorm2_output_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=rmsnorm2_output_cb,
                    data_format=data_format,
                    page_size=rmsnorm2_page_size,
                    tile=rmsnorm2_tile_descriptor,
                )
                rmsnorm2_output_cb_core_ranges = rmsnorm_core_grid

                rmsnorm2_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    rmsnorm2_output_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 20544 B
                    total_size=rmsnorm2_num_tiles * rmsnorm2_page_size,
                )
                rmsnorm2_output_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=rmsnorm2_output_cb,
                        data_format=data_format,
                        page_size=rmsnorm2_page_size,
                        tile=rmsnorm2_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += rmsnorm2_output_cb_descriptor.total_size  # +3072 B

                # CB: Matmul2 input buffer (1x1536 with 1x32 tiles = 48 tiles) — overlap with
                # sdpa_out_interm L1 buffer at offset 9280 B. This CB is consumed before SDPA runs.
                matmul2_input_total_size = matmul2_num_tiles_k * matmul_input_page_size  # 48 * 64 bytes
                matmul2_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul2_input_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 23616 B
                    total_size=matmul2_input_total_size,
                )
                matmul2_input_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=matmul2_input_cb,
                        data_format=data_format,
                        page_size=matmul_input_page_size,
                        tile=matmul_input_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += matmul2_input_cb_descriptor.total_size  # +3072 B

                # CB: Matmul2 weights (created from sharded tensor, 4 tiles per core)
                matmul2_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul2_weights_cb, matmul2_weights_tensor_device
                )

                # CB 12: Matmul2 output buffer — overlap with sdpa_out_interm L1 buffer
                # at offset 12352 B. This CB is consumed before SDPA runs.
                matmul2_output_total_size = matmul2_out_w * matmul_output_page_size  # 4 * 64 = 256 bytes per core
                matmul2_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul2_output_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 26688 B
                    total_size=matmul2_output_total_size,
                )
                matmul2_output_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=matmul2_output_cb,
                        data_format=data_format,
                        page_size=matmul_output_page_size,
                        tile=matmul_output_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += matmul2_output_cb_descriptor.total_size  # +256 B

                # CB 13: Matmul3 weights (created from sharded tensor on Qnope grid)
                matmul3_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul3_weights_cb, matmul3_weights_tensor_device
                )

                # CB 14: Matmul3 output buffer — overlap with sdpa_out_interm L1 buffer
                # at offset 12608 B. This CB is consumed before SDPA runs.
                matmul3_output_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                matmul3_output_page_size = TILE_1x32.get_tile_size(data_format)
                matmul3_output_total_size = matmul3_out_w * matmul3_output_page_size  # 16 tiles
                matmul3_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul3_output_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 26944 B
                    total_size=matmul3_output_total_size,
                )
                matmul3_output_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=matmul3_output_cb,
                        data_format=data_format,
                        page_size=matmul3_output_page_size,
                        tile=matmul3_output_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += matmul3_output_cb_descriptor.total_size  # +1024 B

                # CB 15: Qrope output buffer — overlap with sdpa_out_interm L1 buffer
                # at offset 13632 B. This CB is consumed before SDPA runs.
                qrope_output_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                qrope_output_page_size = TILE_1x32.get_tile_size(data_format)
                qrope_output_total_size = matmul2_out_w * qrope_output_page_size  # 4 tiles
                qrope_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    qrope_output_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 27968 B
                    total_size=qrope_output_total_size,
                )
                qrope_output_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=qrope_output_cb,
                        data_format=data_format,
                        page_size=qrope_output_page_size,
                        tile=qrope_output_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += qrope_output_cb_descriptor.total_size  # +256 B

                # CB 17: Cos (DRAM, read by NCRISC)
                qrope_rope_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                qrope_cos_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=qrope_cos_cb,
                    data_format=data_format,
                    page_size=qrope_rope_tile_size,
                    tile=qrope_rope_tile_descriptor,
                )
                qrope_cos_cb_descriptor = ttnn.CBDescriptor(
                    total_size=qrope_head_dim_per_core_t * qrope_rope_tile_size,
                    core_ranges=qrope_grid,
                    format_descriptors=[qrope_cos_cb_format],
                )

                # CB 18: Sin (DRAM, read by NCRISC)
                qrope_sin_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=qrope_sin_cb,
                    data_format=data_format,
                    page_size=qrope_rope_tile_size,
                    tile=qrope_rope_tile_descriptor,
                )
                qrope_sin_cb_descriptor = ttnn.CBDescriptor(
                    total_size=qrope_head_dim_per_core_t * qrope_rope_tile_size,
                    core_ranges=qrope_grid,
                    format_descriptors=[qrope_sin_cb_format],
                )

                # CB 19: Trans_mat (sharded tensor)
                qrope_trans_mat_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    qrope_trans_mat_cb, trans_mat_tensor_device
                )

                # CB 20: Rotated input intermediate CB — overlap with sdpa_out_interm L1 buffer
                # at offset 13888 B. This CB is consumed before SDPA runs.
                qrope_interm_tile_size = qrope_head_dim_per_core_t * TILE_1x32.get_tile_size(data_format)
                qrope_rotated_input_interm_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    qrope_rotated_input_interm_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 28224 B
                    total_size=qrope_interm_tile_size,
                )
                qrope_rotated_input_interm_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=qrope_rotated_input_interm_cb,
                        data_format=data_format,
                        page_size=TILE_1x32.get_tile_size(data_format),
                        tile=ttnn.TileDescriptor(TILE_1x32),
                    )
                ]
                sdpa_out_interm_running_offset += qrope_rotated_input_interm_cb_descriptor.total_size  # +128 B

                # CB 21: Cos intermediate CB — overlap with sdpa_out_interm L1 buffer
                # at offset 14016 B. This CB is consumed before SDPA runs.
                qrope_cos_interm_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    qrope_cos_interm_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 28352 B
                    total_size=qrope_interm_tile_size,
                )
                qrope_cos_interm_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=qrope_cos_interm_cb,
                        data_format=data_format,
                        page_size=TILE_1x32.get_tile_size(data_format),
                        tile=ttnn.TileDescriptor(TILE_1x32),
                    )
                ]
                sdpa_out_interm_running_offset += qrope_cos_interm_cb_descriptor.total_size  # +128 B

                # CB 22: Sin intermediate CB — overlap with sdpa_out_interm L1 buffer
                # at offset 14144 B. This CB is consumed before SDPA runs.
                qrope_sin_interm_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    qrope_sin_interm_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 28480 B
                    total_size=qrope_interm_tile_size,
                )
                qrope_sin_interm_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=qrope_sin_interm_cb,
                        data_format=data_format,
                        page_size=TILE_1x32.get_tile_size(data_format),
                        tile=ttnn.TileDescriptor(TILE_1x32),
                    )
                ]
                sdpa_out_interm_running_offset += qrope_sin_interm_cb_descriptor.total_size  # +128 B

                # CB 31: CreateQHeads intermediate buffer (row-major data before tilization)
                # Senders write row-major data here via NOC, receiver marks pages, TRISC tilizes to output
                # Allocated on union of sender (QNOPE/QROPE) and receiver (SDPA Input) grids
                # so senders can use get_write_ptr to determine the L1 destination address
                TILE_8x32 = ttnn.Tile((8, 32))
                create_q_heads_interm_tile_descriptor = ttnn.TileDescriptor(TILE_8x32)
                create_q_heads_interm_page_size = TILE_8x32.get_tile_size(data_format)  # 8*32*2 = 512 bytes
                create_q_heads_interm_total_size = (
                    2 * nope_tiles + rope_tiles
                ) * create_q_heads_interm_page_size  # 18 pages (all phases: 8+8+2)
                create_q_heads_interm_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    create_q_heads_receiver_in_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 28608 B
                    total_size=create_q_heads_interm_total_size,
                )
                create_q_heads_interm_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=create_q_heads_receiver_in_cb,
                        data_format=data_format,
                        page_size=create_q_heads_interm_page_size,
                        tile=create_q_heads_interm_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += create_q_heads_interm_cb_descriptor.total_size  # +9216 B

                # CB 16: CreateQHeads output buffer (tilized data, backed by output tensor)
                # Only allocated on receiver cores (SDPA Input grid) - senders no longer write here
                create_q_heads_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    create_q_heads_out_cb, output_tensor_device
                )

                # CB: DKV Matmul weights buffer
                dkv_matmul_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    dkv_matmul_weights_cb, dkv_matmul_weights_tensor_device
                )

                # CB 24: DKV Matmul output — overlap with sdpa_out_interm L1 buffer
                # at offset 14272 B. This CB is consumed before SDPA runs.
                dkv_matmul_output_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                dkv_matmul_output_page_size = TILE_1x32.get_tile_size(data_format)
                dkv_matmul_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    dkv_matmul_output_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 37824 B
                    total_size=dkv_matmul_output_page_size,
                )
                dkv_matmul_output_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=dkv_matmul_output_cb,
                        data_format=data_format,
                        page_size=dkv_matmul_output_page_size,
                        tile=dkv_matmul_output_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += dkv_matmul_output_cb_descriptor.total_size  # +64 B

                # CB 25: KV RMSNorm input buffer — overlap with sdpa_out_interm L1 buffer
                # at offset 14336 B. This CB is consumed before SDPA runs.
                kv_rmsnorm_tile_descriptor = ttnn.TileDescriptor(TILE_16x32)
                kv_rmsnorm_page_size = TILE_16x32.get_tile_size(input_tensor_sample.dtype)
                kv_rmsnorm_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    kv_rmsnorm_input_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 37888 B
                    total_size=1 * kv_rmsnorm_page_size,
                )
                kv_rmsnorm_input_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=kv_rmsnorm_input_cb,
                        data_format=data_format,
                        page_size=kv_rmsnorm_page_size,
                        tile=kv_rmsnorm_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += kv_rmsnorm_input_cb_descriptor.total_size  # +1024 B

                # CB: KV RMSNorm gamma buffer
                kv_rmsnorm_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    kv_rmsnorm_gamma_cb, dkv_rmsnorm_gamma_tensor_device
                )
                kv_rmsnorm_gamma_cb_descriptor.format_descriptors[0].tile = kv_rmsnorm_tile_descriptor
                kv_rmsnorm_gamma_cb_descriptor.format_descriptors[0].page_size = kv_rmsnorm_page_size

                # CB 27: KV RMSNorm output buffer — overlap with sdpa_out_interm L1 buffer
                # at offset 15360 B. This CB is consumed before SDPA runs.
                kv_rmsnorm_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    kv_rmsnorm_output_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 38912 B
                    total_size=kv_rmsnorm_num_tiles * kv_rmsnorm_page_size,
                )
                kv_rmsnorm_output_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=kv_rmsnorm_output_cb,
                        data_format=data_format,
                        page_size=kv_rmsnorm_page_size,
                        tile=kv_rmsnorm_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += kv_rmsnorm_output_cb_descriptor.total_size  # +1024 B

                # CB 29: Cos (DRAM, read by NCRISC)
                krope_rope_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                krope_cos_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=krope_cos_cb,
                    data_format=data_format,
                    page_size=krope_rope_tile_size,
                    tile=krope_rope_tile_descriptor,
                )
                krope_cos_cb_descriptor = ttnn.CBDescriptor(
                    total_size=krope_Wt * krope_rope_tile_size,
                    core_ranges=krope_grid,
                    format_descriptors=[krope_cos_cb_format],
                )
                # CB 30: Sin (DRAM, read by NCRISC)
                krope_sin_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=krope_sin_cb,
                    data_format=data_format,
                    page_size=krope_rope_tile_size,
                    tile=krope_rope_tile_descriptor,
                )
                krope_sin_cb_descriptor = ttnn.CBDescriptor(
                    total_size=krope_Wt * krope_rope_tile_size,
                    core_ranges=krope_grid,
                    format_descriptors=[krope_sin_cb_format],
                )

                # CB 28: KRoPE output — overlap with sdpa_out_interm L1 buffer
                # at offset 16384 B. This CB is consumed before SDPA runs.
                krope_tile_size = TILE_1x32.get_tile_size(data_format)
                krope_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    krope_output_cb,
                    sdpa_out_interm_buffer_device,
                    address_offset=sdpa_out_interm_running_offset,  # 39936 B
                    total_size=1 * krope_tile_size,
                )
                krope_output_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=krope_output_cb,
                        data_format=data_format,
                        page_size=krope_tile_size,
                        tile=ttnn.TileDescriptor(TILE_1x32),
                    )
                ]
                sdpa_out_interm_running_offset += krope_output_cb_descriptor.total_size  # +64 B

                dkv_rmsnorm_grid = dkv_rmsnorm_gamma_tensor_device.memory_config().shard_spec.grid
                TILE_32x32 = ttnn.Tile((32, 32))
                kv_cache_page_size = TILE_32x32.get_tile_size(ttnn.bfloat8_b)
                kv_cache_update_grid = dkv_rmsnorm_grid.merge(krope_grid)
                kv_cache_num_tiles = 16
                kv_cache_input_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=kv_cache_input_cb,
                    data_format=ttnn.bfloat8_b,
                    page_size=kv_cache_page_size,
                    tile=ttnn.TileDescriptor(TILE_32x32),
                )
                kv_cache_input_cb_descriptor = ttnn.CBDescriptor(
                    total_size=kv_cache_num_tiles * kv_cache_page_size,
                    core_ranges=kv_cache_update_grid,
                    format_descriptors=[kv_cache_input_cb_format],
                )
                kv_cache_output_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=kv_cache_output_cb,
                    data_format=ttnn.bfloat8_b,
                    page_size=kv_cache_page_size,
                    tile=ttnn.TileDescriptor(TILE_32x32),
                )
                kv_cache_output_cb_descriptor = ttnn.CBDescriptor(
                    total_size=kv_cache_num_tiles * kv_cache_page_size,
                    core_ranges=kv_cache_update_grid,
                    format_descriptors=[kv_cache_output_cb_format],
                )
                kv_cache_intermed_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=kv_cache_intermed_cb,
                    data_format=ttnn.bfloat16,
                    page_size=TILE_32x32.get_tile_size(ttnn.bfloat16),
                    tile=ttnn.TileDescriptor(TILE_32x32),
                )
                # One extra tile for syncing, can optimize to remove
                kv_cache_intermed_cb_descriptor = ttnn.CBDescriptor(
                    total_size=(kv_cache_num_tiles + 1) * TILE_32x32.get_tile_size(ttnn.bfloat16),
                    core_ranges=kv_cache_update_grid,
                    format_descriptors=[kv_cache_intermed_cb_format],
                )
                kv_cache_tensor_accessor_args = ttnn.TensorAccessorArgs(kv_cache_tensor_device)
                brisc_compile_time_args = kv_cache_tensor_accessor_args.get_compile_time_args()
                ncrisc_compile_time_args = kv_cache_tensor_accessor_args.get_compile_time_args()
                # ========================================================================
                # Mcast2 compile-time args (uses same grid and semaphores as first mcast)
                # ========================================================================
                # BRISC sender: data_size_bytes, src_num_pages, rmsnorm2_output_cb (grid/semaphores reused from mcast)
                mcast2_brisc_named_compile_time_args = [
                    ("mcast2_data_size_bytes", mcast2_data_size_bytes),
                    ("mcast2_src_num_pages", mcast2_src_num_pages),
                    ("rmsnorm2_output_cb", rmsnorm2_output_cb),  # Source CB for mcast2 sender
                ]
                # NCRISC receiver: dst_num_pages (semaphore reused from mcast)
                mcast2_ncrisc_named_compile_time_args = [
                    ("mcast2_dst_num_pages", mcast2_dst_num_pages),
                ]

                # ========================================================================
                # Semaphore descriptors
                # ========================================================================

                # Mcast semaphores (ID 0 and 1)
                mcast_sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=mcast_data_sender_semaphore_id,
                    core_ranges=full_device_grid,
                    initial_value=0,
                )

                mcast_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=mcast_data_receiver_semaphore_id,
                    core_ranges=full_device_grid,
                    initial_value=0,
                )

                # Gather semaphores (ID 2 and 3 - two semaphores for NOC0 and NOC1, but only NOC0 is used)
                gather_noc0_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather_noc0_receiver_semaphore_id,
                    core_ranges=full_device_grid,
                    initial_value=0,
                )

                gather_noc1_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather_noc1_receiver_semaphore_id,
                    core_ranges=full_device_grid,
                    initial_value=0,
                )

                # ================================================================
                # CCL Broadcast common runtime args (computed before UnifiedKernelDescriptor)
                # These are common to all cores since only one core participates in CCL
                # ================================================================
                if skip_ccl:
                    # Single-device mode: empty broadcast args
                    ncrisc_bcast_common_args = [0] * 13
                    dst_nodes = []
                    fabric_node_id = None
                else:
                    # Multi-device mode: CCL broadcast runtime args
                    wait_output_semaphore = is_secondary_sender or is_receiver
                    reset_global_semaphore = is_secondary_sender or is_receiver
                    out_ready_sem_wait_value = 1 * num_links

                    # Build dst_nodes first to compute num_connections = len(dst_nodes)
                    fabric_node_id = mesh_device.get_fabric_node_id(coord)
                    dst_nodes = []

                    # Primary axis connections (forward and backward in column)
                    if num_targets_forward > 0:
                        forward_coord = ttnn.MeshCoordinate(row + 1, col)
                        dst_nodes.append(mesh_device.get_fabric_node_id(forward_coord))

                    if num_targets_backward > 0:
                        backward_coord = ttnn.MeshCoordinate(row - 1, col)
                        dst_nodes.append(mesh_device.get_fabric_node_id(backward_coord))

                    # Secondary axis connection (for sender to secondary sender)
                    if has_secondary_target:
                        secondary_coord = ttnn.MeshCoordinate(row, 1)
                        dst_nodes.append(mesh_device.get_fabric_node_id(secondary_coord))

                    num_connections = len(dst_nodes)

                    ncrisc_bcast_common_args = [
                        int(intermediate_tensor_device.buffer_address()),  # tensor_address0
                        int(out_ready_sem_addr),  # out_ready_sem_bank_addr
                        int(wait_output_semaphore),
                        int(reset_global_semaphore),
                        core_noc_x,  # out_ready_sem_noc0_x
                        core_noc_y,  # out_ready_sem_noc0_y
                        out_ready_sem_wait_value,
                        int(barrier_sem_addr),
                        core_noc_x,  # barrier_sem_noc0_x
                        core_noc_y,  # barrier_sem_noc0_y
                        ring_index,
                        int(secondary_sync_sem_addr),
                        num_connections,
                    ]

                # TRISC common runtime args (shared scalar values)
                trisc_common_runtime_args = [
                    epsilon_packed,  # idx 0
                    scalar_packed,  # idx 1
                    scalar2_packed,  # idx 2
                    kv_scalar_packed,  # idx 3
                    kv_cache_input_cb,
                    kv_cache_output_cb,
                    kv_cache_intermed_cb,
                ]

                # RoPE DRAM address args (per-device)
                qrope_cos_tensor_address = qrope_cos_tensor_device.buffer_address()
                qrope_sin_tensor_address = qrope_sin_tensor_device.buffer_address()
                krope_cos_tensor_address = krope_cos_tensor_device.buffer_address()
                krope_sin_tensor_address = krope_sin_tensor_device.buffer_address()
                position_ids_tensor_addr = position_ids_tensor_device.buffer_address()

                qrope_ncrisc_addr_args = [
                    ("qrope_cos_tensor_address", qrope_cos_tensor_address),
                    ("qrope_sin_tensor_address", qrope_sin_tensor_address),
                    ("qrope_position_ids_tensor_address", position_ids_tensor_addr),
                ]
                krope_ncrisc_addr_args = [
                    ("krope_cos_tensor_address", krope_cos_tensor_address),
                    ("krope_sin_tensor_address", krope_sin_tensor_address),
                    ("krope_position_ids_tensor_address", position_ids_tensor_addr),
                ]

                # Per-core start_tile_offset for QRoPE (all cores read full head_dim, offset=0)
                qrope_cores = ttnn.corerange_to_cores(qrope_grid)
                qrope_start_tile_offset_core_values = [(core, 0) for core in qrope_cores]

                # Per-core start_tile_offset for KRoPE (2 cores, each reads its width slice)
                krope_cores = ttnn.corerange_to_cores(krope_grid)
                krope_start_tile_offset_core_values = [(core, idx * krope_Wt) for idx, core in enumerate(krope_cores)]

                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/kernels/pre_sdpa_kernel.cpp",
                    core_ranges=full_device_grid,
                    ncrisc_compile_time_args=ncrisc_compile_time_args,
                    brisc_compile_time_args=brisc_compile_time_args,
                    ncrisc_named_compile_time_args=bcast_ncrisc_named_compile_time_args
                    + rmsnorm_reader_named_compile_time_args
                    + mcast_receiver_named_compile_time_args
                    + matmul_ncrisc_named_compile_time_args
                    + gather_reduce_sender_named_compile_time_args
                    + rmsnorm2_ncrisc_named_compile_time_args
                    + matmul2_ncrisc_named_compile_time_args
                    + mcast2_ncrisc_named_compile_time_args
                    + matmul3_ncrisc_named_compile_time_args
                    + qrope_ncrisc_named_compile_time_args
                    + qrope_ncrisc_addr_args
                    + create_q_heads_ncrisc_named_compile_time_args
                    + dkv_matmul_ncrisc_named_compile_time_args
                    + kv_rmsnorm_ncrisc_named_compile_time_args
                    + dkv_gather_sender_named_compile_time_args
                    + krope_ncrisc_named_compile_time_args
                    + krope_ncrisc_addr_args,
                    # NCRISC common runtime args:
                    ncrisc_common_runtime_args=ncrisc_bcast_common_args,
                    # BRISC named compile-time args: bcast + rmsnorm reader (for gamma setup) + mcast sender + matmul + gather_reduce receiver + matmul2 + mcast2 + matmul3 + qrope + create_q_heads + dkv_matmul + dkv_gather_receiver + kv_rmsnorm
                    brisc_named_compile_time_args=bcast_brisc_named_compile_time_args
                    + mcast_sender_named_compile_time_args
                    + matmul_brisc_named_compile_time_args
                    + gather_reduce_receiver_named_compile_time_args
                    + matmul2_brisc_named_compile_time_args
                    + mcast2_brisc_named_compile_time_args
                    + matmul3_brisc_named_compile_time_args
                    + qrope_brisc_named_compile_time_args
                    + create_q_heads_brisc_named_compile_time_args
                    + dkv_gather_receiver_named_compile_time_args
                    + kv_rmsnorm_brisc_named_compile_time_args
                    + kv_cache_brisc_named_compile_time_args,
                    # BRISC common runtime args: bcast args
                    brisc_common_runtime_args=[int(kv_cache_tensor_device.buffer_address()), position_id],
                    # TRISC named compile-time args: rmsnorm compute + matmul + gather-reduce + rmsnorm2 + matmul2 + matmul3 + qrope + create_q_heads + dkv_matmul + kv_rmsnorm + krope
                    trisc_named_compile_time_args=bcast_trisc_named_compile_time_args
                    + rmsnorm_compute_named_compile_time_args
                    + matmul_trisc_named_compile_time_args
                    + gather_reduce_trisc_named_compile_time_args
                    + rmsnorm2_trisc_named_compile_time_args
                    + matmul2_trisc_named_compile_time_args
                    + matmul3_trisc_named_compile_time_args
                    + qrope_trisc_named_compile_time_args
                    + create_q_heads_trisc_named_compile_time_args
                    + dkv_matmul_trisc_named_compile_time_args
                    + kv_rmsnorm_trisc_named_compile_time_args
                    + krope_trisc_named_compile_time_args
                    + kv_cache_trisc_named_compile_time_args,
                    # TRISC common runtime args (shared by all cores)
                    trisc_common_runtime_args=trisc_common_runtime_args,
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=fp32_dest_acc_en,
                        dst_full_sync_en=fp32_dest_acc_en,
                    ),
                    # Per-core compile-time role differentiation
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_input_core",
                            core_range=rmsnorm_core,  # First core is the input core
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_matmul_core",
                            core_range=matmul_weights_core_grid,  # 96 matmul cores (12x8)
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_matmul2_core",
                            core_range=matmul2_weights_core_grid,  # matmul2 cores
                            value=1,
                            other_value=0,
                        ),
                        # Qnope/Qrope core differentiation for interleaved Q head layout
                        # Qnope cores: 64 cores (8x8 grid), each handles 1 head of 128 elements
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_qnope_core",
                            core_range=qnope_grid,
                            value=1,
                            other_value=0,
                        ),
                        # Qrope cores: 32 cores (4x8 grid), each handles 2 heads of 64 elements
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_qrope_core",
                            core_range=qrope_grid,
                            value=1,
                            other_value=0,
                        ),
                        # SDPA Input cores: 8 cores (4×2 grid), receive interleaved QNOPE/QROPE data
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_sdpa_input_core",
                            core_range=sdpa_input_grid,
                            value=1,
                            other_value=0,
                        ),
                        # DKV Matmul core: 9x2 grid, each core handles 1 head of 32 dim
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_dkv_matmul_core",
                            core_range=dkv_matmul_weights_core_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_kv_rmsnorm_core",
                            core_range=dkv_rmsnorm_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_knope_core",
                            core_range=dkv_gather_sender_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_krope_core",
                            core_range=krope_grid,
                            value=1,
                            other_value=0,
                        ),
                    ],
                    per_core_compile_time_descriptors=[
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="qrope_start_tile_offset",
                            core_values=qrope_start_tile_offset_core_values,
                            other_value=0,
                        ),
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="krope_start_tile_offset",
                            core_values=krope_start_tile_offset_core_values,
                            other_value=0,
                        ),
                    ],
                    # Per-core runtime args for fabric (BRISC only, on worker_core)
                    # Initialize empty args that will be populated by setup_routing_plane_connection
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(worker_core, [])],  # Fabric args appended after program creation
                    ),
                    noc_mode=noc_mode,
                )

                # ================================================================
                # Create program descriptor
                # ================================================================
                cbs_list = [
                    in_cb_descriptor,
                    gamma_cb_descriptor,
                    rmsnorm_output_cb_descriptor,
                    matmul_weights_cb_descriptor,
                    matmul_output_cb_descriptor,
                    matmul_input_cb_descriptor,
                    rmsnorm2_gamma_cb_descriptor,  # CB 6: RMSNorm2 gamma
                    rmsnorm2_input_cb_descriptor,  # CB 7: RMSNorm2 input
                    gather_reduce_half1_scratch_cb_descriptor,  # CB 8: gather_reduce half1 scratch
                    rmsnorm2_output_cb_descriptor,  # CB 9: RMSNorm2 output
                    matmul2_input_cb_descriptor,  # CB 10: Matmul2 input
                    matmul2_weights_cb_descriptor,  # CB 11: Matmul2 weights
                    matmul2_output_cb_descriptor,  # CB 12: Matmul2 output (intermediate)
                    matmul3_weights_cb_descriptor,  # CB 13: Matmul3 weights
                    matmul3_output_cb_descriptor,  # CB 14: Matmul3 output (Qnope final)
                    qrope_output_cb_descriptor,  # CB 15: Qrope output (RoPE output)
                    create_q_heads_out_cb_descriptor,  # CB 16: CreateQHeads output (tilized, linked to tensor)
                    qrope_cos_cb_descriptor,  # CB 17: Cos (DRAM, read by NCRISC)
                    qrope_sin_cb_descriptor,  # CB 18: Sin (DRAM, read by NCRISC)
                    qrope_trans_mat_cb_descriptor,  # CB 19: Trans_mat (sharded tensor)
                    qrope_rotated_input_interm_cb_descriptor,  # CB 20: Rotated input intermediate
                    qrope_cos_interm_cb_descriptor,  # CB 21: Cos intermediate
                    qrope_sin_interm_cb_descriptor,  # CB 22: Sin intermediate
                    dkv_matmul_weights_cb_descriptor,  # CB 23: DKV Matmul weights
                    dkv_matmul_output_cb_descriptor,  # CB 24: DKV Matmul output
                    kv_rmsnorm_input_cb_descriptor,  # CB 25: KV RMSNorm input
                    kv_rmsnorm_gamma_cb_descriptor,  # CB 26: KV RMSNorm gamma
                    kv_rmsnorm_output_cb_descriptor,  # CB 27: KV RMSNorm output
                    krope_output_cb_descriptor,  # CB 28: KV Cache Branch RoPE output
                    krope_cos_cb_descriptor,  # CB 29: Cos (DRAM, read by NCRISC)
                    krope_sin_cb_descriptor,  # CB 30: Sin (DRAM, read by NCRISC)
                    create_q_heads_interm_cb_descriptor,  # CB 31: CreateQHeads intermediate (row-major)
                    kv_cache_output_cb_descriptor,  # CB 32: KV Cache output
                    kv_cache_intermed_cb_descriptor,  # CB 33: KV Cache intermed
                    kv_cache_input_cb_descriptor,  # CB 34: KV Cache input
                ]
                if not skip_ccl:
                    cbs_list.append(bcast_pkt_cb_descriptor)

                program = ttnn.ProgramDescriptor(
                    kernels=unified_kernel.get_kernel_descriptors().kernels,
                    cbs=cbs_list,
                    semaphores=[
                        mcast_sender_semaphore_descriptor,  # ID 0
                        mcast_receiver_semaphore_descriptor,  # ID 1
                        gather_noc0_receiver_semaphore_descriptor,  # ID 2 (reused by create_q_heads)
                        gather_noc1_receiver_semaphore_descriptor,  # ID 3
                    ],
                )

                # Append fabric connection args to BRISC kernel if needed (CCL mode only)
                # Runtime args are already initialized by UnifiedKernelDescriptor via per_core_runtime_args_descriptors
                if not skip_ccl and num_connections > 0:
                    # Find the BRISC (writer) kernel whose core_ranges includes worker_core
                    for idx, kernel in enumerate(program.kernels):
                        if kernel.core_ranges.contains(worker_core) and (
                            isinstance(kernel.config, ttnn.ReaderConfigDescriptor)
                            or (
                                isinstance(kernel.config, ttnn.DataMovementConfigDescriptor)
                                and kernel.config.processor == ttnn.DataMovementProcessor.RISCV_1
                            )
                        ):
                            writer_rt_args_ref = kernel.runtime_args[worker_core.x][worker_core.y]
                            fabric_args = ttnn.setup_routing_plane_connection(
                                fabric_node_id, dst_nodes, [0], program, idx, worker_core
                            )
                            writer_rt_args_ref.extend(fabric_args)
                            break

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        result = ttnn.generic_op(
            [
                input_tensor_mesh,
                intermediate_tensor_mesh,
                gamma_tensor,
                matmul_weights_tensor,
                rmsnorm2_gamma_tensor,
                matmul2_weights_tensor,
                matmul3_weights_tensor,
                trans_mat_tensor,
                qrope_cos_tensor,
                qrope_sin_tensor,
                krope_cos_tensor,
                krope_sin_tensor,
                position_ids_tensor,
                dkv_matmul_weights_tensor,
                dkv_rmsnorm_gamma_tensor,
                kv_cache_tensor,
                sdpa_kv_cache_buffer,
                sdpa_out_interm_buffer,
                output_tensor,
            ],
            mesh_program_descriptor,
        )

        return result
