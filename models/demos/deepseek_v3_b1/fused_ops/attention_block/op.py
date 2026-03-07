# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import math

import torch

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import (
    KVB12_PROJ_SingleDeviceOverlapSpec,
    O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec,
)
from models.demos.deepseek_v3_b1.circular_buffer_utils import (
    CircularBufferIdManager,
    cb_descriptor_from_overlapped_tensor,
    cb_descriptor_from_overlapped_tensors,
)
from models.demos.deepseek_v3_b1.fused_ops.post_sdpa.op import _extend_runtime_args, _get_element_size_bytes, _round_up
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import (
    FlashMLADecode,
    get_max_page_size_and_num_pages,
    get_noc_max_page_size,
)
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_uint32


def extend_fabric_args(existing_rt_args, fabric_args):
    """Use this to append fabric args that pass in arg_idx to the open_connection api"""
    existing_rt_args.extend([len(fabric_args), *fabric_args])


class AttentionBlock:
    """
    Attention block fused operation implementation using ttnn.generic_op.
    This block includes:
    - RMSNorm
    - Matmul
    - RoPE
    - KV cache update
    - SDPA
    - Concat heads
    - WO matmul
    - All reduce + residual add
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
        kv_cache_tensor,
        scale,
        epsilon=1e-6,
        num_qnope_heads=64,
        num_qrope_heads=64,
        qnope_head_dim=128,
        qrope_head_dim=64,
        heads_per_row=8,
        nope_dim=512,
        rope_dim=64,
        post_sdpa_weights1=None,
        post_sdpa_weights2=None,
        num_tp=1,
    ):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: [1, K]
            gamma_tensor: RMSNorm gamma [1, K]
            matmul_weights_tensor: q_a_proj weights [K, N]
            rmsnorm2_gamma_tensor: q_norm gamma [1, N]
            matmul2_weights_tensor: q_b_proj weights [N, num_qnope_heads*qnope_head_dim + num_qrope_heads*qrope_head_dim]
            matmul3_weights_tensor: kv_b1_proj weights [num_qnope_heads, qnope_head_dim, nope_dim]
            sin_tensor: RoPE sin table [max_seq_len, qrope_head_dim]
            cos_tensor: RoPE cos table [max_seq_len, qrope_head_dim]
            position_ids: global decode position [batch]
            dkv_matmul_weights_tensor: kv_a_proj weights [K, nope_dim + rope_dim]
            dkv_rmsnorm_gamma_tensor: kv_norm gamma [1, nope_dim]
            kv_cache_tensor: KV cache [1, 1, seq_len, nope_dim + rope_dim]
            scale: attention scale factor
            epsilon: RMSNorm epsilon (default 1e-6)
            num_qnope_heads: number of Q NoPE heads (default 64)
            num_qrope_heads: number of Q RoPE heads (default 64)
            qnope_head_dim: Q NoPE head dim (default 128)
            qrope_head_dim: Q/K RoPE head dim (default 64)
            heads_per_row: heads per SDPA core row (default 8)
            nope_dim: KV NoPE dim (default 512)
            rope_dim: KV RoPE dim (default 64)
            post_sdpa_weights1: kv_b2_proj weights [nope_dim, intermediate] per TP device (optional)
            post_sdpa_weights2: o_proj weights [intermediate, output_size] per TP device (optional)
            num_tp: number of TP devices (default 1)

        Returns:
            Tuple of (full_q, new_kv, output):
            - full_q: [1, 1, num_qnope_heads, nope_dim + rope_dim] combined Q heads
            - new_kv: [1, 1, 1, nope_dim + rope_dim] new KV entry written at position_ids[0]
            - output: [1, output_size] post-SDPA output with residual if post_sdpa weights provided,
                      otherwise [num_qnope_heads, nope_dim] FlashMLA attention output
        """
        from models.demos.deepseek_v3_b1.micro_ops.rope.op import RopeSingleCore

        def rmsnorm(x, gamma):
            variance = x.pow(2).mean(-1, keepdim=True)
            normalized = x * torch.rsqrt(variance + epsilon)
            return normalized * gamma

        position_id = position_ids[0]
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

        # Combine QNOPE and QROPE outputs
        combined_head_dim = nope_dim + rope_dim  # 512 + 64 = 576

        full_q = torch.concat([qnope_output, qrope_output], dim=-1).reshape(1, 1, num_qnope_heads, combined_head_dim)

        # KV Cache Branch
        dkv = input_layernorm @ dkv_matmul_weights_tensor
        kv, k_rope = torch.split(dkv, [nope_dim, rope_dim], dim=-1)
        kv = rmsnorm(kv, dkv_rmsnorm_gamma_tensor)
        k_rope = RopeSingleCore.golden(k_rope, cos_tensor, sin_tensor, position_ids).squeeze(0)

        # from 0 to position id, the kv cache is valid
        full_kv = kv_cache_tensor.to(full_q.dtype)
        new_kv = torch.cat([kv, k_rope], dim=-1).reshape(1, 1, 1, combined_head_dim).to(full_q.dtype)
        full_kv[:, :, position_id, :] = new_kv

        sdpa_output = FlashMLADecode.golden(full_q, full_kv, position_ids, nope_dim, scale).squeeze()

        if post_sdpa_weights1 is None or post_sdpa_weights2 is None:
            return full_q, new_kv, sdpa_output

        from models.demos.deepseek_v3_b1.fused_ops.post_sdpa.op import PostSDPA

        heads_per_tp = num_qnope_heads // num_tp
        post_sdpa_result = None
        for tp in range(num_tp):
            tp_sdpa_output = sdpa_output[tp * heads_per_tp : (tp + 1) * heads_per_tp].to(post_sdpa_weights1.dtype)
            tp_result = PostSDPA.golden(tp_sdpa_output, post_sdpa_weights1, post_sdpa_weights2)
            post_sdpa_result = tp_result if post_sdpa_result is None else post_sdpa_result + tp_result

        output = post_sdpa_result + input_tensor
        return full_q, new_kv, output

    @staticmethod
    def get_num_semaphores():
        # 13 form pre sdpa, 4 from post
        return 17

    @staticmethod
    def create_semaphores(mesh_device):
        num_semaphores = AttentionBlock.get_num_semaphores()
        device_grid_size = mesh_device.compute_with_storage_grid_size()
        available_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )
        semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(num_semaphores)]
        return semaphores

    @staticmethod
    def get_program_context(
        input_tensor_mesh,
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
        scale,
        output_tensor,
        sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer,
        sender_coord,
        # Post-SDPA parameters
        post_sdpa_weights1_tensor,
        post_sdpa_weights2_tensor,
        post_sdpa_gather2_output_tensor,
        post_sdpa_gather3_output_tensor,
        post_sdpa_intermediate_tensor,
        sdpa_input_l_mesh,
        sdpa_input_ms_mesh,
        sdpa_output_l_mesh,
        sdpa_intermediate_recv_mesh,
        sdpa_forwarder_scratch_mesh,
        sdpa_per_device_chunk_size,
        attention_block_output_tensor,
        # Shared semaphores, and some default values
        attention_block_semaphores=None,
        bcast_cluster_axis=0,
        bcast_secondary_cluster_axis=1,
        reduce_cluster_axis=1,
        sdpa_cluster_axis=0,
        sdpa_scale_fp32=1.0,
        num_links=1,
        epsilon=1e-6,
        fp32_dest_acc_en=False,
        skip_ccl=False,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
        cb_id_context=None,
    ):
        """
        Execute pre-SDPA fused operation using generic_op.

        Args:
            input_tensor_mesh: Input mesh tensor (must be sharded on single core per device)
            gamma_tensor: OverlappedTensor for attn_norm gamma (shares fused o_proj/gate/gamma buffer)
            matmul_weights_tensor: OverlappedTensor for packed q_a_proj weights (shares fused buffer)
            rmsnorm2_gamma_tensor: OverlappedTensor for q_norm gamma (shares fused o_proj/gate/gamma buffer)
            matmul2_weights_tensor: OverlappedTensor for shuffled q_b_proj weights (shares fused buffer)
            matmul3_weights_tensor: OverlappedTensor for kv_b1_proj weights (shares fused kv_b12 buffer)
            qrope_sin_tensor: Sin tensor (sharded tensor for QRoPE)
            qrope_cos_tensor: Cos tensor (sharded tensor for QRoPE)
            trans_mat_tensor: Trans_mat tensor (sharded tensor for RoPE)
            position_ids_tensor: Position IDs tensor (sharded tensor for RoPE)
            output_tensor: Output tensor for pre-SDPA (sharded on SDPA grid, [8, 576] per core = 8 interleaved heads)
            sender_coord: Tuple (row, col) of sender device in mesh
            semaphores: List of global semaphores [out_ready, barrier, secondary_sync] for CCL
            bcast_cluster_axis: Primary axis for CCL broadcast (0=row, 1=col)
            bcast_secondary_cluster_axis: Secondary axis for CCL broadcast (optional)
            reduce_cluster_axis: Primary axis for CCL reduce (0=row, 1=col)
            num_links: Number of fabric links for CCL
            epsilon: Small value to avoid division by zero
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel
            skip_ccl: If True, skip CCL broadcast (single-device mode)
            noc_mode: NOC mode for the kernel (dedicated or dynamic)

        Returns:
            Tuple of (io_tensors, full_device_grid, per_device_contexts) where:
            - full_device_grid: CoreRangeSet covering the full device compute grid
            - per_device_contexts: List of dicts, one per device, each containing:
                ncrisc_compile_time_args, brisc_compile_time_args,
                brisc_named_compile_time_args, ncrisc_named_compile_time_args,
                trisc_named_compile_time_args, brisc_common_runtime_args,
                ncrisc_common_runtime_args, trisc_common_runtime_args,
                unified_compile_time_core_descriptors, per_core_compile_time_descriptors,
                per_core_ncrisc_args, per_core_brisc_args, per_core_trisc_args,
                cbs_list, worker_core, fabric_node_id, dst_nodes
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
        gamma_fused_tensors_per_device = ttnn.get_device_tensors(gamma_tensor.fused_tensor)
        fused_weights_tensors_per_device = ttnn.get_device_tensors(matmul_weights_tensor.fused_tensor)
        kv_b12_fused_tensors_per_device = ttnn.get_device_tensors(matmul3_weights_tensor.fused_tensor)
        qrope_sin_tensors_per_device = ttnn.get_device_tensors(qrope_sin_tensor)
        qrope_cos_tensors_per_device = ttnn.get_device_tensors(qrope_cos_tensor)
        trans_mat_tensors_per_device = ttnn.get_device_tensors(trans_mat_tensor)
        krope_cos_tensors_per_device = ttnn.get_device_tensors(krope_cos_tensor)
        krope_sin_tensors_per_device = ttnn.get_device_tensors(krope_sin_tensor)
        position_ids_tensors_per_device = ttnn.get_device_tensors(position_ids_tensor)
        kv_cache_tensors_per_device = ttnn.get_device_tensors(kv_cache_tensor)
        sdpa_out_interm_buffers_per_device = ttnn.get_device_tensors(sdpa_out_interm_buffer)
        sdpa_kv_cache_buffers_per_device = ttnn.get_device_tensors(sdpa_kv_cache_buffer)

        # Post-SDPA parameters
        post_sdpa_weights1_fused_tensors_per_device = ttnn.get_device_tensors(post_sdpa_weights1_tensor.fused_tensor)
        post_sdpa_weights2_fused_tensors_per_device = ttnn.get_device_tensors(post_sdpa_weights2_tensor.fused_tensor)
        post_sdpa_gather3_output_tensors_per_device = ttnn.get_device_tensors(post_sdpa_gather3_output_tensor)
        post_sdpa_intermediate_tensors_per_device = ttnn.get_device_tensors(post_sdpa_intermediate_tensor)

        attention_block_output_tensors_per_device = ttnn.get_device_tensors(attention_block_output_tensor)

        assert (
            attention_block_semaphores is not None
            and len(attention_block_semaphores) == AttentionBlock.get_num_semaphores()
        )

        # Semaphore addresses (only needed for CCL mode)
        out_ready_sem_addr = 0
        barrier_sem_addr = 0
        secondary_sync_sem_addr = 0
        semaphore_index = 0
        if not skip_ccl:
            out_ready_semaphore = attention_block_semaphores[semaphore_index]
            semaphore_index += 1
            barrier_semaphore = attention_block_semaphores[semaphore_index]
            semaphore_index += 1
            secondary_sync_semaphore = attention_block_semaphores[semaphore_index]
            semaphore_index += 1
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

        # Matmul1 grid from OverlappedTensor metadata (single packed weight tensor)
        matmul_weights_core_grid = matmul_weights_tensor.core_range_set
        if len(list(matmul_weights_core_grid.ranges())) != 1:
            raise ValueError("matmul weights core grid must be a single rectangular range for packed K-split")
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

        # Calculate per-core width in tiles for matmul1 (from OverlappedTensor)
        matmul_weights_shard_shape = matmul_weights_tensor.shard_shape
        matmul_weights_shard_width = matmul_weights_shard_shape[1]  # Width dimension
        matmul_out_w = matmul_weights_shard_width // matmul_weights_tensor.tile_shape[1]  # Per-core width in tiles

        # Calculate per-core width in tiles for matmul2 (from OverlappedTensor)
        matmul2_weights_core_grid = matmul2_weights_tensor.core_range_set
        matmul2_weights_shard_shape = matmul2_weights_tensor.shard_shape
        matmul2_weights_shard_width = matmul2_weights_shard_shape[1]  # Width dimension
        matmul2_out_w = matmul2_weights_shard_width // matmul2_weights_tensor.tile_shape[1]  # Per-core width in tiles

        # Extract matmul3 weights core grid (for inferring QNOPE grid dimensions)
        matmul3_weights_core_grid = matmul3_weights_tensor.core_range_set

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

        dkv_rmsnorm_grid = dkv_rmsnorm_gamma_tensor.core_range_set

        # Krope grid: columns 8-9, rows 8-9 (2 cores total)
        krope_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(8, 8),
                    ttnn.CoreCoord(8, 9),
                )
            }
        )

        kv_cache_update_grid = dkv_rmsnorm_grid.merge(krope_grid)
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
        dkv_matmul_weights_core_grid = dkv_matmul_weights_tensor.core_range_set

        # Calculate per-core width in tiles for dkv matmul (from overlapped tensor shard spec)
        dkv_matmul_weights_shard_shape = dkv_matmul_weights_tensor.shard_shape
        dkv_matmul_weights_shard_width = dkv_matmul_weights_shard_shape[1]  # Width dimension
        dkv_matmul_out_w = (
            dkv_matmul_weights_shard_width // dkv_matmul_weights_tensor.tile_shape[1]
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

        TILE_1x32 = ttnn.Tile((1, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)
        # Mcast setup: sender core (rmsnorm) -> full mcast grid
        # Only matmul cores (is_matmul_core=true) will actually participate in receive

        # Get NOC coordinates for mcast destination
        mcast_dest_noc_start_core = device.worker_core_from_logical_core(main_grid.start)
        mcast_dest_noc_end_core = device.worker_core_from_logical_core(main_grid.end)

        # Calculate number of mcast cores (full grid)
        mcast_num_cores = main_grid.grid_size().x * main_grid.grid_size().y
        mcast_is_part_of_receiver_grid = main_grid.contains(rmsnorm_core_grid)

        # Post-SDPA Parameters
        # ========================================================================
        # Core grid configuration — derived from production weight overlap specs
        # ========================================================================
        # Matmul4 grid: kv_b2 cores (5×8 + 12×2 = 64 cores)
        kv_b12_spec = KVB12_PROJ_SingleDeviceOverlapSpec()
        matmul4_core_grid = kv_b12_spec.kv_b2_core_range_set
        num_matmul4_cores = matmul4_core_grid.num_cores()  # 64

        # Per-core gather2 sender index: contiguous 0..63 in row-major order.
        # Needed because kv_b2 grid is non-rectangular (5×8 + 12×2).
        matmul4_cores = ttnn.corerange_to_cores(matmul4_core_grid, row_wise=True)
        gather2_sender_idx_per_core = [(core, idx) for idx, core in enumerate(matmul4_cores)]

        # Gather/CCL receiver core: (12, 9)
        gather_core = ttnn.CoreCoord(12, 9)
        gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

        # CCL sender core: (11, 9) - adjacent to gather core
        ccl_sender_core = ttnn.CoreCoord(11, 9)
        ccl_sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ccl_sender_core, ccl_sender_core)])

        mcast3_grid = ttnn.CoreRange(
            ttnn.CoreCoord(MCAST_GRID_START_X, MCAST_GRID_START_Y),
            ttnn.CoreCoord(MCAST_GRID_END_X, MCAST_GRID_END_Y),
        )
        mcast3_core_grid = ttnn.CoreRangeSet([mcast3_grid])
        num_mcast3_cores = mcast3_grid.grid_size().x * mcast3_grid.grid_size().y  # 130

        # Active Matmul5 cores: o_proj cores (12×8 + 8×2 = 112 cores)
        o_proj_spec = O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec()
        matmul5_active_core_grid = o_proj_spec.o_proj_core_range_set
        num_matmul5_cores = matmul5_active_core_grid.num_cores()  # 112

        # Per-core gather3 sender index: contiguous 0..111 in row-major order.
        # Needed because o_proj grid is non-rectangular (12×8 + 8×2).
        # Row-major (y then x) matches WIDTH_SHARDED shard placement order.
        matmul5_cores = ttnn.corerange_to_cores(matmul5_active_core_grid, row_wise=True)
        gather3_sender_idx_per_core = [(core, idx) for idx, core in enumerate(matmul5_cores)]

        # Full grid (union of all cores for semaphore allocation)
        full_grid = matmul4_core_grid.merge(gather_core_grid).merge(mcast3_core_grid).merge(ccl_sender_core_grid)

        # SDPA tensor properties - use same calculation as original sdpa_reduce_to_all op
        sdpa_input_l_sample = ttnn.get_device_tensors(sdpa_input_l_mesh)[0]

        sdpa_worker_grid = sdpa_input_l_sample.memory_config().shard_spec.grid

        # SDPA forwarder grid: derived from the scratch buffer's shard spec so that
        # the kernel runs on exactly the cores where the scratch buffer is allocated.
        assert (
            sdpa_forwarder_scratch_mesh is not None
        ), "sdpa_forwarder_scratch_mesh must be provided when sdpa_enabled=True"
        sdpa_forwarder_scratch_sample = ttnn.get_device_tensors(sdpa_forwarder_scratch_mesh)[0]
        sdpa_forwarder_grid = sdpa_forwarder_scratch_sample.memory_config().shard_spec.grid
        sdpa_forwarder_cores = list(ttnn.corerange_to_cores(sdpa_forwarder_grid, row_wise=True))

        # Add SDPA cores to full grid (workers and forwarders both part of unified kernel)
        full_grid = full_grid.merge(sdpa_worker_grid).merge(sdpa_forwarder_grid)

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

        # SDPA scatter parameters (scatter output to matmul4 cores)
        # Each SDPA worker scatters rows to matmul4 cores (one row per tile when using 8x32 tiles)
        sdpa_scatter_num_rows = sdpa_tile_height  # Rows per tile (8 for 8x32 tiles)
        sdpa_scatter_num_tiles = sdpa_l_tiles_per_worker  # Tiles per worker
        sdpa_scatter_src_tile_size = sdpa_l_tile_size  # Source tile size
        sdpa_scatter_dst_tile_size = tile_1x32_size  # 1x32 destination (row-extracted)
        # Face layout: each 8x32 tile has 2 faces of tile_height × (tile_width/2) = 8 × 16.
        # face_size = stride from Face 0 to Face 1 in source tile
        # row_face_size = one row within a face (also = dest face size for 1×32 dest tiles)
        sdpa_scatter_face_size = sdpa_tile_height * (sdpa_tile_width // 2) * sdpa_element_size_bytes
        sdpa_scatter_row_face_size = 1 * (sdpa_tile_width // 2) * sdpa_element_size_bytes  # dest_h=1 for 1x32

        # SDPA forwarder parameters (using Type A/B worker split like original op)
        sdpa_num_workers = 8
        sdpa_num_forwarders = 2
        sdpa_workers_per_forwarder = sdpa_num_workers // sdpa_num_forwarders  # 4
        sdpa_workers_per_type = sdpa_workers_per_forwarder // 2  # 2 (Type A and Type B alternate)
        sdpa_slots_per_worker = 1 + sdpa_num_l_chunks  # MS + L chunks = 5 slots
        sdpa_fwd_slots_per_round = sdpa_workers_per_type * sdpa_slots_per_worker  # 2 * 5 = 10 slots per direction
        sdpa_fwd_slot_size = (
            ttnn.get_tt_fabric_packet_header_size_bytes() + sdpa_l_chunk_size_bytes
        )  # Header + max payload
        sdpa_fwd_r2_buffer_offset = sdpa_fwd_slots_per_round * sdpa_fwd_slot_size

        # SDPA semaphore ID for scatter arrival (new semaphore)
        scatter_arrival_semaphore_id = 7  # After existing semaphores 0-6

        # SDPA forwarder semaphore IDs (on forwarder cores, signaled by workers)
        sdpa_fwd_r1_sem_id = 8
        sdpa_fwd_r2_sem_id = 9
        sdpa_bwd_r1_sem_id = 10
        sdpa_bwd_r2_sem_id = 11

        # Convert scale to FP32 bits
        import struct

        sdpa_scale_fp32_bits = struct.unpack(">I", struct.pack(">f", sdpa_scale_fp32))[0]

        # ========================================================================
        # Matmul4 parameters: [1, 512] x [512, 128] -> [1, 128]
        # ========================================================================
        matmul4_k_num_tiles = 16  # 512 / 32 = 16 tiles
        matmul4_out_w_per_core = 4  # 128 / 32 = 4 tiles per core

        # ========================================================================
        # Matmul5 parameters: [1, 8192] x [8192, 64] -> [1, 64]
        # ========================================================================
        matmul5_k_num_tiles = 256  # 8192 / 32 = 256 tiles
        matmul5_out_w_per_core = 2  # 64 / 32 = 2 tiles per core

        # ========================================================================
        # Gather2 parameters: 64 cores -> [1, 8192]
        # ========================================================================
        gather2_data_size_bytes = matmul4_out_w_per_core * tile_1x32_size
        gather2_src_num_pages = matmul4_out_w_per_core  # 4 pages per sender
        gather2_dst_num_pages = num_matmul4_cores * matmul4_out_w_per_core  # 64 * 4 = 256 pages
        gather2_noc0_num_senders = num_matmul4_cores
        gather2_noc1_num_senders = 0

        # ========================================================================
        # Mcast3 parameters: [1, 8192] to 130 cores (13x10 grid)
        # ========================================================================
        mcast3_data_size_bytes = gather2_dst_num_pages * tile_1x32_size  # 256 * 64 = 16384 bytes
        mcast3_src_num_pages = gather2_dst_num_pages  # 256 pages
        mcast3_dst_num_pages = gather2_dst_num_pages  # 256 pages per receiver

        # ========================================================================
        # Gather3 parameters: 112 cores -> [1, 7168]
        # ========================================================================
        gather3_data_size_bytes = matmul5_out_w_per_core * tile_1x32_size
        gather3_src_num_pages = matmul5_out_w_per_core  # 2 pages per sender
        gather3_dst_num_pages = num_tiles  # Same as input tensor
        gather3_noc0_num_senders = num_matmul5_cores
        gather3_noc1_num_senders = 0

        # ========================================================================
        # CCL parameters: [1, 7168] all-reduce
        # ========================================================================
        # Using 32x32 tiles to match gather3 output format (for tile-compatible reduction)
        # 7168 elements = 7 tiles of 32x32 (1024 elements each)
        ccl_num_tiles = gather3_dst_num_pages  # 7 tiles of 32x32
        ccl_page_size_bytes = tile_size  # 32x32 tile size
        ccl_num_pages = gather3_dst_num_pages  # 7 pages of 32x32
        ccl_payload_size_bytes = ccl_num_pages * ccl_page_size_bytes  # 7 * 2048 = 14336 bytes
        ccl_packet_header_size_bytes = ttnn.get_tt_fabric_packet_header_size_bytes()
        l1_alignment = 16

        has_residual = 1

        # ========================================================================
        # Semaphore IDs
        # ========================================================================
        gather2_noc0_receiver_semaphore_id = 0
        gather2_noc1_receiver_semaphore_id = 1
        mcast3_data_receiver_semaphore_id = 2
        gather3_noc0_receiver_semaphore_id = 3
        gather3_noc1_receiver_semaphore_id = 4
        gather3_completion_semaphore_id = 5  # Gather3 signals, CCL sender waits

        # Semaphore IDs for mcast synchronization
        mcast_data_sender_semaphore_addr = ttnn.get_global_semaphore_address(
            attention_block_semaphores[semaphore_index]
        )
        semaphore_index += 1
        mcast_data_receiver_semaphore_addr = ttnn.get_global_semaphore_address(
            attention_block_semaphores[semaphore_index]
        )
        semaphore_index += 1

        mcast3_data_sender_semaphore_addr = mcast_data_sender_semaphore_addr

        # Semaphore IDs for gather synchronization
        # Senders on NCRISC use NOC_0, receiver on BRISC uses NOC_1
        # Only use noc0 semaphore since senders are on NOC_0 (default for NCRISC)
        gather_noc0_receiver_semaphore_addr = ttnn.get_global_semaphore_address(
            attention_block_semaphores[semaphore_index]
        )
        semaphore_index += 1
        gather_noc1_receiver_semaphore_addr = ttnn.get_global_semaphore_address(
            attention_block_semaphores[semaphore_index]
        )
        semaphore_index += 1
        # Gather-reduce for matmul path reuses gather semaphore IDs
        gather_reduce_noc0_receiver_semaphore_addr = gather_noc0_receiver_semaphore_addr
        gather_reduce_noc1_receiver_semaphore_addr = gather_noc1_receiver_semaphore_addr

        # CreateQHeads 3-phase semaphore IDs (reuse existing IDs, safe since prior ops have completed)
        # Phase 1: QNOPE first halves, Phase 2: QNOPE second halves, Phase 3: QROPE
        nope_phase1_semaphore_addr = gather_noc0_receiver_semaphore_addr  # ID 2
        nope_phase2_semaphore_addr = gather_noc1_receiver_semaphore_addr  # ID 3
        rope_semaphore_addr = mcast_data_sender_semaphore_addr  # ID 0 (mcast completed before CreateQHeads)

        # Semaphore IDs for MLA
        mla_reducer_semaphore_addr = ttnn.get_global_semaphore_address(attention_block_semaphores[semaphore_index])
        semaphore_index += 1
        mla_mcast_semaphore_addr = ttnn.get_global_semaphore_address(attention_block_semaphores[semaphore_index])
        semaphore_index += 1
        mla_ncrisc_brisc_sync_semaphore_addr = ttnn.get_global_semaphore_address(
            attention_block_semaphores[semaphore_index]
        )
        semaphore_index += 1
        mla_receiver_ready_semaphore_addr = ttnn.get_global_semaphore_address(
            attention_block_semaphores[semaphore_index]
        )
        semaphore_index += 1
        mla_q_input_mcast_semaphore_addr = ttnn.get_global_semaphore_address(
            attention_block_semaphores[semaphore_index]
        )
        semaphore_index += 1
        mla_kv_cache_cur_pos_ready_semaphore_addr = ttnn.get_global_semaphore_address(
            attention_block_semaphores[semaphore_index]
        )
        semaphore_index += 1

        # Post-SDPA semaphores
        sdpa_semaphore1 = attention_block_semaphores[semaphore_index]
        semaphore_index += 1
        sdpa_semaphore2 = attention_block_semaphores[semaphore_index]
        semaphore_index += 1
        sdpa_semaphore1_addr = ttnn.get_global_semaphore_address(sdpa_semaphore1)
        sdpa_semaphore2_addr = ttnn.get_global_semaphore_address(sdpa_semaphore2)
        ccl_semaphore1 = attention_block_semaphores[semaphore_index]
        semaphore_index += 1
        ccl_semaphore2 = attention_block_semaphores[semaphore_index]
        semaphore_index += 1
        ccl_semaphore1_addr = ttnn.get_global_semaphore_address(ccl_semaphore1)
        ccl_semaphore2_addr = ttnn.get_global_semaphore_address(ccl_semaphore2)

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

        q_df = data_format
        k_df = kv_cache_tensor.dtype
        stats_df = ttnn.bfloat16
        untilize_df = ttnn.bfloat16

        # ==================================================================
        # CB indices (auto-assigned via CircularBufferIdManager)
        # ==================================================================
        assert cb_id_context is not None, "cb_id_context must be provided"

        TD_INTERP = ttnn.TileDescriptor(interpreted_tile)
        TD_1x32 = ttnn.TileDescriptor(TILE_1x32)
        TD_16x32 = ttnn.TileDescriptor(HALF_16x32_TILE)
        TD_32x32 = ttnn.TileDescriptor(FULL_32x32_TILE)
        TD_8x32 = ttnn.TileDescriptor(ttnn.Tile((8, 32)))
        TD_SDPA = ttnn.TileDescriptor(sdpa_tile)
        TD_KV = ttnn.TileDescriptor(kv_cache_tensor.get_tile())

        # CB indices (grouped by stage)
        input_cb = cb_id_context.get_cb_id(data_format, TD_INTERP)
        gamma_cb = cb_id_context.get_cb_id(data_format, TD_INTERP)
        rmsnorm_output_cb = cb_id_context.get_cb_id(data_format, TD_INTERP)
        # Matmul1 + gather-reduce + RMSNorm2 path
        matmul_weights_cb_overlapped = cb_id_context.get_cb_id(
            matmul_weights_tensor.dtype, ttnn.TileDescriptor(matmul_weights_tensor.get_tile())
        )
        matmul_output_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
        matmul_input_cb = cb_id_context.get_cb_id(data_format, TD_1x32)
        rmsnorm2_gamma_cb = cb_id_context.get_cb_id(
            data_format, TD_16x32
        )  # Gamma for second RMSNorm (1536 elements = 3 tiles of 16x32)
        rmsnorm2_input_cb = cb_id_context.get_cb_id(data_format, TD_16x32)  # Input CB for RMSNorm2
        gather_reduce_half1_scratch_cb = cb_id_context.get_cb_id(
            data_format, TD_16x32
        )  # Dedicated half1 scratch CB for gather_reduce
        rmsnorm2_output_cb = cb_id_context.get_cb_id(data_format, TD_16x32)  # Output CB for RMSNorm2
        # Matmul2 + Matmul3 + QRoPE/CreateQHeads path
        matmul2_input_cb = cb_id_context.get_cb_id(
            data_format, TD_1x32
        )  # Input CB for second matmul (1x1536 with 1x32 tiles)
        matmul2_output_cb = cb_id_context.get_cb_id(
            data_format, TD_1x32
        )  # Output CB for second matmul ([64, 1, 128] + [64, 1, 64])
        matmul3_weights_cb = cb_id_context.get_cb_id(  # Weights CB for third matmul (height sharded on Qnope grid)
            matmul3_weights_tensor.dtype, ttnn.TileDescriptor(matmul3_weights_tensor.get_tile())
        )
        matmul3_output_cb = cb_id_context.get_cb_id(
            data_format, TD_1x32
        )  # Output CB for third matmul (Qnope final output)
        qrope_output_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Output CB for Qrope (RoPE output)
        create_q_heads_out_cb = cb_id_context.get_cb_id(
            data_format, TD_8x32
        )  # Output CB for CreateQHeads (linked to output tensor on receiver cores)
        qrope_cos_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Cos CB for RoPE
        qrope_sin_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Sin CB for RoPE
        qrope_trans_mat_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Trans_mat CB for RoPE
        qrope_rotated_input_interm_cb = cb_id_context.get_cb_id(
            data_format, TD_1x32
        )  # Rotated input intermediate CB for RoPE
        qrope_cos_interm_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Cos intermediate CB for RoPE
        qrope_sin_interm_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Sin intermediate CB for RoPE
        # KV cache branch
        dkv_matmul_output_cb = cb_id_context.get_cb_id(
            data_format, TD_1x32
        )  # DKV Matmul output CB, 64 bytes (1 tile per core for rope input)
        kv_rmsnorm_input_cb = cb_id_context.get_cb_id(data_format, TD_16x32)  # Input CB for KV Cache Branch RMSNorm
        kv_rmsnorm_gamma_cb = cb_id_context.get_cb_id(data_format, TD_16x32)  # Gamma CB for KV Cache Branch RMSNorm
        kv_rmsnorm_output_cb = cb_id_context.get_cb_id(data_format, TD_16x32)  # Output CB for KV Cache Branch RMSNorm
        krope_output_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Output CB for KV Cache Branch RoPE
        krope_cos_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Cos CB for RoPE
        krope_sin_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Sin CB for RoPE
        create_q_heads_receiver_in_cb = cb_id_context.get_cb_id(
            data_format, TD_8x32
        )  # Intermediate CB for CreateQHeads (row-major data before tilization)

        kv_cache_output_cb = cb_id_context.get_cb_id(k_df, TD_32x32)  # Output CB for KV Cache Branch
        kv_cache_intermed_cb = cb_id_context.get_cb_id(untilize_df, TD_32x32)  # Intermed CB for KV Cache Branch
        kv_cache_input_cb = cb_id_context.get_cb_id(k_df, TD_32x32)  # Input CB for KV Cache Branch

        # MLA parameters
        mla_q_in_cb = create_q_heads_out_cb  # In for MLA q heads
        mla_k_in_cb = cb_id_context.get_cb_id(k_df, TD_KV)  # Input CB for MLA
        mla_interm_out_cb = cb_id_context.get_cb_id(stats_df, TD_8x32)  # Intermediate output CB for MLA
        mla_interm_ms_cb = cb_id_context.get_cb_id(stats_df, TD_8x32)  # Intermediate MS CB for MLA
        mla_out_in_cb = cb_id_context.get_cb_id(stats_df, TD_8x32)  # Output input CB for MLA
        mla_ms_in_cb = cb_id_context.get_cb_id(stats_df, TD_8x32)  # Output MS CB for MLA
        mla_out_o_cb = cb_id_context.get_cb_id(stats_df, TD_8x32)  # Output O CB for MLA
        mla_out_ms_cb = cb_id_context.get_cb_id(stats_df, TD_8x32)  # Output MS CB for MLA
        mla_mask_cb = cb_id_context.get_cb_id(q_df, TD_8x32)  # Mask CB for MLA
        mla_out_final_cb = mla_out_o_cb  # Output final CB for MLA, unused for full fused attention block

        # CB indices for CCL broadcast (use separate CBs to avoid conflicts)
        bcast_pkt_cb = cb_id_context.get_cb_id(data_format, TD_INTERP)  # Packet buffer for CCL broadcast

        # SDPA CB indices
        sdpa_cb_local_l = mla_out_o_cb
        sdpa_cb_local_ms = mla_out_ms_cb
        sdpa_cb_neighbor_l = cb_id_context.get_cb_id(data_format, TD_SDPA)
        sdpa_cb_neighbor_ms = cb_id_context.get_cb_id(data_format, TD_SDPA)
        sdpa_cb_r1_result_l = cb_id_context.get_cb_id(data_format, TD_SDPA)
        sdpa_cb_r1_result_ms = cb_id_context.get_cb_id(data_format, TD_SDPA)
        sdpa_cb_l_out = cb_id_context.get_cb_id(data_format, TD_SDPA)
        sdpa_cb_packet_slot = cb_id_context.get_cb_id(ttnn.uint32, TD_32x32)

        matmul4_in0_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Matmul4 input (kv_b2 grid)
        matmul4_in1_cb = cb_id_context.get_cb_id(  # Matmul4 weights (kv_b2 grid)
            post_sdpa_weights1_tensor.dtype, ttnn.TileDescriptor(post_sdpa_weights1_tensor.get_tile())
        )
        matmul4_out_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Matmul4 output (kv_b2 grid)
        gather2_dst_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Gather2 output = Mcast3 source (gather core)
        matmul5_in0_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Mcast3 dst = Matmul5 input (13x10 mcast3 grid)
        matmul5_in1_cb = cb_id_context.get_cb_id(  # Matmul5 weights (112 active cores)
            post_sdpa_weights2_tensor.dtype, ttnn.TileDescriptor(post_sdpa_weights2_tensor.get_tile())
        )
        matmul5_out_cb = cb_id_context.get_cb_id(data_format, TD_1x32)  # Matmul5 output (112 active cores)
        gather3_dst_cb = cb_id_context.get_cb_id(
            data_format, TD_INTERP
        )  # Gather3 output = CCL local data (gather core)
        ccl_sender_in_cb = cb_id_context.get_cb_id(
            data_format, TD_INTERP
        )  # CCL sender reads gather3 output (sender core)
        ccl_remote_data_cb = cb_id_context.get_cb_id(data_format, TD_INTERP)  # CCL received remote data (receiver core)
        ccl_residual_cb = input_cb
        ccl_temp_cb = cb_id_context.get_cb_id(data_format, TD_INTERP)  # CCL temp for compute (receiver core)
        ccl_output_cb = cb_id_context.get_cb_id(data_format, TD_INTERP)  # CCL output (receiver core)
        ccl_packet_header_cb = cb_id_context.get_cb_id(
            ttnn.uint32, TD_32x32
        )  # CCL packet headers (sender + receiver cores)

        attention_block_output_cb = ccl_output_cb  # Attention block output (receiver core)

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
            ("mcast_data_sender_semaphore_addr", mcast_data_sender_semaphore_addr),
            ("mcast_data_receiver_semaphore_addr", mcast_data_receiver_semaphore_addr),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", rmsnorm_output_cb),
            ("mcast_dst_cb", matmul_input_cb),
            ("mcast_src_num_pages", mcast_src_num_pages),
            ("mcast_is_part_of_receiver_grid", mcast_is_part_of_receiver_grid),
        ]

        # Mcast receiver compile-time args (named args for NCRISC)
        mcast_receiver_named_compile_time_args = [
            ("mcast_data_receiver_semaphore_addr", mcast_data_receiver_semaphore_addr),
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
            ("matmul_in1", matmul_weights_cb_overlapped),
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
            ("matmul_in1", matmul_weights_cb_overlapped),
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
            ("matmul2_in1", matmul_weights_cb_overlapped),
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
            ("matmul2_in1", matmul_weights_cb_overlapped),
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
        matmul3_weights_shard_shape = matmul3_weights_tensor.shard_shape
        matmul3_weights_shard_width = matmul3_weights_shard_shape[1]  # Width dimension (512)
        matmul3_out_w = matmul3_weights_shard_width // matmul3_weights_tensor.tile_shape[1]  # 512/32 = 16 tiles

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
            ("cqh_nope_phase1_semaphore_addr", nope_phase1_semaphore_addr),
            ("cqh_nope_phase2_semaphore_addr", nope_phase2_semaphore_addr),
            ("cqh_rope_semaphore_addr", rope_semaphore_addr),
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
            ("cqh_nope_phase1_semaphore_addr", nope_phase1_semaphore_addr),
            ("cqh_nope_phase2_semaphore_addr", nope_phase2_semaphore_addr),
            ("cqh_rope_semaphore_addr", rope_semaphore_addr),
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
            ("gather_reduce_receiver_semaphore_addr", gather_reduce_noc0_receiver_semaphore_addr),
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
            ("gather_reduce_noc0_receiver_semaphore_addr", gather_reduce_noc0_receiver_semaphore_addr),
            ("gather_reduce_noc1_receiver_semaphore_addr", gather_reduce_noc1_receiver_semaphore_addr),
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
            ("dkv_matmul_in1", matmul_weights_cb_overlapped),
            ("dkv_matmul_k_num_tiles", dkv_matmul_k_num_tiles),
            ("dkv_matmul_out_w_per_core", dkv_matmul_out_w),
        ]
        dkv_matmul_trisc_named_compile_time_args = [
            (
                "dkv_matmul_in0",
                matmul_input_cb,
            ),  # Inputs are multicasted from the main branch, same input as first matmul
            ("dkv_matmul_in1", matmul_weights_cb_overlapped),
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
        dkv_gather_receiver_core = dkv_rmsnorm_gamma_tensor.core_range_set.ranges()[0].start
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
            ("dkv_gather_receiver_semaphore_addr", gather_noc0_receiver_semaphore_addr),
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
            ("dkv_gather_noc0_receiver_semaphore_addr", gather_noc0_receiver_semaphore_addr),
            ("dkv_gather_noc1_receiver_semaphore_addr", gather_noc1_receiver_semaphore_addr),
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
            ("full_grid_mcast_start_x", mcast_dest_noc_start_core.x),
            ("full_grid_mcast_start_y", mcast_dest_noc_start_core.y),
            ("full_grid_mcast_end_x", mcast_dest_noc_end_core.x),
            ("full_grid_mcast_end_y", mcast_dest_noc_end_core.y),
            ("full_grid_mcast_num_dests", mcast_num_cores - 1),
            ("kv_cache_cur_pos_ready_semaphore_addr", mla_kv_cache_cur_pos_ready_semaphore_addr),
        ]
        kv_cache_trisc_named_compile_time_args = [
            ("kv_rmsnorm_output_cb", kv_rmsnorm_output_cb),
            ("kv_cache_output_cb", kv_cache_output_cb),
            ("kv_cache_input_cb", kv_cache_input_cb),
            ("kv_cache_intermed_cb", kv_cache_intermed_cb),
        ]

        # Flash MLA named compile-time args
        # a lot of the setup is reused from the FlashMLADecode op
        flash_mla_program_config = FlashMLADecode.ProgramConfig()
        device_chunk_size = flash_mla_program_config.device_chunk_size
        k_chunk_size = flash_mla_program_config.k_chunk_size
        num_q_heads_per_core = 8
        k_shape = kv_cache_tensor.padded_shape
        kv_cache_mem_config = kv_cache_tensor.memory_config()
        kv_shard_shape = kv_cache_mem_config.nd_shard_spec.shard_shape
        assert kv_cache_mem_config.is_sharded(), "KV cache must be ND sharded (not interleaved)"
        assert (
            hasattr(kv_cache_mem_config, "nd_shard_spec") and kv_cache_mem_config.nd_shard_spec is not None
        ), "KV cache must use ND sharding with nd_shard_spec"

        # Validate k_chunk_size matches KV cache shard height
        kv_shard_shape = kv_cache_mem_config.nd_shard_spec.shard_shape
        kv_shard_height = kv_shard_shape[2]  # Shape is [batch, nkv, seq_len, head_dim]
        assert kv_shard_height == k_chunk_size, (
            f"k_chunk_size ({k_chunk_size}) must match KV cache shard height ({kv_shard_height}). "
            f"Each KV shard should contain exactly one k_chunk."
        )

        S = k_shape[2]
        DH = k_shape[3]

        # number of Q shards
        B = sdpa_input_grid.num_cores()  # 8 shards
        assert QROPE_HEAD_DIM == B * num_q_heads_per_core

        Q_TILE_HEIGHT = num_q_heads_per_core
        K_TILE_HEIGHT = 32
        TILE_WIDTH = 32

        num_kv_heads = k_shape[1]
        Bkv = k_shape[0]
        St = S // K_TILE_HEIGHT  # K/V use standard tile height
        DHt = DH // TILE_WIDTH
        vDHt = QNOPE_DATA_SIZE // TILE_WIDTH
        PNHt = num_q_heads_per_core // Q_TILE_HEIGHT  # Q uses its own tile height

        Sk_chunk_t = k_chunk_size // K_TILE_HEIGHT  # K chunks use K tile height

        optimized_mla_grid = flash_mla_program_config.grid
        num_s_blocks = optimized_mla_grid.NUM_BLOCKS
        cores_per_s_block = optimized_mla_grid.CORES_PER_BLOCK

        k_tile_size = kv_cache_tensor.get_tile().get_tile_size(kv_cache_tensor.dtype)
        k_chunk_tiles = Sk_chunk_t * DHt
        noc_max_page_size = get_noc_max_page_size()
        k_page_size, k_num_pages = get_max_page_size_and_num_pages(noc_max_page_size, k_chunk_tiles, k_tile_size)

        # Validate Q shards fit in one S block (max 8 Q shards)
        assert B <= cores_per_s_block, f"Too many Q shards ({B}), max is {cores_per_s_block}"

        # Calculate parallelization parameters
        # Each batch (Q shard) gets 8 cores: 1 from each S block
        num_cores_per_batch = num_s_blocks  # 8 cores per Q shard (seq len parallelism)
        num_active_mla_cores = B * num_cores_per_batch  # Total active cores
        num_cores_per_head = num_cores_per_batch // num_kv_heads  # Cores per KV head
        num_heads_per_core = max(1, math.ceil(num_kv_heads / num_cores_per_batch))

        assert PNHt == 1, f"PNHt must be 1, got {PNHt}"
        assert num_kv_heads == 1, f"num_kv_heads must be 1, got {num_kv_heads}"
        assert num_heads_per_core == 1, f"num_heads_per_core must be 1, got {num_heads_per_core}"
        assert Bkv == 1, f"Bkv must be 1, got {Bkv}"

        if fp32_dest_acc_en:
            mla_dst_size = 8 if fp32_dest_acc_en else 16
        else:
            mla_dst_size = 4 if fp32_dest_acc_en else 8

        assert mla_dst_size >= 8, f"mla_dst_size must be >= 8, got {mla_dst_size}"

        q_tiles = PNHt * DHt
        q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))
        q_tile_size = q_tiny_tile.get_tile_size(data_format)
        q_chunk_size_bytes = q_tiles * q_tile_size

        # Double buffer K for overlap between DRAM reads and compute.
        # Receivers signal sender when ready (CB reserved) to ensure consistent addresses.
        k_tiles = Sk_chunk_t * DHt * 2

        out0_t = PNHt * vDHt
        statistics_tiles = PNHt

        s1_cores, _ = FlashMLADecode.ProgramConfig.grid.BLOCKS[0]
        mla_input_output_crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in s1_cores]
        )

        # This gives the interleaved layout needed for parallelization
        mla_all_cores = []
        for batch_idx in range(B):
            for s_block_idx in range(num_s_blocks):
                x, y = optimized_mla_grid.get_cores(s_block_idx)[batch_idx]
                mla_all_cores.append((x, y))

        # Build core grid from all active cores
        mla_core_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in mla_all_cores]
        )

        # Create core group with the same layout
        mla_core_group = [ttnn.CoreCoord(x, y) for x, y in mla_all_cores]

        # Multicast: each S block's first core (Q1) reads KV and multicasts to others in that S block
        # Since all S blocks have 8 cores, num_mcast_dests is the same for all (7 = 8-1)
        num_mcast_dests = cores_per_s_block - 1  # 7 receivers per S block

        mla_brisc_named_compile_time_args = [
            ("vDHt", vDHt),
            ("Sk_chunk_t", Sk_chunk_t),
            ("num_cores_per_head", num_cores_per_head),
            ("mla_reducer_semaphore_addr", mla_reducer_semaphore_addr),
            ("k_chunk_size", k_chunk_size),
            ("q_chunk_size_bytes", q_chunk_size_bytes),
            ("DHt", DHt),
            ("num_mcast_dests", num_mcast_dests),
            ("full_grid_mcast_start_x", mcast_dest_noc_start_core.x),
            ("full_grid_mcast_start_y", mcast_dest_noc_start_core.y),
            ("full_grid_mcast_end_x", mcast_dest_noc_end_core.x),
            ("full_grid_mcast_end_y", mcast_dest_noc_end_core.y),
            ("full_grid_mcast_num_dests", mcast_num_cores - 1),
            ("mla_q_input_mcast_semaphore_addr", mla_q_input_mcast_semaphore_addr),
            ("mla_mcast_semaphore_addr", mla_mcast_semaphore_addr),
            ("k_num_pages", k_num_pages),
            ("k_page_size", k_page_size),
            ("num_tree_reduction_steps", optimized_mla_grid.NUM_TREE_REDUCTION_STEPS),
            ("mla_receiver_ready_semaphore_addr", mla_receiver_ready_semaphore_addr),
            ("mla_ncrisc_brisc_sync_semaphore_addr", mla_ncrisc_brisc_sync_semaphore_addr),
            ("mla_k_in_cb", mla_k_in_cb),
            ("mla_q_in_cb", mla_q_in_cb),
            ("mla_mask_cb", mla_mask_cb),
            ("mla_out_in_cb", mla_out_in_cb),
            ("mla_ms_in_cb", mla_ms_in_cb),
            ("mla_out_o_cb", mla_out_o_cb),
            ("mla_out_ms_cb", mla_out_ms_cb),
        ]
        mla_ncrisc_named_compile_time_args = [
            ("St", St),
            ("DHt", DHt),
            ("Sk_chunk_t", Sk_chunk_t),
            ("num_cores_per_head", num_cores_per_head),
            ("k_chunk_size", k_chunk_size),
            ("mla_mcast_semaphore_addr", mla_mcast_semaphore_addr),
            ("k_page_size", k_page_size),
            ("k_num_pages", k_num_pages),
            ("mla_ncrisc_brisc_sync_semaphore_addr", mla_ncrisc_brisc_sync_semaphore_addr),
            ("mla_receiver_ready_semaphore_addr", mla_receiver_ready_semaphore_addr),
            ("mla_kv_cache_cur_pos_ready_semaphore_addr", mla_kv_cache_cur_pos_ready_semaphore_addr),
            ("mla_kv_cache_cur_pos_ready_value", kv_cache_update_grid.num_cores()),
            ("mla_k_in_cb", mla_k_in_cb),
        ]
        mla_trisc_named_compile_time_args = [
            ("St", St),
            ("DHt", DHt),
            ("vDHt", vDHt),
            ("PNHt", PNHt),
            ("Sk_chunk_t", Sk_chunk_t),
            ("k_chunk_size", k_chunk_size),
            ("num_cores_per_head", num_cores_per_head),
            ("q_heads_parallel_factor", B),
            ("scale_fp32", float_to_uint32(scale)),
            ("num_tree_reduction_steps", optimized_mla_grid.NUM_TREE_REDUCTION_STEPS),
            ("dst_size", mla_dst_size),
            ("mla_q_in_cb", mla_q_in_cb),
            ("mla_k_in_cb", mla_k_in_cb),
            ("mla_mask_cb", mla_mask_cb),
            ("mla_interm_out_cb", mla_interm_out_cb),
            ("mla_interm_ms_cb", mla_interm_ms_cb),
            ("mla_out_in_cb", mla_out_in_cb),
            ("mla_ms_in_cb", mla_ms_in_cb),
            ("mla_out_o_cb", mla_out_o_cb),
            ("mla_out_ms_cb", mla_out_ms_cb),
            ("mla_out_final_cb", mla_out_final_cb),
        ]

        # Get NOC coordinates for this device
        gather_dest_noc_core = device.worker_core_from_logical_core(gather_core)
        mcast3_dest_noc_start_core = device.worker_core_from_logical_core(mcast3_grid.start)
        mcast3_dest_noc_end_core = device.worker_core_from_logical_core(mcast3_grid.end)
        ccl_sender_noc_core = device.worker_core_from_logical_core(ccl_sender_core)
        ccl_receiver_noc_core = gather_dest_noc_core  # Same as gather core

        # Buffer addresses
        # TODO: is it possible to get these from CB write_ptrs?
        post_sdpa_gather2_output_tensor_device = post_sdpa_gather2_output_tensor.device()
        gather2_receiver_data_addr = post_sdpa_gather2_output_tensor.buffer_address()
        # Gather3 writes to gather3_output_tensor, CCL reads from there and writes to output_tensor
        post_sdpa_gather3_output_tensor_device_sample = post_sdpa_gather3_output_tensors_per_device[0]
        gather3_receiver_data_addr = post_sdpa_gather3_output_tensor_device_sample.buffer_address()

        mcast3_is_part_of_receiver_grid = mcast3_grid.contains(gather_core)

        # ========================================================================
        # NCRISC compile-time args
        # ========================================================================
        post_sdpa_ncrisc_named_compile_time_args = [
            # Matmul4
            ("matmul4_in0", matmul4_in0_cb),
            ("matmul4_in1", matmul4_in1_cb),
            ("matmul4_out", matmul4_out_cb),
            ("matmul4_k_num_tiles", matmul4_k_num_tiles),
            ("matmul4_out_w_per_core", matmul4_out_w_per_core),
            # Gather2 sender
            ("gather2_dest_noc_x", gather_dest_noc_core.x),
            ("gather2_dest_noc_y", gather_dest_noc_core.y),
            ("gather2_data_size_bytes", gather2_data_size_bytes),
            ("gather2_receiver_semaphore_id", gather2_noc0_receiver_semaphore_id),
            ("gather2_src_cb", matmul4_out_cb),
            ("gather2_src_num_pages", gather2_src_num_pages),
            ("gather2_sender_grid_start_x", 0),
            ("gather2_sender_grid_start_y", 0),
            ("gather2_sender_grid_end_x", 0),
            ("gather2_sender_grid_end_y", 0),
            ("gather2_row_major", 1),
            ("gather2_receiver_data_addr", gather2_receiver_data_addr),
            # Mcast3 receiver
            ("mcast3_data_receiver_semaphore", mcast3_data_receiver_semaphore_id),
            ("mcast3_dst_cb", matmul5_in0_cb),
            ("mcast3_dst_num_pages", mcast3_dst_num_pages),
            # Matmul5
            ("matmul5_in0", matmul5_in0_cb),
            ("matmul5_in1", matmul5_in1_cb),
            ("matmul5_out", matmul5_out_cb),
            ("matmul5_k_num_tiles", matmul5_k_num_tiles),
            ("matmul5_out_w_per_core", matmul5_out_w_per_core),
            # Gather3 sender
            ("gather3_dest_noc_x", gather_dest_noc_core.x),
            ("gather3_dest_noc_y", gather_dest_noc_core.y),
            ("gather3_data_size_bytes", gather3_data_size_bytes),
            ("gather3_receiver_semaphore_id", gather3_noc0_receiver_semaphore_id),
            ("gather3_src_cb", matmul5_out_cb),
            ("gather3_src_num_pages", gather3_src_num_pages),
            ("gather3_sender_grid_start_x", 0),
            ("gather3_sender_grid_start_y", 0),
            ("gather3_sender_grid_end_x", 0),
            ("gather3_sender_grid_end_y", 0),
            ("gather3_row_major", 1),
            ("gather3_receiver_data_addr", gather3_receiver_data_addr),
            # CCL sender (NCRISC reads from gather core)
            ("ccl_sender_cb0_id", ccl_sender_in_cb),
            ("ccl_sender_num_tiles", ccl_num_pages),
            ("ccl_sender_tensor_page_size", ccl_page_size_bytes),
            ("ccl_sender_data_noc_x", ccl_receiver_noc_core.x),
            ("ccl_sender_data_noc_y", ccl_receiver_noc_core.y),
            ("ccl_sender_gather3_completion_semaphore_id", gather3_completion_semaphore_id),
            # CCL receiver (NCRISC waits for remote data)
            ("ccl_receiver_packet_header_cb_id", ccl_packet_header_cb),
            ("ccl_receiver_cb_in1", ccl_remote_data_cb),
            ("ccl_receiver_l1_alignment", l1_alignment),
            ("ccl_receiver_cb_in2", gather3_dst_cb),  # Local data from gather3
            ("ccl_receiver_remote_sender_noc_x", ccl_sender_noc_core.x),
            ("ccl_receiver_remote_sender_noc_y", ccl_sender_noc_core.y),
            ("ccl_receiver_num_standard_tiles", ccl_num_tiles),
            ("ccl_receiver_cb_residual", ccl_residual_cb),
            ("ccl_receiver_has_residual", has_residual),
            ("ccl_receiver_skip_local_push", 1),  # Skip local push since gather3 already pushed to CB7
        ]

        # Add SDPA NCRISC compile-time args when enabled
        post_sdpa_ncrisc_named_compile_time_args.extend(
            [
                # SDPA CB indices
                ("sdpa_cb_local_l", sdpa_cb_local_l),
                ("sdpa_cb_local_ms", sdpa_cb_local_ms),
                ("sdpa_cb_neighbor_l", sdpa_cb_neighbor_l),
                ("sdpa_cb_neighbor_ms", sdpa_cb_neighbor_ms),
                # SDPA tile/chunk sizes
                ("sdpa_ms_tile_size_bytes", sdpa_ms_tile_size),
                ("sdpa_l_chunk_size_bytes", sdpa_l_chunk_size_bytes),
                ("sdpa_num_l_chunks", sdpa_num_l_chunks),
                ("sdpa_tiles_per_l_chunk", sdpa_tiles_per_l_chunk),
                # SDPA position
                ("sdpa_position_enabled", 1),
                ("sdpa_per_device_chunk_size", sdpa_per_device_chunk_size),
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
        post_sdpa_brisc_named_compile_time_args = [
            # Matmul4/2 (no-op)
            ("matmul4_out", matmul4_out_cb),
            ("matmul5_out", matmul5_out_cb),
            # Gather2 receiver
            ("gather2_noc0_num_senders", gather2_noc0_num_senders),
            ("gather2_noc1_num_senders", gather2_noc1_num_senders),
            ("gather2_noc0_receiver_semaphore_id", gather2_noc0_receiver_semaphore_id),
            ("gather2_noc1_receiver_semaphore_id", gather2_noc1_receiver_semaphore_id),
            ("gather2_dst_cb", gather2_dst_cb),
            ("gather2_dst_num_pages", gather2_dst_num_pages),
            # Mcast3 sender
            ("mcast3_dest_noc_start_x", mcast3_dest_noc_start_core.x),
            ("mcast3_dest_noc_start_y", mcast3_dest_noc_start_core.y),
            ("mcast3_dest_noc_end_x", mcast3_dest_noc_end_core.x),
            ("mcast3_dest_noc_end_y", mcast3_dest_noc_end_core.y),
            ("mcast3_num_cores", num_mcast3_cores),
            ("mcast3_data_sender_semaphore_addr", mcast3_data_sender_semaphore_addr),
            ("mcast3_data_receiver_semaphore", mcast3_data_receiver_semaphore_id),
            ("mcast3_data_size_bytes", mcast3_data_size_bytes),
            ("mcast3_src_cb", gather2_dst_cb),
            ("mcast3_src_num_pages", mcast3_src_num_pages),
            ("mcast3_dst_cb", matmul5_in0_cb),
            ("mcast3_is_part_of_receiver_grid", mcast3_is_part_of_receiver_grid),
            # Gather3 receiver
            ("gather3_noc0_num_senders", gather3_noc0_num_senders),
            ("gather3_noc1_num_senders", gather3_noc1_num_senders),
            ("gather3_noc0_receiver_semaphore_id", gather3_noc0_receiver_semaphore_id),
            ("gather3_noc1_receiver_semaphore_id", gather3_noc1_receiver_semaphore_id),
            ("gather3_dst_cb", gather3_dst_cb),
            ("gather3_dst_num_pages", gather3_dst_num_pages),
            # Gather3 completion signal for CCL sender synchronization
            ("gather3_completion_semaphore_id", gather3_completion_semaphore_id),
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
        post_sdpa_brisc_named_compile_time_args.extend(
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
        post_sdpa_trisc_named_compile_time_args = [
            # Matmul4
            ("matmul4_in0", matmul4_in0_cb),
            ("matmul4_in1", matmul4_in1_cb),
            ("matmul4_out", matmul4_out_cb),
            ("matmul4_k_num_tiles", matmul4_k_num_tiles),
            ("matmul4_out_w_per_core", matmul4_out_w_per_core),
            # Matmul5
            ("matmul5_in0", matmul5_in0_cb),
            ("matmul5_in1", matmul5_in1_cb),
            ("matmul5_out", matmul5_out_cb),
            ("matmul5_k_num_tiles", matmul5_k_num_tiles),
            ("matmul5_out_w_per_core", matmul5_out_w_per_core),
            # CCL receiver compute (reduction)
            ("ccl_receiver_cb_in0", ccl_remote_data_cb),
            ("ccl_receiver_cb_in1", gather3_dst_cb),  # Local data
            ("ccl_receiver_cb_out0", ccl_output_cb),
            ("ccl_receiver_cb_residual", ccl_residual_cb),
            ("ccl_receiver_cb_temp", ccl_temp_cb),
            ("ccl_receiver_has_residual", has_residual),
            ("ccl_receiver_num_tiles", ccl_num_tiles),
        ]

        # Add SDPA TRISC compile-time args when enabled
        post_sdpa_trisc_named_compile_time_args.extend(
            [
                # SDPA CB indices
                ("sdpa_cb_local_l", sdpa_cb_local_l),
                ("sdpa_cb_local_ms", sdpa_cb_local_ms),
                ("sdpa_cb_neighbor_l", sdpa_cb_neighbor_l),
                ("sdpa_cb_neighbor_ms", sdpa_cb_neighbor_ms),
                ("sdpa_cb_r1_result_l", sdpa_cb_r1_result_l),
                ("sdpa_cb_r1_result_ms", sdpa_cb_r1_result_ms),
                ("sdpa_cb_l_out", sdpa_cb_l_out),
                # SDPA compute params
                ("sdpa_scale_fp32", sdpa_scale_fp32_bits),
                ("sdpa_tiles_per_l_chunk", sdpa_tiles_per_l_chunk),
                ("sdpa_num_l_chunks", sdpa_num_l_chunks),
                # SDPA position
                ("sdpa_position_enabled", 1),
                ("sdpa_per_device_chunk_size", sdpa_per_device_chunk_size),
            ]
        )

        # Create tile descriptor for proper tile dimensions
        tile_descriptor = ttnn.TileDescriptor(interpreted_tile)

        # RMSNorm2 uses separate CBs with exact sizes (16x32 tiles)
        TILE_16x32 = ttnn.Tile((16, 32))
        rmsnorm2_tile_descriptor = ttnn.TileDescriptor(TILE_16x32)
        rmsnorm2_page_size = TILE_16x32.get_tile_size(data_format)

        per_device_contexts = []

        # ========================================================================
        # Kernel descriptors
        # ========================================================================

        for row in range(mesh_rows):
            # ("start of loop for device row {}".format(row))
            for col in range(mesh_cols):
                mesh_coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col

                # CCL role calculation (only matters if not skipping CCL)
                if skip_ccl:
                    is_sender = False
                    is_secondary_sender = False
                    is_receiver = False
                else:
                    is_sender = (row == sender_row) and (col == sender_col)
                    is_secondary_sender = (
                        bcast_secondary_cluster_axis is not None and (row == sender_row) and (col != sender_col)
                    )
                    is_receiver = not is_sender and not is_secondary_sender

                # Ring index along the cluster axis
                ring_index = row if reduce_cluster_axis == 0 else col
                is_first_chip = ring_index == 0

                # Determine CCL neighbor and semaphores based on position (only when CCL is enabled)
                ccl_sender_link = 0 if is_first_chip else 1
                ccl_receiver_link = 1 if is_first_chip else 0
                ccl_sender_semaphore_addr = ccl_semaphore1_addr if is_first_chip else ccl_semaphore2_addr
                ccl_receiver_semaphore_addr = ccl_semaphore2_addr if is_first_chip else ccl_semaphore1_addr

                # Calculate neighbor coordinate
                if is_first_chip:
                    neighbor_row = row + 1 if reduce_cluster_axis == 0 else row
                    neighbor_col = col if reduce_cluster_axis == 0 else col + 1
                else:
                    neighbor_row = row - 1 if reduce_cluster_axis == 0 else row
                    neighbor_col = col if reduce_cluster_axis == 0 else col - 1

                # Get the device's tensors
                input_tensor_device = input_tensors_per_device[device_idx]
                gamma_fused_tensor_device = gamma_fused_tensors_per_device[device_idx]
                fused_weights_tensor_device = fused_weights_tensors_per_device[device_idx]
                kv_b12_fused_tensor_device = kv_b12_fused_tensors_per_device[device_idx]
                qrope_cos_tensor_device = qrope_cos_tensors_per_device[device_idx]
                qrope_sin_tensor_device = qrope_sin_tensors_per_device[device_idx]
                trans_mat_tensor_device = trans_mat_tensors_per_device[device_idx]
                krope_cos_tensor_device = krope_cos_tensors_per_device[device_idx]
                krope_sin_tensor_device = krope_sin_tensors_per_device[device_idx]
                position_ids_tensor_device = position_ids_tensors_per_device[device_idx]
                kv_cache_tensor_device = kv_cache_tensors_per_device[device_idx]
                sdpa_kv_cache_buffer_device = sdpa_kv_cache_buffers_per_device[device_idx]
                sdpa_out_interm_buffer_device = sdpa_out_interm_buffers_per_device[device_idx]

                post_sdpa_weights1_fused_tensor_device = post_sdpa_weights1_fused_tensors_per_device[device_idx]
                post_sdpa_weights2_fused_tensor_device = post_sdpa_weights2_fused_tensors_per_device[device_idx]
                post_sdpa_gather3_output_tensor_device = post_sdpa_gather3_output_tensors_per_device[device_idx]
                post_sdpa_intermediate_tensor_device = post_sdpa_intermediate_tensors_per_device[device_idx]
                attention_block_output_tensor_device = attention_block_output_tensors_per_device[device_idx]

                # Get worker core from per-device input tensor shard grid
                device_local = input_tensor_device.device()
                device_input_shard_grid = input_tensor_device.memory_config().shard_spec.grid
                device_shard_grid_start = device_input_shard_grid.bounding_box().start
                broadcast_worker_core = ttnn.CoreCoord(device_shard_grid_start.x, device_shard_grid_start.y)
                broadcast_worker_core_set = ttnn.CoreRangeSet(
                    [ttnn.CoreRange(broadcast_worker_core, broadcast_worker_core)]
                )
                assert (
                    rmsnorm_core_grid == broadcast_worker_core_set
                ), "RMSNorm core grid does not match broadcast worker core"

                # Get physical core for NOC addressing
                data_core_physical = device_local.worker_core_from_logical_core(broadcast_worker_core)
                core_noc_x = data_core_physical.x
                core_noc_y = data_core_physical.y

                # Calculate ring index and targets for primary axis (column)
                ring_size = mesh_rows
                ring_index = row

                # For Linear topology, calculate forward and backward targets
                num_targets_forward = ring_size - ring_index - 1
                num_targets_backward = ring_index

                # Determine if this device has secondary axis connections
                has_secondary_target = is_sender and (mesh_cols > 1) and (bcast_secondary_cluster_axis is not None)

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
                # CBs overlapped with sdpa_kv_cache L1 buffer (consumed before SDPA runs)
                sdpa_kv_cache_running_offset = 0
                # CBs overlapped with sdpa_kv_cache L1 buffer permanently allocated on the mcast core
                # Nothing should reuse this space on the mcast core
                sdpa_kv_cache_running_offset_mcast_core = 0

                # Create circular buffer descriptors
                # CB: Input (created from sharded tensor)
                broadcast_address = 0
                if skip_ccl:
                    in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor_device)
                else:
                    in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        input_cb,
                        sdpa_kv_cache_buffer_device,
                        address_offset=sdpa_kv_cache_running_offset_mcast_core,
                        total_size=num_tiles * cb_page_size,
                    )
                    broadcast_address = ttnn.get_cb_address(in_cb_descriptor)
                    in_cb_descriptor.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=input_cb,
                            data_format=data_format,
                            page_size=cb_page_size,
                            tile=tile_descriptor,
                        )
                    ]
                    sdpa_kv_cache_running_offset_mcast_core += in_cb_descriptor.total_size

                # CB: Gamma (backed by fused overlapped tensor)
                gamma_cb_descriptor = cb_descriptor_from_overlapped_tensor(
                    gamma_cb, gamma_tensor, gamma_fused_tensor_device
                )
                gamma_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                gamma_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # CB: RMSNorm2 Gamma (backed by fused overlapped tensor)
                rmsnorm2_gamma_cb_descriptor = cb_descriptor_from_overlapped_tensor(
                    rmsnorm2_gamma_cb, rmsnorm2_gamma_tensor, gamma_fused_tensor_device
                )
                rmsnorm2_gamma_cb_descriptor.format_descriptors[0].tile = rmsnorm2_tile_descriptor
                rmsnorm2_gamma_cb_descriptor.format_descriptors[0].page_size = rmsnorm2_page_size

                # CB: CCL broadcast packet buffer
                bcast_pkt_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(bcast_pkt_cb, input_tensor_device)

                # CB: RMSNorm output buffer
                rmsnorm_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    rmsnorm_output_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=sdpa_kv_cache_running_offset_mcast_core,
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
                sdpa_kv_cache_running_offset_mcast_core += rmsnorm_output_cb_descriptor.total_size

                # CB: Fused matmul weights (single CB backing matmul1, matmul2, dkv_matmul)
                fused_matmul_weights_cb_descriptor = cb_descriptor_from_overlapped_tensors(
                    matmul_weights_cb_overlapped,
                    [matmul_weights_tensor, matmul2_weights_tensor, dkv_matmul_weights_tensor],
                    fused_weights_tensor_device,
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
                    sdpa_kv_cache_buffer_device,
                    address_offset=sdpa_kv_cache_running_offset_mcast_core,  # 14400 B
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
                sdpa_kv_cache_running_offset_mcast_core += rmsnorm2_input_cb_descriptor.total_size  # +3072 B

                # CB9 lifecycle:
                # 1) RMSNorm2 writes normalized output here
                # 2) Mcast2 reads from CB9 and writes to matmul2 input CB

                # CB 8: gather_reduce half1 scratch buffer (3 tiles) — overlap with sdpa_out_interm L1 buffer
                # at offset 3136 B. This CB is consumed before SDPA runs.
                gather_reduce_half1_scratch_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    gather_reduce_half1_scratch_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=sdpa_kv_cache_running_offset_mcast_core,  # 17472 B
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
                sdpa_kv_cache_running_offset_mcast_core += (
                    gather_reduce_half1_scratch_cb_descriptor.total_size
                )  # +3072 B

                # CB: RMSNorm2 output buffer (3 tiles)
                rmsnorm2_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    rmsnorm2_output_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=sdpa_kv_cache_running_offset_mcast_core,  # 20544 B
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
                sdpa_kv_cache_running_offset_mcast_core += rmsnorm2_output_cb_descriptor.total_size  # +3072 B

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

                # CB 13: Matmul3 weights (backed by fused kv_b12 overlapped tensor)
                matmul3_weights_cb_descriptor = cb_descriptor_from_overlapped_tensor(
                    matmul3_weights_cb, matmul3_weights_tensor, kv_b12_fused_tensor_device
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
                TILE_8x32 = ttnn.Tile((Q_TILE_HEIGHT, 32))
                create_q_heads_interm_tile_descriptor = ttnn.TileDescriptor(TILE_8x32)
                create_q_heads_interm_page_size = TILE_8x32.get_tile_size(untilize_df)  # 8*32*2 = 512 bytes
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
                        data_format=untilize_df,
                        page_size=create_q_heads_interm_page_size,
                        tile=create_q_heads_interm_tile_descriptor,
                    )
                ]
                sdpa_out_interm_running_offset += create_q_heads_interm_cb_descriptor.total_size  # +9216 B

                # CB 16: CreateQHeads output buffer (tilized data, backed by output tensor)
                # Only allocated on receiver cores (SDPA Input grid) - senders no longer write here
                # This CB is input to SDPA, put it on all cores but only 8 cores have data
                create_q_heads_out_tile_descriptor = ttnn.TileDescriptor(TILE_8x32)
                create_q_heads_out_page_size = create_q_heads_interm_page_size
                create_q_heads_out_total_size = create_q_heads_out_page_size * q_tiles
                create_q_heads_out_cb_descriptor = ttnn.CBDescriptor(
                    total_size=create_q_heads_out_total_size,
                    core_ranges=mla_core_grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            create_q_heads_out_cb,
                            data_format,
                            create_q_heads_out_page_size,
                            create_q_heads_out_tile_descriptor,
                        )
                    ],
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

                # CB: KV RMSNorm gamma buffer (backed by fused overlapped tensor)
                kv_rmsnorm_gamma_cb_descriptor = cb_descriptor_from_overlapped_tensor(
                    kv_rmsnorm_gamma_cb, dkv_rmsnorm_gamma_tensor, gamma_fused_tensor_device
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

                TILE_32x32 = ttnn.Tile((32, 32))
                kv_cache_page_size = TILE_32x32.get_tile_size(k_df)
                kv_cache_num_tiles = 16
                kv_cache_input_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=kv_cache_input_cb,
                    data_format=k_df,
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
                    data_format=k_df,
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
                    data_format=untilize_df,
                    page_size=TILE_32x32.get_tile_size(untilize_df),
                    tile=ttnn.TileDescriptor(TILE_32x32),
                )
                # One extra tile for syncing, can optimize to remove
                kv_cache_intermed_cb_descriptor = ttnn.CBDescriptor(
                    total_size=(kv_cache_num_tiles + 1) * TILE_32x32.get_tile_size(untilize_df),
                    core_ranges=kv_cache_update_grid,
                    format_descriptors=[kv_cache_intermed_cb_format],
                )

                # Flash MLA cb descriptors
                mla_cb_descriptors = []

                q_tile_descriptor = ttnn.TileDescriptor(q_tiny_tile)
                stats_tile_descriptor = ttnn.TileDescriptor(q_tiny_tile)
                stats_tile = q_tiny_tile
                stats_tile_size = stats_tile.get_tile_size(stats_df)

                mla_intermed_output_tiles = out0_t * optimized_mla_grid.NUM_TREE_REDUCTION_STEPS
                mla_intermed_ms_tiles = PNHt * optimized_mla_grid.NUM_TREE_REDUCTION_STEPS

                # cb_k_in: K input (full tile)
                mla_k_in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    mla_k_in_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=0,
                    total_size=k_tiles * k_tile_size,
                )
                mla_k_in_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=mla_k_in_cb,
                        data_format=k_df,
                        page_size=k_tile_size,
                        tile=ttnn.TileDescriptor(kv_cache_tensor.get_tile()),
                    )
                ]
                mla_cb_descriptors.append(mla_k_in_cb_descriptor)
                # V is read directly from K buffer (strided matmul) - no separate V CB needed

                # cb_mask: Mask input
                mla_cb_descriptors.append(
                    ttnn.CBDescriptor(
                        total_size=q_tile_size,
                        core_ranges=mla_core_grid,
                        format_descriptors=[ttnn.CBFormatDescriptor(mla_mask_cb, q_df, q_tile_size)],
                    )
                )

                if optimized_mla_grid.NUM_TREE_REDUCTION_STEPS > 0:
                    # cb_out_in: output input (tiny tile)
                    mla_out_in_total_size = mla_intermed_output_tiles * stats_tile_size
                    mla_out_in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        mla_out_in_cb,
                        sdpa_out_interm_buffer_device,
                        address_offset=0,
                        total_size=mla_out_in_total_size,
                    )
                    mla_out_in_cb_descriptor.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=mla_out_in_cb,
                            data_format=stats_df,
                            page_size=stats_tile_size,
                            tile=stats_tile_descriptor,
                        )
                    ]
                    mla_cb_descriptors.append(mla_out_in_cb_descriptor)
                    # cb_ms_in: m/s stats input (m and s are packed into single tile)
                    mla_ms_in_total_size = mla_intermed_ms_tiles * stats_tile_size
                    mla_ms_in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        mla_ms_in_cb,
                        sdpa_out_interm_buffer_device,
                        address_offset=mla_out_in_total_size,
                        total_size=mla_ms_in_total_size,
                    )
                    mla_ms_in_cb_descriptor.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=mla_ms_in_cb,
                            data_format=stats_df,
                            page_size=stats_tile_size,
                            tile=stats_tile_descriptor,
                        )
                    ]
                    mla_cb_descriptors.append(mla_ms_in_cb_descriptor)

                # Position tensor is now height-sharded - no CB needed, read directly from L1

                # mla_out_o_cb/mla_interm_out_cb: output O (tiny tile)
                mla_cb_descriptors.append(
                    ttnn.CBDescriptor(
                        total_size=out0_t * stats_tile_size,
                        core_ranges=mla_core_grid,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(mla_out_o_cb, stats_df, stats_tile_size, stats_tile_descriptor),
                            ttnn.CBFormatDescriptor(
                                mla_interm_out_cb, stats_df, stats_tile_size, stats_tile_descriptor
                            ),
                        ],
                    )
                )

                # cb_out_ms/cb_interm_ms: output m/s stats (tiny tile, shared for both m and s)
                mla_cb_descriptors.append(
                    ttnn.CBDescriptor(
                        total_size=statistics_tiles * stats_tile_size,
                        core_ranges=mla_core_grid,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(mla_out_ms_cb, stats_df, stats_tile_size, stats_tile_descriptor),
                            ttnn.CBFormatDescriptor(mla_interm_ms_cb, stats_df, stats_tile_size, stats_tile_descriptor),
                        ],
                    )
                )

                # Post SDPA
                # ========================================================================
                # Circular buffer descriptors
                # ========================================================================
                running_address_offset = 0

                # CB 0: Matmul4 input (from sharded tensor, kv_b2 grid)
                matmul4_in0_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                matmul4_in0_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul4_in0_cb,
                    data_format=data_format,
                    page_size=tile_1x32_size,
                    tile=matmul4_in0_tile_descriptor,
                )
                matmul4_in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul4_in0_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=running_address_offset,
                    total_size=matmul4_k_num_tiles * tile_1x32_size,
                )
                matmul4_in0_cb_descriptor.format_descriptors = [matmul4_in0_cb_format]
                running_address_offset += matmul4_in0_cb_descriptor.total_size

                # CB 1: Matmul4 weights (from kv_b2 overlapped tensor)
                matmul4_in1_cb_descriptor = cb_descriptor_from_overlapped_tensor(
                    matmul4_in1_cb, post_sdpa_weights1_tensor, post_sdpa_weights1_fused_tensor_device
                )

                # CB 2: Matmul4 output (4 tiles of 1x32 per core, kv_b2 grid)
                # When kv_cache buffer is available, overlap into it. Otherwise standalone.
                matmul4_out_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                matmul4_out_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul4_out_cb,
                    data_format=data_format,
                    page_size=tile_1x32_size,
                    tile=matmul4_out_tile_descriptor,
                )

                matmul4_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul4_out_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=running_address_offset,
                    total_size=matmul4_out_w_per_core * tile_1x32_size,
                )
                matmul4_out_cb_descriptor.format_descriptors = [matmul4_out_cb_format]
                running_address_offset += matmul4_out_cb_descriptor.total_size

                # CB 3: Gather2 output = Mcast3 source (from sharded tensor, gather core)
                gather2_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    gather2_dst_cb, post_sdpa_gather2_output_tensor
                )

                # CB 4: Mcast3 destination = Matmul5 input (256 tiles of 1x32 per core)
                matmul5_in0_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul5_in0_cb,
                    data_format=data_format,
                    page_size=tile_1x32_size,
                    tile=matmul4_out_tile_descriptor,
                )
                matmul5_in0_cb_grid = mcast3_core_grid.merge(gather_core_grid)
                matmul5_in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul5_in0_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=running_address_offset,
                    total_size=mcast3_dst_num_pages * tile_1x32_size,
                )
                matmul5_in0_cb_descriptor.format_descriptors = [matmul5_in0_cb_format]
                running_address_offset += matmul5_in0_cb_descriptor.total_size

                # CB 5: Matmul5 weights (from o_proj overlapped tensor)
                matmul5_in1_cb_descriptor = cb_descriptor_from_overlapped_tensor(
                    matmul5_in1_cb, post_sdpa_weights2_tensor, post_sdpa_weights2_fused_tensor_device
                )

                # CB 6: Matmul5 output (2 tiles of 1x32 per core, 112 active matmul5 cores)
                matmul5_out_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul5_out_cb,
                    data_format=data_format,
                    page_size=tile_1x32_size,
                    tile=matmul4_out_tile_descriptor,
                )
                matmul5_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul5_out_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=running_address_offset,
                    total_size=matmul5_out_w_per_core * tile_1x32_size,
                )
                matmul5_out_cb_descriptor.format_descriptors = [matmul5_out_cb_format]
                running_address_offset += matmul5_out_cb_descriptor.total_size

                # CB 7: Gather3 output = CCL local data (backed by tensor on gather core)
                # CCL sender reads from this tensor via NOC, not from local CB
                gather3_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    gather3_dst_cb, post_sdpa_gather3_output_tensor_device
                )
                gather3_dst_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                gather3_dst_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                post_sdpa_cb_list = [
                    matmul4_in0_cb_descriptor,
                    matmul4_in1_cb_descriptor,
                    matmul4_out_cb_descriptor,
                    gather2_dst_cb_descriptor,
                    matmul5_in0_cb_descriptor,
                    matmul5_in1_cb_descriptor,
                    matmul5_out_cb_descriptor,
                    gather3_dst_cb_descriptor,
                ]

                # CCL CBs (8-13): only when CCL is enabled
                # CB 8: CCL sender input (reads from gather3 output via NOC)
                ccl_sender_in_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=ccl_sender_in_cb,
                    data_format=data_format,
                    page_size=tile_size,
                    tile=tile_descriptor,
                )
                ccl_sender_in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    ccl_sender_in_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=running_address_offset,
                    total_size=ccl_num_pages * tile_size,
                )
                ccl_sender_in_cb_descriptor.format_descriptors = [ccl_sender_in_cb_format]
                running_address_offset += ccl_sender_in_cb_descriptor.total_size
                post_sdpa_cb_list.append(ccl_sender_in_cb_descriptor)

                # CB 9: CCL remote data (backed by intermediate tensor with 32x32 tiles)
                # The intermediate tensor is where the CCL sender writes remote data
                ccl_remote_data_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    ccl_remote_data_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=sdpa_kv_cache_running_offset_mcast_core,
                    total_size=ccl_num_pages * tile_size,
                )
                ccl_remote_data_cb_descriptor.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=ccl_remote_data_cb,
                        data_format=data_format,
                        page_size=tile_size,
                        tile=tile_descriptor,
                    )
                ]
                sdpa_kv_cache_running_offset_mcast_core += ccl_remote_data_cb_descriptor.total_size
                post_sdpa_cb_list.append(ccl_remote_data_cb_descriptor)
                ccl_send_addr = ttnn.get_cb_address(ccl_remote_data_cb_descriptor)

                # CB 11: CCL temp scratch buffer (not backed by tensor)
                ccl_temp_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=ccl_temp_cb,
                    data_format=data_format,
                    page_size=cb_page_size,
                    tile=tile_descriptor,
                )
                ccl_temp_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    ccl_temp_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=sdpa_kv_cache_running_offset_mcast_core,
                    total_size=ccl_num_tiles * tile_size,
                )
                ccl_temp_cb_descriptor.format_descriptors = [ccl_temp_cb_format]
                sdpa_kv_cache_running_offset_mcast_core += ccl_temp_cb_descriptor.total_size
                post_sdpa_cb_list.append(ccl_temp_cb_descriptor)

                # CB 12: CCL output (from sharded tensor)
                attention_block_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    attention_block_output_cb, attention_block_output_tensor_device
                )
                attention_block_output_cb_descriptor.core_ranges = gather_core_grid
                attention_block_output_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                attention_block_output_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # CB 13: CCL packet headers
                ccl_packet_header_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=ccl_packet_header_cb,
                    data_format=ttnn.uint32,
                    page_size=ccl_packet_header_size_bytes,
                )
                ccl_packet_header_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    ccl_packet_header_cb,
                    sdpa_kv_cache_buffer_device,
                    address_offset=running_address_offset,
                    total_size=2 * ccl_packet_header_size_bytes,
                )
                ccl_packet_header_cb_descriptor.format_descriptors = [ccl_packet_header_cb_format]
                running_address_offset += ccl_packet_header_cb_descriptor.total_size
                post_sdpa_cb_list.append(ccl_packet_header_cb_descriptor)
                # Get per-device SDPA tensors
                sdpa_input_l_device = ttnn.get_device_tensors(sdpa_input_l_mesh)[device_idx]
                sdpa_input_ms_device = ttnn.get_device_tensors(sdpa_input_ms_mesh)[device_idx]
                sdpa_output_l_device = ttnn.get_device_tensors(sdpa_output_l_mesh)[device_idx]
                sdpa_intermediate_recv_device = ttnn.get_device_tensors(sdpa_intermediate_recv_mesh)[device_idx]

                # Get from fused
                # CB 14: SDPA local L (aliased to input tensor)
                # sdpa_local_l_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                #    sdpa_cb_local_l, sdpa_input_l_device
                # )
                # post_sdpa_cb_list.append(sdpa_local_l_cb_descriptor)

                # CB 15: SDPA local MS (aliased to input tensor)
                # sdpa_local_ms_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                #    sdpa_cb_local_ms, sdpa_input_ms_device
                # )
                # post_sdpa_cb_list.append(sdpa_local_ms_cb_descriptor)

                # CB 16: SDPA neighbor L (aliased to intermediate recv buffer)
                # The recv buffer holds both L and MS data, but this CB should only
                # cover the L portion. Override total_size like the original op does.
                sdpa_neighbor_l_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    sdpa_cb_neighbor_l, sdpa_intermediate_recv_device
                )
                sdpa_neighbor_l_cb_descriptor.total_size = 2 * sdpa_l_tiles_per_worker * sdpa_l_tile_size
                post_sdpa_cb_list.append(sdpa_neighbor_l_cb_descriptor)

                # CB 17: SDPA R1 neighbor MS (scratch, not backed by tensor)
                # Must use sdpa_tile (e.g., 8x32) to match MS input tensor tile format
                sdpa_neighbor_ms_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    sdpa_cb_neighbor_ms, sdpa_intermediate_recv_device
                )
                sdpa_neighbor_ms_cb_descriptor.total_size = 2 * sdpa_ms_tile_size
                post_sdpa_cb_list.append(sdpa_neighbor_ms_cb_descriptor)

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
                post_sdpa_cb_list.append(sdpa_r1_result_l_cb_descriptor)

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
                post_sdpa_cb_list.append(sdpa_r1_result_ms_cb_descriptor)

                # CB 20: SDPA L output (aliased to output tensor)
                sdpa_l_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(sdpa_cb_l_out, sdpa_output_l_device)
                post_sdpa_cb_list.append(sdpa_l_out_cb_descriptor)

                # CB 21: SDPA packet slot (for fabric packet headers)
                sdpa_packet_header_cb_size = 2 * ttnn.get_tt_fabric_packet_header_size_bytes()
                sdpa_packet_slot_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=sdpa_cb_packet_slot,
                    data_format=ttnn.uint32,
                    page_size=sdpa_packet_header_cb_size,
                )
                sdpa_packet_slot_cb_descriptor = ttnn.CBDescriptor(
                    total_size=sdpa_packet_header_cb_size,
                    core_ranges=sdpa_worker_grid,
                    format_descriptors=[sdpa_packet_slot_cb_format],
                )
                post_sdpa_cb_list.append(sdpa_packet_slot_cb_descriptor)

                # ========================================================================
                # Semaphore descriptors
                # ========================================================================
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
                mcast3_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=mcast3_data_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                gather3_noc0_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather3_noc0_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                gather3_noc1_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather3_noc1_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                semaphore_list = [
                    gather2_noc0_semaphore_descriptor,
                    gather2_noc1_semaphore_descriptor,
                    mcast3_receiver_semaphore_descriptor,
                    gather3_noc0_semaphore_descriptor,
                    gather3_noc1_semaphore_descriptor,
                ]
                gather3_completion_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather3_completion_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                semaphore_list.append(gather3_completion_semaphore_descriptor)

                # SDPA scatter arrival semaphore (for matmul4 cores to wait for scatter data)
                scatter_arrival_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=scatter_arrival_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                semaphore_list.append(scatter_arrival_semaphore_descriptor)

                # SDPA forwarder semaphores (workers signal these to forwarders)
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

                kv_cache_tensor_accessor_args = ttnn.TensorAccessorArgs(kv_cache_tensor_device)
                brisc_compile_time_args = kv_cache_tensor_accessor_args.get_compile_time_args()
                ncrisc_compile_time_args = kv_cache_tensor_accessor_args.get_compile_time_args()
                # =======================================================================
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

                k_addr = kv_cache_tensor_device.buffer_address()

                # Setup MLA per core runtime args
                mla_ncrisc_per_core_args = []
                mla_brisc_per_core_args = []
                mla_trisc_per_core_args = []

                output_core_physical_xs = []
                output_core_physical_ys = []

                for i in range(num_active_mla_cores):
                    if i % num_cores_per_batch == 0:  # First core in batch group is output (S1)
                        core_physical = device.worker_core_from_logical_core(mla_core_group[i])
                        output_core_physical_xs.append(core_physical.x)
                        output_core_physical_ys.append(core_physical.y)

                s_block_mcast_coords = []
                for s_idx in range(num_s_blocks):
                    coords = optimized_mla_grid.physical_multicast_coords(device, s_idx)
                    s_block_mcast_coords.append(coords)

                for i in range(num_active_mla_cores):
                    core = mla_core_group[i]

                    s_block_idx = i % num_s_blocks
                    cur_batch = i // num_cores_per_batch
                    core_num_in_reduce = i % num_cores_per_head
                    core_num_in_output = i % num_cores_per_batch

                    do_reduce = 1 if optimized_mla_grid.is_tree_reduction_receiver(s_block_idx) else 0
                    is_output_core = 1 if s_block_idx == 0 else 0
                    is_mcast_sender = 1 if i < num_s_blocks else 0

                    mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, _ = s_block_mcast_coords[s_block_idx]

                    if s_block_idx < 4:
                        vc = s_block_idx & 0x1
                    else:
                        vc = 2 + ((s_block_idx - 4) & 0x1)

                    output_core_noc_x = (
                        output_core_physical_xs[cur_batch] if cur_batch < len(output_core_physical_xs) else 0
                    )
                    output_core_noc_y = (
                        output_core_physical_ys[cur_batch] if cur_batch < len(output_core_physical_ys) else 0
                    )

                    # NCRISC per-core runtime args (common args: k_addr, pos_addr)
                    mla_ncrisc_per_core_args.append(
                        (
                            core,
                            [
                                cur_batch,
                                core_num_in_reduce,
                                is_mcast_sender,
                                mcast_start_x,
                                mcast_start_y,
                                vc,
                            ],
                        )
                    )

                    # Tree reduction partner coordinates
                    tree_reduction_info = optimized_mla_grid.get_tree_reduction_partner_coords(
                        device, s_block_idx, cur_batch
                    )

                    # BRISC per-core runtime args (common args: pos_addr)
                    mla_brisc_args = [
                        cur_batch,
                        core_num_in_reduce,
                        is_output_core,
                        is_mcast_sender,
                        output_core_noc_x,
                        output_core_noc_y,
                        mcast_start_x,
                        mcast_start_y,
                        mcast_end_x,
                        mcast_end_y,
                    ]
                    for role_code, partner_s_block_idx, partner_x, partner_y in tree_reduction_info:
                        mla_brisc_args.extend([role_code, partner_s_block_idx, partner_x, partner_y])
                    mla_brisc_per_core_args.append((core, mla_brisc_args))

                    is_sender_after_reduce = (
                        1 if (do_reduce and optimized_mla_grid.is_tree_reduction_sender(s_block_idx)) else 0
                    )

                    # TRISC per-core runtime args (common args: pos_addr)
                    mla_trisc_args = [
                        do_reduce,
                        0,  # do_output, set to 0 in fused
                        cur_batch,
                        core_num_in_reduce,
                        1,  # is_sender_after_reduce, set to 1 in fused
                    ]
                    for role_code, partner_s_block_idx, partner_x, partner_y in tree_reduction_info:
                        mla_trisc_args.extend([role_code, partner_s_block_idx])
                    mla_trisc_per_core_args.append((core, mla_trisc_args))

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
                    fabric_node_id = mesh_device.get_fabric_node_id(mesh_coord)
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
                        int(broadcast_address),  # tensor_address0
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

                # RoPE DRAM address args (per-device)
                qrope_cos_tensor_address = qrope_cos_tensor_device.buffer_address()
                qrope_sin_tensor_address = qrope_sin_tensor_device.buffer_address()
                krope_cos_tensor_address = krope_cos_tensor_device.buffer_address()
                krope_sin_tensor_address = krope_sin_tensor_device.buffer_address()
                position_ids_tensor_addr = position_ids_tensor_device.buffer_address()

                # Compute address overrides for each matmul's weights within the fused buffer
                fused_weights_base_addr = fused_weights_tensor_device.buffer_address()
                matmul_weights_addr = fused_weights_base_addr + matmul_weights_tensor.byte_offset
                matmul2_weights_addr = fused_weights_base_addr + matmul2_weights_tensor.byte_offset
                dkv_matmul_weights_addr = fused_weights_base_addr + dkv_matmul_weights_tensor.byte_offset

                # TRISC common runtime args (shared scalar values)
                trisc_common_runtime_args = [
                    epsilon_packed,  # idx 0
                    scalar_packed,  # idx 1
                    scalar2_packed,  # idx 2
                    kv_scalar_packed,  # idx 3
                    kv_cache_input_cb,  # idx 4
                    kv_cache_output_cb,  # idx 5
                    kv_cache_intermed_cb,  # idx 6
                    position_ids_tensor_addr,  # idx 7
                    matmul_weights_addr,  # idx 8
                    matmul2_weights_addr,  # idx 9
                    dkv_matmul_weights_addr,  # idx 10
                ]

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

                kv_cache_sp_named_compile_time_args = [
                    ("kv_cache_device_chunk_size", device_chunk_size),
                    ("kv_cache_sp_device_idx", row),
                    ("kv_cache_num_sp_devices", mesh_rows),
                ]

                ncrisc_named_compile_time_args = (
                    bcast_ncrisc_named_compile_time_args
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
                    + krope_ncrisc_addr_args
                    + kv_cache_sp_named_compile_time_args
                    + mla_ncrisc_named_compile_time_args
                    + post_sdpa_ncrisc_named_compile_time_args
                )
                ncrisc_common_runtime_args = ncrisc_bcast_common_args + [
                    k_addr,
                    position_ids_tensor_addr,
                ]

                brisc_named_compile_time_args = (
                    bcast_brisc_named_compile_time_args
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
                    + kv_cache_brisc_named_compile_time_args
                    + kv_cache_sp_named_compile_time_args
                    + mla_brisc_named_compile_time_args
                    + post_sdpa_brisc_named_compile_time_args
                )
                brisc_common_runtime_args = [k_addr, position_ids_tensor_addr]

                trisc_named_compile_time_args = (
                    bcast_trisc_named_compile_time_args
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
                    + kv_cache_trisc_named_compile_time_args
                    + kv_cache_sp_named_compile_time_args
                    + mla_trisc_named_compile_time_args
                    + post_sdpa_trisc_named_compile_time_args
                )

                unified_compile_time_core_descriptors = [
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_input_core",
                        core_range=rmsnorm_core,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_matmul_core",
                        core_range=matmul_weights_core_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_matmul2_core",
                        core_range=matmul2_weights_core_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_qnope_core",
                        core_range=qnope_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_qrope_core",
                        core_range=qrope_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_sdpa_input_core",
                        core_range=sdpa_input_grid,
                        value=1,
                        other_value=0,
                    ),
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
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_mla_core",
                        core_range=mla_core_grid,
                        value=1,
                        other_value=0,
                    ),
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
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_matmul4_core",
                        core_range=matmul4_core_grid,
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
                        named_compile_time_arg="is_matmul5_core",
                        core_range=matmul5_active_core_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_mcast3_receiver_core",
                        core_range=mcast3_core_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_ccl_sender_core",
                        core_range=ccl_sender_core_grid,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_ccl_receiver_core",
                        core_range=gather_core_grid,  # CCL receiver = gather core
                        value=1,
                        other_value=0,
                    ),
                ]

                per_core_compile_time_descriptors = [
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
                    PerCoreCompileTimeDescriptor(
                        named_compile_time_arg="gather2_sender_idx",
                        core_values=gather2_sender_idx_per_core,
                        other_value=0,
                    ),
                    PerCoreCompileTimeDescriptor(
                        named_compile_time_arg="gather3_sender_idx",
                        core_values=gather3_sender_idx_per_core,
                        other_value=0,
                    ),
                ]

                per_core_ncrisc_args = mla_ncrisc_per_core_args
                per_core_brisc_args = mla_brisc_per_core_args
                per_core_trisc_args = mla_trisc_per_core_args

                cbs_list = [
                    gamma_cb_descriptor,
                    rmsnorm_output_cb_descriptor,
                    fused_matmul_weights_cb_descriptor,
                    matmul_output_cb_descriptor,
                    matmul_input_cb_descriptor,
                    rmsnorm2_gamma_cb_descriptor,
                    rmsnorm2_input_cb_descriptor,
                    gather_reduce_half1_scratch_cb_descriptor,
                    rmsnorm2_output_cb_descriptor,
                    matmul2_input_cb_descriptor,
                    matmul2_output_cb_descriptor,
                    matmul3_weights_cb_descriptor,
                    matmul3_output_cb_descriptor,
                    qrope_output_cb_descriptor,
                    create_q_heads_out_cb_descriptor,
                    qrope_cos_cb_descriptor,
                    qrope_sin_cb_descriptor,
                    qrope_trans_mat_cb_descriptor,
                    qrope_rotated_input_interm_cb_descriptor,
                    qrope_cos_interm_cb_descriptor,
                    qrope_sin_interm_cb_descriptor,
                    dkv_matmul_output_cb_descriptor,
                    kv_rmsnorm_input_cb_descriptor,
                    kv_rmsnorm_gamma_cb_descriptor,
                    kv_rmsnorm_output_cb_descriptor,
                    krope_output_cb_descriptor,
                    krope_cos_cb_descriptor,
                    krope_sin_cb_descriptor,
                    create_q_heads_interm_cb_descriptor,
                    kv_cache_output_cb_descriptor,
                    kv_cache_intermed_cb_descriptor,
                    kv_cache_input_cb_descriptor,
                    *mla_cb_descriptors,
                    *post_sdpa_cb_list,
                ]
                if not skip_ccl:
                    cbs_list.append(bcast_pkt_cb_descriptor)

                # ========================================================================
                # Pre-compute CCL runtime args (fabric setup deferred to op())
                # ========================================================================
                ccl_sender_ncrisc_common_rt_args = [
                    gather3_receiver_data_addr,
                ]
                ccl_sender_brisc_common_rt_args = [
                    ccl_send_addr,
                    ccl_sender_semaphore_addr,
                ]
                ccl_receiver_ncrisc_common_rt_args = [
                    ccl_receiver_semaphore_addr,
                ]
                fabric_node_id = mesh_device.get_fabric_node_id(mesh_coord)
                neighbor_coord_obj = ttnn.MeshCoordinate(neighbor_row, neighbor_col)
                neighbor_fabric_node_id = mesh_device.get_fabric_node_id(neighbor_coord_obj)
                ccl_ctx = {
                    "sender_ncrisc_common_rt_args": ccl_sender_ncrisc_common_rt_args,
                    "sender_brisc_common_rt_args": ccl_sender_brisc_common_rt_args,
                    "receiver_ncrisc_common_rt_args": ccl_receiver_ncrisc_common_rt_args,
                    "sender_link": ccl_sender_link,
                    "receiver_link": ccl_receiver_link,
                    "fabric_node_id": fabric_node_id,
                    "neighbor_fabric_node_id": neighbor_fabric_node_id,
                }

                # ========================================================================
                # Pre-compute SDPA runtime args (fabric setup deferred to op())
                # ========================================================================
                # Get per-device SDPA tensors
                sdpa_intermediate_recv_device = ttnn.get_device_tensors(sdpa_intermediate_recv_mesh)[device_idx]
                sdpa_forwarder_scratch_device = ttnn.get_device_tensors(sdpa_forwarder_scratch_mesh)[device_idx]

                # Get device for logical to NOC coordinate conversion
                device = sdpa_intermediate_recv_device.device()

                # Get fabric node IDs for SDPA CCL
                sdpa_fabric_node_id = mesh_device.get_fabric_node_id(mesh_coord)

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

                # Ring-local device indices for position validity (ring along sdpa_cluster_axis)
                sdpa_ring_idx = row if sdpa_cluster_axis == 0 else col
                sdpa_fwd_ring_idx = fwd_row if sdpa_cluster_axis == 0 else fwd_col
                sdpa_bwd_ring_idx = bwd_row if sdpa_cluster_axis == 0 else bwd_col
                sdpa_num_ring_devices = mesh_shape[0] if sdpa_cluster_axis == 0 else mesh_shape[1]

                # Position tensor address (0 if position disabled)
                sdpa_pos_addr = position_ids_tensor_addr

                # SDPA worker runtime args (per-core)
                sdpa_worker_ncrisc_rt_args = ttnn.RuntimeArgs()
                sdpa_worker_brisc_rt_args = ttnn.RuntimeArgs()
                sdpa_worker_trisc_rt_args = ttnn.RuntimeArgs()

                # Get matmul4 input buffer address for scatter destination
                scatter_dest_l1_addr = (
                    sdpa_kv_cache_buffer_device.buffer_address() + matmul4_in0_cb_descriptor.address_offset
                )

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

                sdpa_worker_cores = ttnn.corerange_to_cores(sdpa_worker_grid)
                for worker_idx, worker_core in enumerate(sdpa_worker_cores):
                    # Convert worker core to NOC coordinates (like original sdpa_reduce_to_all op)
                    worker_core_noc = device.worker_core_from_logical_core(worker_core)

                    # NCRISC runtime args: semaphore addresses, recv buffer addresses
                    r1_neighbor_sem_addr = sdpa_semaphore1_addr
                    r2_neighbor_sem_addr = sdpa_semaphore2_addr
                    r1_recv_buffer_addr = sdpa_intermediate_recv_device.buffer_address()
                    r2_recv_buffer_addr = r1_recv_buffer_addr + (sdpa_l_tiles_per_worker + 1) * sdpa_l_tile_size

                    ncrisc_base_args = [
                        r1_neighbor_sem_addr,
                        r2_neighbor_sem_addr,
                        r1_recv_buffer_addr,
                        r2_recv_buffer_addr,
                    ]
                    sdpa_worker_ncrisc_rt_args[worker_core.x][worker_core.y] = ncrisc_base_args

                    # BRISC runtime args: fabric destinations, scatter destinations
                    # Determine which matmul4 cores this worker scatters to (8 rows per worker)
                    scatter_dest_noc_coords = []
                    for scatter_row in range(sdpa_scatter_num_rows):
                        matmul4_core_idx = worker_idx * sdpa_scatter_num_rows + scatter_row
                        matmul4_core_logical = matmul4_cores[matmul4_core_idx]
                        # Convert to NOC coordinates
                        matmul4_core_noc = device.worker_core_from_logical_core(matmul4_core_logical)
                        scatter_dest_noc_coords.append((matmul4_core_noc.x, matmul4_core_noc.y))

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
                        r1_recv_buffer_addr,  # r1_neighbor_dst_addr
                        sdpa_semaphore1_addr,  # r1_neighbor_sem_addr
                        int(r2_dst_fabric_node_id.mesh_id),  # r2_dst_mesh_id (varies by type!)
                        r2_dst_fabric_node_id.chip_id,  # r2_dst_chip_id
                        r2_recv_buffer_addr,  # r2_neighbor_dst_addr
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

                    # Position runtime args (TRISC and NCRISC extension)
                    # Compute ring-local neighbor indices based on Type A/B
                    # Type A: R1=forward, R2=backward
                    # Type B: R1=backward, R2=forward
                    if is_type_a:
                        pos_r1_neighbor_idx = sdpa_fwd_ring_idx
                        pos_r2_neighbor_idx = sdpa_bwd_ring_idx
                    else:
                        pos_r1_neighbor_idx = sdpa_bwd_ring_idx
                        pos_r2_neighbor_idx = sdpa_fwd_ring_idx

                    # R2 neighbor's R1 neighbor: determine the R2 neighbor's type, then its R1 direction
                    r2_neighbor_row_in_ring = pos_r2_neighbor_idx
                    r2_neighbor_is_type_a = ((r2_neighbor_row_in_ring + worker_idx) % 2) == 0
                    if r2_neighbor_is_type_a:
                        pos_r2_neighbor_r1_idx = (pos_r2_neighbor_idx + 1) % sdpa_num_ring_devices
                    else:
                        pos_r2_neighbor_r1_idx = (
                            pos_r2_neighbor_idx - 1 + sdpa_num_ring_devices
                        ) % sdpa_num_ring_devices

                    # TRISC args: pos_addr, device_idx, r1_neighbor, r2_neighbor, r2_neighbor_r1_neighbor
                    sdpa_worker_trisc_rt_args[worker_core.x][worker_core.y] = [
                        sdpa_pos_addr,
                        sdpa_ring_idx,
                        pos_r1_neighbor_idx,
                        pos_r2_neighbor_idx,
                        pos_r2_neighbor_r1_idx,
                    ]

                    # Extend NCRISC args: pos_addr, r1_neighbor, r2_neighbor, r2_neighbor_r1_neighbor
                    sdpa_worker_ncrisc_rt_args[worker_core.x][worker_core.y].extend(
                        [
                            sdpa_pos_addr,
                            pos_r1_neighbor_idx,
                            pos_r2_neighbor_idx,
                            pos_r2_neighbor_r1_idx,
                        ]
                    )

                # Pre-compute forwarder base args (fabric args added in op())
                sdpa_forwarder_brisc_base_args = {}
                sdpa_forwarder_ncrisc_base_args = {}
                for fwd_idx, fwd_core in enumerate(sdpa_forwarder_cores):
                    sdpa_forwarder_brisc_base_args[(fwd_core.x, fwd_core.y)] = [
                        forwarder_buffer_base,
                        0,
                        sdpa_fwd_r1_sem_id,
                        sdpa_fwd_r2_sem_id,
                    ]
                    sdpa_forwarder_ncrisc_base_args[(fwd_core.x, fwd_core.y)] = [
                        forwarder_buffer_base,
                        ncrisc_buffer_offset,
                        sdpa_bwd_r1_sem_id,
                        sdpa_bwd_r2_sem_id,
                    ]

                sdpa_ctx = {
                    "worker_ncrisc_rt_args": sdpa_worker_ncrisc_rt_args,
                    "worker_brisc_rt_args": sdpa_worker_brisc_rt_args,
                    "worker_trisc_rt_args": sdpa_worker_trisc_rt_args,
                    "forwarder_brisc_base_args": sdpa_forwarder_brisc_base_args,
                    "forwarder_ncrisc_base_args": sdpa_forwarder_ncrisc_base_args,
                    "fabric_node_id": sdpa_fabric_node_id,
                    "fwd_fabric_node_id": fwd_fabric_node_id,
                    "bwd_fabric_node_id": bwd_fabric_node_id,
                }

                per_device_contexts.append(
                    {
                        "mesh_coord": mesh_coord,
                        "ncrisc_compile_time_args": ncrisc_compile_time_args,
                        "brisc_compile_time_args": brisc_compile_time_args,
                        "brisc_named_compile_time_args": brisc_named_compile_time_args,
                        "ncrisc_named_compile_time_args": ncrisc_named_compile_time_args,
                        "trisc_named_compile_time_args": trisc_named_compile_time_args,
                        "brisc_common_runtime_args": brisc_common_runtime_args,
                        "ncrisc_common_runtime_args": ncrisc_common_runtime_args,
                        "trisc_common_runtime_args": trisc_common_runtime_args,
                        "unified_compile_time_core_descriptors": unified_compile_time_core_descriptors,
                        "per_core_compile_time_descriptors": per_core_compile_time_descriptors,
                        "per_core_ncrisc_args": per_core_ncrisc_args,
                        "per_core_brisc_args": per_core_brisc_args,
                        "per_core_trisc_args": per_core_trisc_args,
                        "input_cb_descriptor": in_cb_descriptor,
                        "output_cb_descriptor": attention_block_output_cb_descriptor,
                        "cbs_list": cbs_list,
                        "broadcast_worker_core": broadcast_worker_core,
                        "fabric_node_id": fabric_node_id,
                        "dst_nodes": dst_nodes,
                        # Post-SDPA context
                        "semaphore_list": semaphore_list,
                        "fp32_dest_acc_en": fp32_dest_acc_en,
                        "ccl_sender_core": ccl_sender_core,
                        "gather_core": gather_core,
                        "sdpa_forwarder_cores": sdpa_forwarder_cores,
                        "ccl": ccl_ctx,
                        "sdpa": sdpa_ctx,
                    }
                )

        return full_device_grid, per_device_contexts

    @staticmethod
    def op(
        input_tensor_mesh,
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
        scale,
        output_tensor,
        sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer,
        sender_coord,
        # Post-SDPA parameters
        post_sdpa_weights1_tensor,
        post_sdpa_weights2_tensor,
        post_sdpa_gather2_output_tensor,
        post_sdpa_gather3_output_tensor,
        post_sdpa_intermediate_tensor,
        sdpa_input_l_mesh,
        sdpa_input_ms_mesh,
        sdpa_output_l_mesh,
        sdpa_intermediate_recv_mesh,
        sdpa_forwarder_scratch_mesh,
        sdpa_per_device_chunk_size,
        attention_block_output_tensor,
        # Shared semaphores, and some default values
        attention_block_semaphores=None,
        bcast_cluster_axis=0,
        bcast_secondary_cluster_axis=1,
        reduce_cluster_axis=1,
        sdpa_cluster_axis=0,
        sdpa_scale_fp32=1.0,
        num_links=1,
        epsilon=1e-6,
        fp32_dest_acc_en=False,
        skip_ccl=False,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
    ):
        io_tensors = [
            input_tensor_mesh,
            gamma_tensor.fused_tensor,
            matmul_weights_tensor.fused_tensor,
            matmul3_weights_tensor.fused_tensor,
            trans_mat_tensor,
            qrope_cos_tensor,
            qrope_sin_tensor,
            krope_cos_tensor,
            krope_sin_tensor,
            position_ids_tensor,
            kv_cache_tensor,
            sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer,
            attention_block_output_tensor,
        ]
        cb_id_manager = CircularBufferIdManager()
        cb_id_context = cb_id_manager.create_context()
        full_device_grid, attention_block_per_device_contexts = AttentionBlock.get_program_context(
            input_tensor_mesh,
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
            scale,
            output_tensor,
            sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer,
            sender_coord,
            # Post-SDPA parameters
            post_sdpa_weights1_tensor,
            post_sdpa_weights2_tensor,
            post_sdpa_gather2_output_tensor,
            post_sdpa_gather3_output_tensor,
            post_sdpa_intermediate_tensor,
            sdpa_input_l_mesh,
            sdpa_input_ms_mesh,
            sdpa_output_l_mesh,
            sdpa_intermediate_recv_mesh,
            sdpa_forwarder_scratch_mesh,
            sdpa_per_device_chunk_size,
            attention_block_output_tensor,
            # Shared semaphores, and some default values
            attention_block_semaphores,
            bcast_cluster_axis,
            bcast_secondary_cluster_axis,
            reduce_cluster_axis,
            sdpa_cluster_axis,
            sdpa_scale_fp32,
            num_links,
            epsilon,
            fp32_dest_acc_en,
            skip_ccl,
            noc_mode,
            cb_id_context,
        )

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        for ctx in attention_block_per_device_contexts:
            unified_kernel = UnifiedKernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/fused_ops/attention_block/kernels/attention_block_kernel.cpp",
                core_ranges=full_device_grid,
                ncrisc_compile_time_args=ctx["ncrisc_compile_time_args"],
                brisc_compile_time_args=ctx["brisc_compile_time_args"],
                ncrisc_named_compile_time_args=ctx["ncrisc_named_compile_time_args"],
                ncrisc_common_runtime_args=ctx["ncrisc_common_runtime_args"],
                brisc_named_compile_time_args=ctx["brisc_named_compile_time_args"],
                brisc_common_runtime_args=ctx["brisc_common_runtime_args"],
                trisc_named_compile_time_args=ctx["trisc_named_compile_time_args"],
                trisc_common_runtime_args=ctx["trisc_common_runtime_args"],
                trisc_compute_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=fp32_dest_acc_en,
                    dst_full_sync_en=fp32_dest_acc_en,
                ),
                unified_compile_time_core_descriptors=ctx["unified_compile_time_core_descriptors"],
                per_core_compile_time_descriptors=ctx["per_core_compile_time_descriptors"],
                per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                    ncrisc_args=ctx["per_core_ncrisc_args"],
                    brisc_args=ctx["per_core_brisc_args"],
                    trisc_args=ctx["per_core_trisc_args"],
                ),
                noc_mode=noc_mode,
            )

            kernel_result = unified_kernel.get_kernel_descriptors()
            program = ttnn.ProgramDescriptor(
                kernels=kernel_result.kernels,
                cbs=[ctx["input_cb_descriptor"], *ctx["cbs_list"], ctx["output_cb_descriptor"]],
                semaphores=ctx["semaphore_list"],
            )

            mesh_coord = ctx["mesh_coord"]
            broadcast_worker_core = ctx["broadcast_worker_core"]
            dst_nodes = ctx["dst_nodes"]
            if not skip_ccl and len(dst_nodes) > 0:
                for idx, kernel in enumerate(program.kernels):
                    if kernel.core_ranges.contains(broadcast_worker_core) and (
                        isinstance(kernel.config, ttnn.ReaderConfigDescriptor)
                        or (
                            isinstance(kernel.config, ttnn.DataMovementConfigDescriptor)
                            and kernel.config.processor == ttnn.DataMovementProcessor.RISCV_1
                        )
                    ):
                        writer_rt_args_ref = kernel.runtime_args[broadcast_worker_core.x][broadcast_worker_core.y]
                        fabric_args = ttnn.setup_routing_plane_connection(
                            ctx["fabric_node_id"], dst_nodes, [0], program, idx, broadcast_worker_core
                        )
                        extend_fabric_args(writer_rt_args_ref, fabric_args)
                        break

            # ==================================================================
            # SDPA runtime args and fabric connection setup
            # ==================================================================
            if ctx["sdpa"]:
                sdpa = ctx["sdpa"]
                sdpa_forwarder_cores = ctx["sdpa_forwarder_cores"]

                for group in kernel_result.groups:
                    if group.compile_time_arg_values.get("is_sdpa_worker_core") == 1:
                        crs = group.core_range_set
                        _extend_runtime_args(
                            program.kernels[group.ncrisc_kernel_index].runtime_args, sdpa["worker_ncrisc_rt_args"], crs
                        )
                        _extend_runtime_args(
                            program.kernels[group.brisc_kernel_index].runtime_args, sdpa["worker_brisc_rt_args"], crs
                        )
                        if sdpa["worker_trisc_rt_args"] is not None:
                            _extend_runtime_args(
                                program.kernels[group.trisc_kernel_index].runtime_args,
                                sdpa["worker_trisc_rt_args"],
                                crs,
                            )

                sdpa_forwarder_brisc_rt_args = ttnn.RuntimeArgs()
                sdpa_forwarder_ncrisc_rt_args = ttnn.RuntimeArgs()

                for fwd_idx, fwd_core in enumerate(sdpa_forwarder_cores):
                    sdpa_forwarder_brisc_rt_args[fwd_core.x][fwd_core.y] = list(
                        sdpa["forwarder_brisc_base_args"][(fwd_core.x, fwd_core.y)]
                    )
                    brisc_fabric_args = ttnn.setup_fabric_connection(
                        src_fabric_node_id=sdpa["fabric_node_id"],
                        dst_fabric_node_id=sdpa["fwd_fabric_node_id"],
                        link_idx=fwd_idx,
                        program_descriptor=program,
                        worker_core=fwd_core,
                    )
                    extend_fabric_args(sdpa_forwarder_brisc_rt_args[fwd_core.x][fwd_core.y], brisc_fabric_args)

                    sdpa_forwarder_ncrisc_rt_args[fwd_core.x][fwd_core.y] = list(
                        sdpa["forwarder_ncrisc_base_args"][(fwd_core.x, fwd_core.y)]
                    )
                    ncrisc_fabric_args = ttnn.setup_fabric_connection(
                        src_fabric_node_id=sdpa["fabric_node_id"],
                        dst_fabric_node_id=sdpa["bwd_fabric_node_id"],
                        link_idx=fwd_idx,
                        program_descriptor=program,
                        worker_core=fwd_core,
                    )
                    extend_fabric_args(sdpa_forwarder_ncrisc_rt_args[fwd_core.x][fwd_core.y], ncrisc_fabric_args)

                for group in kernel_result.groups:
                    if group.compile_time_arg_values.get("is_sdpa_forwarder_core") == 1:
                        crs = group.core_range_set
                        _extend_runtime_args(
                            program.kernels[group.brisc_kernel_index].runtime_args, sdpa_forwarder_brisc_rt_args, crs
                        )
                        _extend_runtime_args(
                            program.kernels[group.ncrisc_kernel_index].runtime_args, sdpa_forwarder_ncrisc_rt_args, crs
                        )

            if ctx["ccl"]:
                ccl = ctx["ccl"]
                ccl_sender_core = ctx["ccl_sender_core"]
                gather_core = ctx["gather_core"]

                ccl_sender_group = kernel_result.get_group_by_arg("is_ccl_sender_core", 1)
                ccl_receiver_group = kernel_result.get_group_by_arg("is_ccl_receiver_core", 1)

                sender_brisc_kernel_idx = ccl_sender_group.brisc_kernel_index
                sender_ncrisc_kernel_idx = ccl_sender_group.ncrisc_kernel_index
                receiver_ncrisc_kernel_idx = ccl_receiver_group.ncrisc_kernel_index

                ccl_sender_ncrisc_rt_args_ref = program.kernels[ccl_sender_group.ncrisc_kernel_index].runtime_args[
                    ccl_sender_core.x
                ][ccl_sender_core.y]
                ccl_sender_ncrisc_rt_args_ref.extend(ccl["sender_ncrisc_common_rt_args"])
                ccl_sender_brisc_rt_args_ref = program.kernels[ccl_sender_group.brisc_kernel_index].runtime_args[
                    ccl_sender_core.x
                ][ccl_sender_core.y]
                ccl_sender_brisc_rt_args_ref.extend(ccl["sender_brisc_common_rt_args"])
                ccl_receiver_ncrisc_rt_args_ref = program.kernels[ccl_receiver_group.ncrisc_kernel_index].runtime_args[
                    gather_core.x
                ][gather_core.y]
                ccl_receiver_ncrisc_rt_args_ref.extend(ccl["receiver_ncrisc_common_rt_args"])

                fabric_node_id = ccl["fabric_node_id"]
                neighbor_fabric_node_id = ccl["neighbor_fabric_node_id"]

                sender_brisc_rt_args_ref = program.kernels[sender_brisc_kernel_idx].runtime_args[ccl_sender_core.x][
                    ccl_sender_core.y
                ]
                sender_fabric_args = ttnn.setup_routing_plane_connection(
                    fabric_node_id,
                    [neighbor_fabric_node_id],
                    [ccl["sender_link"]],
                    program,
                    sender_brisc_kernel_idx,
                    ccl_sender_core,
                )
                extend_fabric_args(sender_brisc_rt_args_ref, sender_fabric_args)

                receiver_ncrisc_kernel_idx = ccl_receiver_group.ncrisc_kernel_index
                receiver_ncrisc_rt_args_ref = program.kernels[receiver_ncrisc_kernel_idx].runtime_args[gather_core.x][
                    gather_core.y
                ]

            mesh_program_descriptor[ttnn.MeshCoordinateRange(mesh_coord, mesh_coord)] = program

        result = ttnn.generic_op(io_tensors, mesh_program_descriptor)
        return result
