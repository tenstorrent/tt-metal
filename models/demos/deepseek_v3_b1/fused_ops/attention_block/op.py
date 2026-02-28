# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.pre_sdpa.op import PreSDPA
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor


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
        from models.demos.deepseek_v3_b1.micro_ops.rmsnorm.op import RMSNormSingleCore
        from models.demos.deepseek_v3_b1.micro_ops.rope.op import RopeSingleCore

        position_id = position_ids[0]
        # RMSNorm -> Matmul: [1, K] @ [K, N] -> [1, N]
        input_layernorm = RMSNormSingleCore.golden(input_tensor, gamma_tensor, epsilon)
        matmul_result = input_layernorm @ matmul_weights_tensor

        # RMSNorm2 -> Matmul2: [1, N] @ [N, M] -> [1, M]
        matmul2_result = (
            RMSNormSingleCore.golden(matmul_result, rmsnorm2_gamma_tensor, epsilon) @ matmul2_weights_tensor
        )

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
        kv = RMSNormSingleCore.golden(kv, dkv_rmsnorm_gamma_tensor, epsilon)
        k_rope = RopeSingleCore.golden(k_rope, cos_tensor, sin_tensor, position_ids).squeeze(0)

        # from 0 to position id, the kv cache is valid
        full_kv = kv_cache_tensor.to(full_q.dtype)
        new_kv = torch.cat([kv, k_rope], dim=-1).reshape(1, 1, 1, combined_head_dim).to(full_q.dtype)
        full_kv[:, :, position_id, :] = new_kv

        output = FlashMLADecode.golden(full_q, full_kv, position_ids, nope_dim, scale).squeeze()
        return full_q, new_kv, output

    @staticmethod
    def get_num_semaphores(skip_ccl=False):
        return 10 if skip_ccl else 13

    @staticmethod
    def create_semaphores(mesh_device, skip_ccl=False):
        num_semaphores = AttentionBlock.get_num_semaphores(skip_ccl)
        device_grid_size = mesh_device.compute_with_storage_grid_size()
        available_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )
        semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(num_semaphores)]
        return semaphores

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
        post_sdpa_residual_tensor_mesh,
        sdpa_input_l_mesh,
        sdpa_input_ms_mesh,
        sdpa_output_l_mesh,
        sdpa_r1_recv_mesh,
        sdpa_r2_recv_mesh,
        sdpa_forwarder_scratch_mesh,
        sdpa_per_device_chunk_size,
        attention_block_output_tensor,
        # Shared semaphores, and some default values
        attention_block_semaphores=None,
        cluster_axis=0,
        secondary_cluster_axis=1,
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
            intermediate_tensor_mesh,
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
            output_tensor,
        ]
        sdpa_and_pre_sdpa_semaphores = attention_block_semaphores
        full_device_grid, pre_sdpa_per_device_contexts = PreSDPA.get_program_context(
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
            scale,
            output_tensor,
            sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer,
            sender_coord,
            sdpa_and_pre_sdpa_semaphores,
            cluster_axis,
            secondary_cluster_axis,
            num_links,
            epsilon,
            fp32_dest_acc_en,
            skip_ccl,
            noc_mode,
        )

        post_sdpa_semaphores = attention_block_semaphores
        full_device_grid, post_sdpa_per_device_contexts = PostSDPA.get_program_context(
            input_tensor_mesh,  # TODO: FIX
            post_sdpa_weights1_tensor,
            post_sdpa_weights2_tensor,
            post_sdpa_gather2_output_tensor,
            post_sdpa_gather3_output_tensor,
            post_sdpa_intermediate_tensor,
            post_sdpa_output_tensor,  # TODO: FIX
            post_sdpa_ccl_semaphores,
            1,  # cluster_axis
            post_sdpa_residual_tensor_mesh,
            fp32_dest_acc_en,
            not skip_ccl,
            sdpa_kv_cache_buffer,
            sdpa_input_l_mesh,
            sdpa_input_ms_mesh,
            sdpa_output_l_mesh,
            sdpa_r1_recv_mesh,
            sdpa_r2_recv_mesh,
            sdpa_forwarder_scratch_mesh,
            post_sdpa_semaphores,
            sdpa_scale_fp32,
            sdpa_cluster_axis,
            position_ids_tensor,
            sdpa_per_device_chunk_size,
        )
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        for ctx in per_device_contexts:
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

            program = ttnn.ProgramDescriptor(
                kernels=unified_kernel.get_kernel_descriptors().kernels,
                cbs=[ctx["input_cb_descriptor"], *ctx["cbs_list"], ctx["output_cb_descriptor"]],
                semaphores=[],
            )

            coord = ctx["coord"]
            worker_core = ctx["worker_core"]
            dst_nodes = ctx["dst_nodes"]
            if not skip_ccl and len(dst_nodes) > 0:
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
                            ctx["fabric_node_id"], dst_nodes, [0], program, idx, worker_core
                        )
                        writer_rt_args_ref.extend(fabric_args)
                        break

            mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        result = ttnn.generic_op(io_tensors, mesh_program_descriptor)
        return result
