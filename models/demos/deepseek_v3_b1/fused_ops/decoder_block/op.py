# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Fused Decoder Block: AttentionBlock (pre_sdpa + post_sdpa) + MoE

Builds two program descriptors (one from AttentionBlock, one from MoeOp) and merges
them into a single unified kernel invocation per device.

Merge strategy:
  - Compile-time args: Concatenate attention_block + moe (no conflicts after moe_ prefix rename)
  - CBs: Use attention_block's CBs; MOE's are applied via reconfig_cb_interfaces() at runtime
  - Semaphores: Concatenate both lists (attention_block: local, MOE: global)
  - IO tensors: Union of both ops' tensor lists
  - Kernel: Points to the merged decoder_block_kernel.cpp
"""

from typing import Optional

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.attention_block.op import AttentionBlock, extend_fabric_args
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.fused_ops.post_sdpa.op import _extend_runtime_args
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

KERNEL_PATH = "models/demos/deepseek_v3_b1/fused_ops/decoder_block/kernels/decoder_block_kernel.cpp"


class DecoderBlock:
    """
    Fused MLA + MoE decoder block.

    Phase 1: AttentionBlock (pre_sdpa + post_sdpa)
    Phase 2: CB Reconfiguration
    Phase 3: MoE (routed + shared expert)
    """

    @staticmethod
    def golden(
        # MLA params
        input_tensor,
        gamma_tensor,
        q_a_proj_weights,
        q_norm_gamma,
        q_b_proj_weights,
        kv_b1_proj_weights,
        sin_tensor,
        cos_tensor,
        position_ids,
        kv_a_proj_weights,
        kv_norm_gamma,
        kv_cache_tensor,
        scale,
        kv_b2_weights,
        o_proj_weights,
        # MoE params
        moe_rmsnorm_gamma,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        gate_proj_weights_dict=None,
        up_proj_weights_dict=None,
        down_proj_weights_dict=None,
        # MoE routing params
        enable_routing=True,
        routing_weights_tensor=None,
        bias_tensor=None,
        moe_eps=1e-20,
        moe_scaling_factor=2.5,
        use_hardcoded_expert_index=False,
        hardcoded_expert_index=0,
        explicit_expert_scale=None,
        # Shared
        epsilon=1e-6,
        num_qnope_heads=64,
        num_qrope_heads=64,
        qnope_head_dim=128,
        qrope_head_dim=64,
        nope_dim=512,
        rope_dim=64,
    ):
        """
        End-to-end golden for one decoder block: MLA attention + MoE.

        MLA: input → RMSNorm → Q/KV projections → SDPA → kv_b2 → o_proj → + residual
        MoE: mla_output → RMSNorm → shared expert (+ residual) + routed expert → output

        Args:
            input_tensor: [1, K] hidden state entering this decoder layer
            gamma_tensor .. o_proj_weights: MLA weight tensors (see mla_golden)
            moe_rmsnorm_gamma: [1, K] MoE RMSNorm gamma
            shared_gate/up/down_weights: shared expert weight tensors
            gate/up/down_proj_weights_dict: routed expert weight dicts
            enable_routing: whether to run MoE routing gate
            routing_weights_tensor .. explicit_expert_scale: MoE gate params

        Returns:
            (full_q, new_kv, mla_output, top8_scores, top8_indices, moe_output)
        """
        import torch

        from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
        from models.demos.deepseek_v3_b1.micro_ops.rope.op import RopeSingleCore
        from models.demos.deepseek_v3_b1.unified_kernels.flash_mla_decode import FlashMLADecode

        def rmsnorm(x, gamma):
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + epsilon) * gamma

        position_id = position_ids[0]
        combined_head_dim = nope_dim + rope_dim

        # === MLA: Pre-SDPA ===
        input_layernorm = rmsnorm(input_tensor, gamma_tensor)

        # Q path
        q_compressed = input_layernorm @ q_a_proj_weights
        q_expanded = rmsnorm(q_compressed, q_norm_gamma) @ q_b_proj_weights
        qnope = q_expanded[:, : num_qnope_heads * qnope_head_dim].reshape(num_qnope_heads, 1, qnope_head_dim)
        qrope = q_expanded[:, num_qnope_heads * qnope_head_dim :].reshape(num_qrope_heads, 1, qrope_head_dim)
        qnope_output = torch.bmm(qnope, kv_b1_proj_weights)
        qrope_reshaped = qrope.permute(1, 0, 2).unsqueeze(0)
        position_ids_expanded = position_ids.unsqueeze(1)
        qrope_output = RopeSingleCore.golden(qrope_reshaped, cos_tensor, sin_tensor, position_ids_expanded)
        qrope_output = qrope_output.squeeze(0).permute(1, 0, 2)
        full_q = torch.cat([qnope_output, qrope_output], dim=-1).reshape(1, 1, num_qnope_heads, combined_head_dim)

        # KV path
        dkv = input_layernorm @ kv_a_proj_weights
        kv, k_rope = torch.split(dkv, [nope_dim, rope_dim], dim=-1)
        kv = rmsnorm(kv, kv_norm_gamma)
        k_rope = RopeSingleCore.golden(k_rope, cos_tensor, sin_tensor, position_ids).squeeze(0)
        full_kv = kv_cache_tensor.to(full_q.dtype)
        new_kv = torch.cat([kv, k_rope], dim=-1).reshape(1, 1, 1, combined_head_dim).to(full_q.dtype)
        full_kv[:, :, position_id, :] = new_kv

        # === MLA: SDPA + Post-SDPA ===
        sdpa_output = FlashMLADecode.golden(full_q, full_kv, position_ids, nope_dim, scale).squeeze()
        up_proj_output = torch.bmm(sdpa_output.unsqueeze(1), kv_b2_weights)
        mla_output = input_tensor + up_proj_output.reshape(1, -1) @ o_proj_weights

        # === MoE ===
        top8_scores, top8_indices, moe_output = MoeOp.golden(
            mla_output,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            gate_proj_weights_dict=gate_proj_weights_dict,
            up_proj_weights_dict=up_proj_weights_dict,
            down_proj_weights_dict=down_proj_weights_dict,
            rmsnorm_gamma=moe_rmsnorm_gamma,
            rmsnorm_epsilon=epsilon,
            enable_routing=enable_routing,
            routing_weights_tensor=routing_weights_tensor,
            bias_tensor=bias_tensor,
            eps=moe_eps,
            scaling_factor=moe_scaling_factor,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            hardcoded_expert_index=hardcoded_expert_index,
            explicit_expert_scale=explicit_expert_scale,
        )

        return full_q, new_kv, mla_output, top8_scores, top8_indices, moe_output

    @staticmethod
    def op(
        # AttentionBlock parameters
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
        attention_block_semaphores=None,
        bcast_cluster_axis=0,
        bcast_secondary_cluster_axis=1,
        reduce_cluster_axis=1,
        sdpa_cluster_axis=0,
        sdpa_scale_fp32=1.0,
        num_links=1,
        # MoE parameters
        shared_residual_mcast_src_tensor=None,
        gate_mm_weights_tensor=None,
        gate_bias_tensor=None,
        gate_indices_tensor=None,
        gate_output_scores_tensor=None,
        gate_output_indices_tensor=None,
        gate_proj_weights_tensor=None,
        up_proj_weights_tensor=None,
        down_proj_weights_tensor=None,
        moe_final_output_tensor=None,
        rmsnorm_gamma_tensor=None,
        shared_gate_weights_overlapped=None,
        shared_up_weights_overlapped=None,
        shared_down_weights_tensor=None,
        shared_k_parallel=None,
        shared_n_parallel=None,
        moe_semaphores=None,
        reduce_intermediate_tensors: Optional[list] = None,
        reduce_output_tensor: Optional[ttnn.Tensor] = None,
        reduce_semaphores: Optional[list] = None,
        reduce_root_coord: Optional[ttnn.MeshCoordinate] = None,
        # Shared parameters
        epsilon=1e-6,
        fp32_dest_acc_en=False,
        skip_ccl=False,
        enable_routing=True,
        use_hardcoded_expert_index=False,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
        extra_defines=None,
    ):
        # Phase 1: Build AttentionBlock program context
        full_device_grid, attn_ctxs = AttentionBlock.get_program_context(
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
            attention_block_semaphores=attention_block_semaphores,
            bcast_cluster_axis=bcast_cluster_axis,
            bcast_secondary_cluster_axis=bcast_secondary_cluster_axis,
            reduce_cluster_axis=reduce_cluster_axis,
            sdpa_cluster_axis=sdpa_cluster_axis,
            sdpa_scale_fp32=sdpa_scale_fp32,
            num_links=num_links,
            epsilon=epsilon,
            fp32_dest_acc_en=fp32_dest_acc_en,
            skip_ccl=skip_ccl,
            noc_mode=noc_mode,
        )

        # Phase 2: Build MoE program context with reconfig enabled
        moe = MoeOp(
            shared_residual_mcast_src_tensor,
            gate_mm_weights_tensor=gate_mm_weights_tensor,
            gate_bias_tensor=gate_bias_tensor,
            gate_indices_tensor=gate_indices_tensor,
            gate_output_scores_tensor=gate_output_scores_tensor,
            gate_output_indices_tensor=gate_output_indices_tensor,
            gate_proj_weights_tensor=gate_proj_weights_tensor,
            up_proj_weights_tensor=up_proj_weights_tensor,
            down_proj_weights_tensor=down_proj_weights_tensor,
            final_output_tensor=moe_final_output_tensor,
            rmsnorm_gamma_tensor=rmsnorm_gamma_tensor,
            shared_gate_weights_overlapped=shared_gate_weights_overlapped,
            shared_up_weights_overlapped=shared_up_weights_overlapped,
            shared_down_weights_tensor=shared_down_weights_tensor,
            shared_k_parallel=shared_k_parallel,
            shared_n_parallel=shared_n_parallel,
            epsilon=epsilon,
            enable_routing=enable_routing,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer=sdpa_out_interm_buffer,
            reduce_intermediate_tensors=reduce_intermediate_tensors,
            reduce_output_tensor=reduce_output_tensor,
            reduce_semaphores=reduce_semaphores,
            reduce_root_coord=reduce_root_coord,
            reconfig_moe_cbs=True,
            semaphores=moe_semaphores,
            noc_mode=noc_mode,
        )
        moe._build_descriptors()
        moe_ctx = moe.ctx

        # Build unified IO tensors
        attn_io = [
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
            attention_block_output_tensor,
        ]
        from models.demos.deepseek_v3_b1.blitz_decode_weights import OverlappedTensor

        def _unwrap(t):
            return t.fused_tensor if isinstance(t, OverlappedTensor) else t

        io_tensors = attn_io + [_unwrap(t) for t in moe.io_tensors]

        # Phase 3: Merge per-device and execute
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        for ac in attn_ctxs:
            coord = ac["mesh_coord"]
            row = getattr(coord, "row", 0)
            col = getattr(coord, "col", 0)
            chip_id = row * moe_ctx.mesh_cols + col

            moe._setup_per_device_args(chip_id, 1, reduce_root_coord, coord, row, col)
            reconfig_addr = moe.reconfig_tensor.buffer_address()

            merged_ncrisc = (
                ac["ncrisc_named_compile_time_args"] + moe.ncrisc_args + [("reconfig_cb_config_l1_addr", reconfig_addr)]
            )
            merged_brisc = (
                ac["brisc_named_compile_time_args"] + moe.brisc_args + [("reconfig_cb_config_l1_addr", reconfig_addr)]
            )
            merged_trisc = (
                ac["trisc_named_compile_time_args"] + moe.trisc_args + [("reconfig_cb_config_l1_addr", reconfig_addr)]
            )

            merged_ncrisc_crt = ac.get("ncrisc_common_runtime_args", []) + moe.ncrisc_common_rt_args
            merged_brisc_crt = ac.get("brisc_common_runtime_args", [])
            merged_trisc_crt = ac.get("trisc_common_runtime_args", []) + [
                moe_ctx.rmsnorm_epsilon_packed,
                moe_ctx.rmsnorm_scalar_packed,
            ]

            merged_ucd = ac["unified_compile_time_core_descriptors"] + moe.device_unified_core_descs
            merged_pcd = ac["per_core_compile_time_descriptors"] + moe.device_per_core_descs

            moe_defines = dict(moe.kernel_defines) if moe.kernel_defines else {}
            if not skip_ccl:
                moe_defines.pop("SKIP_CCL", None)
            if extra_defines:
                moe_defines.update(extra_defines)

            uk = UnifiedKernelDescriptor(
                kernel_source=KERNEL_PATH,
                core_ranges=full_device_grid,
                ncrisc_compile_time_args=ac.get("ncrisc_compile_time_args", []),
                brisc_compile_time_args=ac.get("brisc_compile_time_args", []),
                ncrisc_named_compile_time_args=merged_ncrisc,
                ncrisc_common_runtime_args=merged_ncrisc_crt,
                brisc_named_compile_time_args=merged_brisc,
                brisc_common_runtime_args=merged_brisc_crt,
                trisc_named_compile_time_args=merged_trisc,
                trisc_common_runtime_args=merged_trisc_crt,
                trisc_compute_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=fp32_dest_acc_en,
                    dst_full_sync_en=fp32_dest_acc_en,
                ),
                unified_compile_time_core_descriptors=merged_ucd,
                per_core_compile_time_descriptors=merged_pcd,
                per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                    ncrisc_args=ac.get("per_core_ncrisc_args", [])
                    + (moe.device_rt_args_desc.ncrisc_args if moe.device_rt_args_desc else []),
                    brisc_args=ac.get("per_core_brisc_args", [])
                    + (moe.device_rt_args_desc.brisc_args if moe.device_rt_args_desc else []),
                    trisc_args=ac.get("per_core_trisc_args", [])
                    + (moe.device_rt_args_desc.trisc_args if moe.device_rt_args_desc else []),
                ),
                defines=list(moe_defines.items()),
                noc_mode=noc_mode,
            )

            kr = uk.get_kernel_descriptors()

            merged_cbs = [ac["input_cb_descriptor"], *ac["cbs_list"], ac["output_cb_descriptor"]]

            attn_sems = ac.get("semaphore_list", [])
            program = ttnn.ProgramDescriptor(
                kernels=kr.kernels,
                cbs=merged_cbs,
                semaphores=attn_sems + moe.device_sem_descs,
            )

            # AttentionBlock fabric connections (CCL broadcast)
            wc = ac.get("broadcast_worker_core")
            dn = ac.get("dst_nodes", [])
            if not skip_ccl and dn and len(dn) > 0:
                for idx, kernel in enumerate(program.kernels):
                    if kernel.core_ranges.contains(wc) and (
                        isinstance(kernel.config, ttnn.ReaderConfigDescriptor)
                        or (
                            isinstance(kernel.config, ttnn.DataMovementConfigDescriptor)
                            and kernel.config.processor == ttnn.DataMovementProcessor.RISCV_1
                        )
                    ):
                        rt_ref = kernel.runtime_args[wc.x][wc.y]
                        fa = ttnn.setup_routing_plane_connection(ac["fabric_node_id"], dn, [0], program, idx, wc)
                        extend_fabric_args(rt_ref, fa)
                        break

            # SDPA runtime args and fabric connection setup
            if ac.get("sdpa"):
                sdpa = ac["sdpa"]
                sdpa_forwarder_cores = ac["sdpa_forwarder_cores"]

                for group in kr.groups:
                    if group.compile_time_arg_values.get("is_sdpa_worker_core") == 1:
                        crs = group.core_range_set
                        _extend_runtime_args(
                            program.kernels[group.ncrisc_kernel_index].runtime_args,
                            sdpa["worker_ncrisc_rt_args"],
                            crs,
                        )
                        _extend_runtime_args(
                            program.kernels[group.brisc_kernel_index].runtime_args,
                            sdpa["worker_brisc_rt_args"],
                            crs,
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

                for group in kr.groups:
                    if group.compile_time_arg_values.get("is_sdpa_forwarder_core") == 1:
                        crs = group.core_range_set
                        _extend_runtime_args(
                            program.kernels[group.brisc_kernel_index].runtime_args,
                            sdpa_forwarder_brisc_rt_args,
                            crs,
                        )
                        _extend_runtime_args(
                            program.kernels[group.ncrisc_kernel_index].runtime_args,
                            sdpa_forwarder_ncrisc_rt_args,
                            crs,
                        )

            # CCL all-reduce fabric connection setup
            if ac.get("ccl"):
                ccl = ac["ccl"]
                ccl_sender_core = ac["ccl_sender_core"]
                gather_core = ac["gather_core"]

                ccl_sender_group = kr.get_group_by_arg("is_ccl_sender_core", 1)
                ccl_receiver_group = kr.get_group_by_arg("is_ccl_receiver_core", 1)

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

                sender_brisc_kernel_idx = ccl_sender_group.brisc_kernel_index
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

            # MOE fabric connections (reduce-to-one)
            moe._setup_fabric_connections(coord, row, col, reduce_root_coord, kr, program)

            mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        ttnn.generic_op(io_tensors, mesh_program_descriptor)

        if moe_ctx.enable_reduce_to_one:
            if moe_ctx.enable_routing:
                return (
                    moe_ctx.gate_output_scores_tensor,
                    moe_ctx.gate_output_indices_tensor,
                    moe_ctx.reduce_output_tensor,
                )
            return moe_ctx.reduce_output_tensor
        if moe_ctx.enable_routing:
            return moe_ctx.gate_output_scores_tensor, moe_ctx.gate_output_indices_tensor, moe_ctx.final_output_tensor
        return moe_ctx.final_output_tensor
