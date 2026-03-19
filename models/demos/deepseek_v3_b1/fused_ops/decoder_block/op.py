# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import ttnn
from models.demos.deepseek_v3_b1.circular_buffer_utils import (
    CircularBufferIdManager,
    build_cb_reconfig_tensor,
    record_cb_metadata,
)
from models.demos.deepseek_v3_b1.fused_ops.attention_block.op import AttentionBlock, extend_fabric_args
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp, MoeSem
from models.demos.deepseek_v3_b1.fused_ops.post_sdpa.op import _extend_runtime_args
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor


class DecoderBlock:
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
        post_sdpa_weights1,
        post_sdpa_weights2,
        epsilon=1e-6,
        num_qnope_heads=64,
        num_qrope_heads=64,
        qnope_head_dim=128,
        qrope_head_dim=64,
        heads_per_row=8,
        nope_dim=512,
        rope_dim=64,
        # MoE parameters (when None, only attention golden is computed)
        moe_shared_gate_weights=None,
        moe_shared_up_weights=None,
        moe_shared_down_weights=None,
        moe_gate_proj_weights_dict=None,
        moe_up_proj_weights_dict=None,
        moe_down_proj_weights_dict=None,
        moe_rmsnorm_gamma=None,
        moe_rmsnorm_epsilon=None,
        moe_routing_weights=None,
        moe_bias=None,
        moe_gate_eps=1e-20,
        moe_gate_scaling_factor=2.5,
        moe_enable_routing=True,
    ):
        full_q, new_kv, attn_output = AttentionBlock.golden(
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
            post_sdpa_weights1,
            post_sdpa_weights2,
            epsilon=epsilon,
            num_qnope_heads=num_qnope_heads,
            num_qrope_heads=num_qrope_heads,
            qnope_head_dim=qnope_head_dim,
            qrope_head_dim=qrope_head_dim,
            heads_per_row=heads_per_row,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )
        if moe_shared_gate_weights is None:
            return full_q, new_kv, attn_output, None, None, None

        if moe_rmsnorm_epsilon is None:
            moe_rmsnorm_epsilon = epsilon

        moe_golden_kwargs = dict(
            rmsnorm_gamma=moe_rmsnorm_gamma,
            rmsnorm_epsilon=moe_rmsnorm_epsilon,
            routing_weights_tensor=moe_routing_weights,
            bias_tensor=moe_bias,
            eps=moe_gate_eps,
            scaling_factor=moe_gate_scaling_factor,
        )

        # Golden path models the logical block semantics directly:
        # h = x + Attention(x), y = h + MoE(h). No per-device decomposition here.
        moe_scores, moe_indices, moe_output = MoeOp.golden(
            attn_output,
            shared_gate_weights=moe_shared_gate_weights,
            shared_up_weights=moe_shared_up_weights,
            shared_down_weights=moe_shared_down_weights,
            gate_proj_weights_dict=moe_gate_proj_weights_dict,
            up_proj_weights_dict=moe_up_proj_weights_dict,
            down_proj_weights_dict=moe_down_proj_weights_dict,
            routing_mode=moe_enable_routing,
            include_residual=True,
            **moe_golden_kwargs,
        )
        return full_q, new_kv, attn_output, moe_scores, moe_indices, moe_output

    @staticmethod
    def get_num_semaphores():
        return AttentionBlock.get_num_semaphores() + MoeSem.NUM_SEMAPHORES

    @staticmethod
    def create_semaphores(mesh_device):
        return AttentionBlock.create_semaphores(mesh_device) + MoeOp.create_semaphores(mesh_device)

    @staticmethod
    def op(
        # AttentionBlock parameters
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
        position_ids_tensor,
        sdpa_scale,
        sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer,
        sender_coord,
        post_sdpa_weights1_tensor,
        post_sdpa_weights2_tensor,
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
        reduce_intermediate_tensors=None,
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
        num_iterations=1,
        upstream_socket=None,
        downstream_socket=None,
        persistent_next_iter_semaphore=None,
        persistent_mode=False,
        is_torus=True,
    ):
        cb_id_manager = CircularBufferIdManager()
        mla_cb_id_context = cb_id_manager.create_context()
        moe_cb_id_context = cb_id_manager.create_context()
        full_device_grid, decoder_cbs, decoder_per_device_contexts = AttentionBlock.get_program_context(
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
            position_ids_tensor,
            sdpa_scale,
            None,
            sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer,
            sender_coord,
            # Post-SDPA parameters
            post_sdpa_weights1_tensor,
            post_sdpa_weights2_tensor,
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
            num_links,
            epsilon,
            fp32_dest_acc_en,
            skip_ccl,
            noc_mode,
            mla_cb_id_context,
            upstream_socket=upstream_socket,
        )

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
            cb_id_context=moe_cb_id_context,
            downstream_socket=downstream_socket,
            persistent_next_iter_semaphore=persistent_next_iter_semaphore,
            persistent_mode=persistent_mode,
            bcast_sender_coord=sender_coord,
            is_torus=is_torus,
        )

        moe._build_descriptors()
        moe_ctx = moe.ctx

        io_tensors = []
        cb_metadata = record_cb_metadata(decoder_cbs)
        reconfig_tensor = build_cb_reconfig_tensor(cb_metadata, full_device_grid, input_tensor_mesh.device())
        io_tensors.append(reconfig_tensor)
        cbs_list = cb_id_manager.build_dummy_cb_descriptors(full_device_grid)

        # TODO: Passing the address here as a named compile time arg is not ideal. Done for simplicity.
        additional_named_compile_time_args = [
            ("mla_reconfig_cb_config_l1_addr", reconfig_tensor.buffer_address()),
            ("num_iterations", num_iterations),
        ]

        io_tensors += [
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
        io_tensors += moe.io_tensors

        def _patch_named_compile_time_args(named_args, overrides):
            """Replace selected named compile-time args while preserving order."""
            return [(name, overrides.get(name, value)) for name, value in named_args]

        def _adapt_moe_ct_args(named_args):
            """Rename/filter MoE CT args that conflict with attention CT args."""
            result = []
            for name, value in named_args:
                if name == "num_iterations":
                    continue
                if name == "reconfig_cb_config_l1_addr":
                    result.append(("moe_reconfig_cb_config_l1_addr", value))
                else:
                    result.append((name, value))
            return result

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        for ctx in decoder_per_device_contexts:
            mesh_coord = ctx["mesh_coord"]
            row = mesh_coord[0]
            col = mesh_coord[1]
            chip_id = row * moe_ctx.mesh_cols + col

            # ── MoE per-device setup ──
            moe._setup_per_device_args(chip_id, num_iterations, reduce_root_coord, mesh_coord, row, col)

            moe_ncrisc_ct = _adapt_moe_ct_args(moe.ncrisc_args)
            moe_brisc_ct = _adapt_moe_ct_args(moe.brisc_args)
            moe_trisc_ct = _adapt_moe_ct_args(moe.trisc_args)
            attn_ncrisc_common = ctx["ncrisc_common_runtime_args"]
            attn_trisc_common = ctx["trisc_common_runtime_args"]
            moe_ncrisc_ct = _patch_named_compile_time_args(
                moe_ncrisc_ct, {"reduce_ncrisc_common_rt_arg_base": len(attn_ncrisc_common)}
            )
            moe_trisc_ct = _patch_named_compile_time_args(
                moe_trisc_ct, {"moe_rmsnorm_trisc_common_rt_arg_base": len(attn_trisc_common)}
            )
            attn_per_core_brisc = ctx["per_core_brisc_args"]
            attn_brisc_prefix_len_by_core = {}
            for core_coord, args in attn_per_core_brisc:
                core_key = (core_coord.x, core_coord.y)
                attn_brisc_prefix_len_by_core[core_key] = attn_brisc_prefix_len_by_core.get(core_key, 0) + len(args)
            moe_per_core_brisc = moe.device_rt_args_desc.brisc_args if moe.device_rt_args_desc else []

            # Compute the correct bases and patch directly into moe_brisc_ct.
            # Both worker and fabric cores start reading their moe per-core args immediately after
            # the attn per-core args, so both bases equal attn_base. All reduce cores (worker and
            # fabric) must have the same attn_base since attn assigns the same per-core args to
            # every core on the device.
            if moe_per_core_brisc:
                attn_bases = {attn_brisc_prefix_len_by_core.get((c.x, c.y), 0) for c, _ in moe_per_core_brisc}
                assert (
                    len(attn_bases) == 1
                ), f"All reduce cores must have the same attn per-core arg count, got: {attn_bases}"
                reduce_rt_arg_base = next(iter(attn_bases))
            else:
                reduce_rt_arg_base = 0
            moe_brisc_ct = _patch_named_compile_time_args(
                moe_brisc_ct,
                {
                    "reduce_brisc_rt_arg_base": reduce_rt_arg_base,
                    "reduce_brisc_fabric_rt_arg_base": reduce_rt_arg_base,
                },
            )
            merged_ucd = ctx["unified_compile_time_core_descriptors"] + moe.device_unified_core_descs
            merged_pcd = ctx["per_core_compile_time_descriptors"] + moe.device_per_core_descs
            mesh_coord_args = [("mesh_row", row), ("mesh_col", col)]

            my_defines = moe.kernel_defines
            if ctx["device_kernel_defines"] is not None:
                my_defines = ctx["device_kernel_defines"] + moe.kernel_defines
            unified_kernel = UnifiedKernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/fused_ops/decoder_block/kernels/decoder_block_kernel.cpp",
                core_ranges=full_device_grid,
                defines=my_defines,
                ncrisc_compile_time_args=ctx["ncrisc_compile_time_args"],
                brisc_compile_time_args=ctx["brisc_compile_time_args"],
                ncrisc_named_compile_time_args=mesh_coord_args
                + ctx["ncrisc_named_compile_time_args"]
                + moe_ncrisc_ct
                + additional_named_compile_time_args,
                ncrisc_common_runtime_args=ctx["ncrisc_common_runtime_args"] + moe.ncrisc_common_rt_args,
                brisc_named_compile_time_args=mesh_coord_args
                + ctx["brisc_named_compile_time_args"]
                + moe_brisc_ct
                + additional_named_compile_time_args,
                brisc_common_runtime_args=ctx["brisc_common_runtime_args"],
                trisc_named_compile_time_args=mesh_coord_args
                + ctx["trisc_named_compile_time_args"]
                + moe_trisc_ct
                + additional_named_compile_time_args,
                trisc_common_runtime_args=ctx["trisc_common_runtime_args"]
                + [moe_ctx.rmsnorm_epsilon_packed, moe_ctx.rmsnorm_scalar_packed],
                trisc_compute_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=fp32_dest_acc_en,
                    dst_full_sync_en=fp32_dest_acc_en,
                ),
                unified_compile_time_core_descriptors=merged_ucd,
                per_core_compile_time_descriptors=merged_pcd,
                per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                    ncrisc_args=ctx["per_core_ncrisc_args"]
                    + (moe.device_rt_args_desc.ncrisc_args if moe.device_rt_args_desc else []),
                    brisc_args=ctx["per_core_brisc_args"]
                    + (moe.device_rt_args_desc.brisc_args if moe.device_rt_args_desc else []),
                    trisc_args=ctx["per_core_trisc_args"]
                    + (moe.device_rt_args_desc.trisc_args if moe.device_rt_args_desc else []),
                ),
                noc_mode=noc_mode,
            )
            kernel_result = unified_kernel.get_kernel_descriptors()
            program = ttnn.ProgramDescriptor(
                kernels=kernel_result.kernels,
                cbs=cbs_list,
                semaphores=ctx["semaphore_list"] + moe.device_sem_descs,
            )
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

            # MoE fabric connections (reduce-to-one)
            moe._setup_fabric_connections(mesh_coord, row, col, reduce_root_coord, kernel_result, program)
            mesh_program_descriptor[ttnn.MeshCoordinateRange(mesh_coord, mesh_coord)] = program
        result = ttnn.generic_op(io_tensors, mesh_program_descriptor)
        return result, attention_block_output_tensor
