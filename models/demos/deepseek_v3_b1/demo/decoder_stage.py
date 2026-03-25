# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Decoder and dense block pipeline stages (socket-fed bcast + fused DecoderBlock + reduce-to-one)."""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.demo.stage import StageContext, StageKind
from models.demos.deepseek_v3_b1.fused_ops.attention_block.op import AttentionBlock
from models.demos.deepseek_v3_b1.fused_ops.decoder_block.op import DecoderBlock
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.prepare_weights import DeepSeekV3DenseLayerWeights, DeepSeekV3MoELayerWeights
from models.demos.deepseek_v3_b1.tests.unit_tests.test_decoder_block import create_decoder_block_tensors


class DecoderBlockStage(StageKind):
    """Decoder block compute stage: bcast + fused attention + MoE + reduce-to-one.

    Accepts weights in two forms (mutually exclusive):
    - ``weights``: a pre-loaded :class:`DeepSeekV3MoELayerWeights` (production path via
      ``WeightProvider.load_moe_layer``).
    - ``state_dict``: a raw HF-format state dict (test path; requires ``layer_idx`` and
      ``num_routed_experts`` for weight processing).
    """

    PIPELINE_CORE = ttnn.CoreCoord(12, 8)
    MOE_SENDER_CORE = ttnn.CoreCoord(12, 9)
    M = 1
    K = 7168
    EMBEDDING_SIZE_BYTES = K * 2  # bfloat16
    EMBEDDING_FIFO_SIZE = EMBEDDING_SIZE_BYTES * 1
    TOKEN_SIZE_BYTES = 64

    def __init__(
        self,
        state_dict: dict[str, torch.Tensor] | None = None,
        *,
        weights: DeepSeekV3MoELayerWeights | None = None,
        layer_idx: int = 4,
        num_routed_experts: int = 256,
        position_id: int = 0,
        max_seq_len: int = 32 * 1024,
        persistent_mode: bool = True,
        use_hardcoded_expert_index: bool = False,
        enable_routing: bool = True,
        is_torus: bool = True,
    ) -> None:
        if state_dict is None and weights is None:
            raise ValueError("Either state_dict or weights must be provided")
        if state_dict is not None and weights is not None:
            raise ValueError("Provide state_dict or weights, not both")
        self._state_dict = state_dict
        self._weights = weights
        self._layer_idx = layer_idx
        self._num_routed_experts = num_routed_experts
        self._position_id = position_id
        self._max_seq_len = max_seq_len
        self._persistent_mode = persistent_mode
        self._use_hardcoded_expert_index = use_hardcoded_expert_index
        self._enable_routing = enable_routing
        self._is_torus = is_torus
        self._state: dict[str, Any] = {}

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        pipeline_config = ctx.pipeline_config
        my_mesh_id = ctx.my_mesh_id

        gate_proj_noc = ttnn.NOC.NOC_0
        gate_proj_worker_cores = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
        gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
        shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
        aggregator_core = shard_cores_list[0]

        stage_entry_device = pipeline_config[my_mesh_id].entry_node_coord
        reduce_root_coord = pipeline_config[my_mesh_id].exit_node_coord

        return PipelineBlock(
            mesh_device,
            self.PIPELINE_CORE,
            upstream_d2d_socket_fifo_size=self.EMBEDDING_FIFO_SIZE,
            downstream_d2d_socket_fifo_size=self.EMBEDDING_FIFO_SIZE,
            upstream_d2d_socket_page_size=self.EMBEDDING_SIZE_BYTES,
            downstream_d2d_socket_page_size=self.EMBEDDING_SIZE_BYTES,
            entry_node_downstream=ttnn.MeshCoreCoord(stage_entry_device, self.MOE_SENDER_CORE),
            exit_node_upstream=ttnn.MeshCoreCoord(reduce_root_coord, aggregator_core),
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        mesh_device = ctx.mesh_device
        pipeline_config = ctx.pipeline_config
        my_mesh_id = ctx.my_mesh_id

        sender_coord = pipeline_config[my_mesh_id].entry_node_coord
        reduce_root_coord = pipeline_config[my_mesh_id].exit_node_coord

        downstream_socket = pipeline_block.exit_socket_interface.get_upstream_socket()

        num_cores = mesh_device.compute_with_storage_grid_size().x * mesh_device.compute_with_storage_grid_size().y
        available_cores = ttnn.num_cores_to_corerangeset(
            num_cores, mesh_device.compute_with_storage_grid_size(), row_wise=True
        )

        attn_semaphores = AttentionBlock.create_semaphores(mesh_device)
        moe_semaphores = MoeOp.create_semaphores(mesh_device)
        reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(4)]
        persistent_next_iter_semaphore = (
            ttnn.create_global_semaphore(mesh_device, available_cores, 1) if self._persistent_mode else None
        )

        d = create_decoder_block_tensors(
            mesh_device,
            mesh_device.shape[0],
            mesh_device.shape[1],
            sender_coord[0],
            sender_coord[1],
            self._position_id,
            self._state_dict,
            self._layer_idx,
            self._max_seq_len,
            reduce_root_coord=reduce_root_coord,
            num_routed_experts=self._num_routed_experts,
            preloaded_weights=self._weights,
        )
        ttnn.synchronize_device(mesh_device)

        recv_socket = pipeline_block.get_downstream_socket()

        self._state = {
            "d": d,
            "attn_semaphores": attn_semaphores,
            "moe_semaphores": moe_semaphores,
            "reduce_semaphores": reduce_semaphores,
            "reduce_root_coord": reduce_root_coord,
            "recv_socket": recv_socket,
            "downstream_socket": downstream_socket,
        }

        if persistent_next_iter_semaphore is not None:
            self._state["persistent_next_iter_semaphore"] = persistent_next_iter_semaphore

        logger.info(f"[rank={my_mesh_id}] DecoderBlockStage setup complete")

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        d = self._state["d"]
        DecoderBlock.op(
            d["input_tensor_mesh"],
            d["gamma_overlapped"],
            d["matmul_weights_overlapped"],
            d["rmsnorm2_gamma_overlapped"],
            d["matmul2_weights_overlapped"],
            d["matmul3_weights_overlapped"],
            d["ttnn_qrope_sin"],
            d["ttnn_qrope_cos"],
            d["ttnn_trans_mat"],
            d["ttnn_krope_cos"],
            d["ttnn_krope_sin"],
            d["dkv_matmul_weights_overlapped"],
            d["dkv_rmsnorm_gamma_overlapped"],
            d["ttnn_kv_cache"],
            d["ttnn_position_ids"],
            d["scale"],
            d["sdpa_kv_cache_buffer"],
            d["sdpa_out_interm_buffer"],
            d["sender_coord"],
            d["kv_b2_overlapped"],
            d["o_proj_overlapped"],
            None,
            None,
            None,
            d["ttnn_sdpa_intermediate_recv"],
            d["ttnn_sdpa_forwarder_scratch"],
            d["device_chunk_size"],
            d["ttnn_attention_block_output"],
            attention_block_semaphores=self._state["attn_semaphores"],
            shared_residual_mcast_src_tensor=d["ttnn_residual_mcast_src"],
            gate_mm_weights_tensor=d["gate_mm_overlapped"],
            gate_bias_tensor=d["ttnn_gate_bias"],
            gate_indices_tensor=d["ttnn_gate_indices"],
            gate_output_scores_tensor=d["gate_output_scores_tensor"],
            gate_output_indices_tensor=d["gate_output_indices_tensor"],
            gate_proj_weights_tensor=d["gate_proj_weights"],
            up_proj_weights_tensor=d["up_proj_weights"],
            down_proj_weights_tensor=d["down_proj_weights"],
            moe_final_output_tensor=None,
            rmsnorm_gamma_tensor=d["ffn_norm_overlapped"],
            shared_gate_weights_overlapped=d["shared_gate_weights_overlapped"],
            shared_up_weights_overlapped=d["shared_up_weights_overlapped"],
            shared_down_weights_tensor=d["shared_down_weights_tensor"],
            shared_k_parallel=d["shared_k_parallel"],
            shared_n_parallel=d["shared_n_parallel"],
            moe_semaphores=self._state["moe_semaphores"],
            reduce_intermediate_tensors=d["reduce_intermediate_tensors"],
            reduce_output_tensor=d["reduce_output_tensor"],
            reduce_semaphores=self._state["reduce_semaphores"],
            reduce_root_coord=self._state["reduce_root_coord"],
            enable_routing=self._enable_routing,
            use_hardcoded_expert_index=self._use_hardcoded_expert_index,
            bcast_cluster_axis=0,
            bcast_secondary_cluster_axis=1,
            reduce_cluster_axis=1,
            sdpa_cluster_axis=0,
            sdpa_scale_fp32=d["scale"],
            num_links=1,
            skip_ccl=False,
            upstream_socket=self._state["recv_socket"],
            downstream_socket=self._state["downstream_socket"],
            persistent_next_iter_semaphore=self._state.get("persistent_next_iter_semaphore"),
            persistent_mode=self._persistent_mode,
            is_torus=self._is_torus,
        )


class DenseBlockStage(StageKind):
    """Dense decoder block stage: bcast + fused attention + dense MLP + reduce-to-one.

    Accepts weights in two forms (mutually exclusive):
    - ``weights``: a pre-loaded :class:`DeepSeekV3DenseLayerWeights` (production path via
      ``WeightProvider.load_dense_layer``).
    - ``state_dict``: a raw HF-format state dict (test path; requires ``layer_idx``).
    """

    PIPELINE_CORE = ttnn.CoreCoord(12, 8)
    MOE_SENDER_CORE = ttnn.CoreCoord(12, 9)
    M = 1
    K = 7168
    EMBEDDING_SIZE_BYTES = K * 2  # bfloat16
    EMBEDDING_FIFO_SIZE = EMBEDDING_SIZE_BYTES * 1
    TOKEN_SIZE_BYTES = 64

    def __init__(
        self,
        state_dict: dict[str, torch.Tensor] | None = None,
        *,
        weights: DeepSeekV3DenseLayerWeights | None = None,
        layer_idx: int = 0,
        position_id: int = 0,
        max_seq_len: int = 32 * 1024,
        persistent_mode: bool = True,
        is_torus: bool = True,
    ) -> None:
        if state_dict is None and weights is None:
            raise ValueError("Either state_dict or weights must be provided")
        if state_dict is not None and weights is not None:
            raise ValueError("Provide state_dict or weights, not both")
        self._state_dict = state_dict
        self._weights = weights
        self._layer_idx = layer_idx
        self._position_id = position_id
        self._max_seq_len = max_seq_len
        self._persistent_mode = persistent_mode
        self._is_torus = is_torus
        self._state: dict[str, Any] = {}

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        pipeline_config = ctx.pipeline_config
        my_mesh_id = ctx.my_mesh_id

        gate_proj_noc = ttnn.NOC.NOC_0
        gate_proj_worker_cores = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
        gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
        shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
        aggregator_core = shard_cores_list[0]

        stage_entry_device = pipeline_config[my_mesh_id].entry_node_coord
        reduce_root_coord = pipeline_config[my_mesh_id].exit_node_coord

        return PipelineBlock(
            mesh_device,
            self.PIPELINE_CORE,
            upstream_d2d_socket_fifo_size=self.EMBEDDING_FIFO_SIZE,
            downstream_d2d_socket_fifo_size=self.EMBEDDING_FIFO_SIZE,
            upstream_d2d_socket_page_size=self.EMBEDDING_SIZE_BYTES,
            downstream_d2d_socket_page_size=self.EMBEDDING_SIZE_BYTES,
            entry_node_downstream=ttnn.MeshCoreCoord(stage_entry_device, self.MOE_SENDER_CORE),
            exit_node_upstream=ttnn.MeshCoreCoord(reduce_root_coord, aggregator_core),
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        mesh_device = ctx.mesh_device
        pipeline_config = ctx.pipeline_config
        my_mesh_id = ctx.my_mesh_id

        sender_coord = pipeline_config[my_mesh_id].entry_node_coord
        reduce_root_coord = pipeline_config[my_mesh_id].exit_node_coord

        downstream_socket = pipeline_block.exit_socket_interface.get_upstream_socket()

        num_cores = mesh_device.compute_with_storage_grid_size().x * mesh_device.compute_with_storage_grid_size().y
        available_cores = ttnn.num_cores_to_corerangeset(
            num_cores, mesh_device.compute_with_storage_grid_size(), row_wise=True
        )

        attn_semaphores = AttentionBlock.create_semaphores(mesh_device)
        moe_semaphores = MoeOp.create_semaphores(mesh_device)
        reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(4)]
        persistent_next_iter_semaphore = (
            ttnn.create_global_semaphore(mesh_device, available_cores, 1) if self._persistent_mode else None
        )

        d = create_decoder_block_tensors(
            mesh_device,
            mesh_device.shape[0],
            mesh_device.shape[1],
            sender_coord[0],
            sender_coord[1],
            self._position_id,
            self._state_dict,
            self._layer_idx,
            self._max_seq_len,
            reduce_root_coord=reduce_root_coord,
            is_moe=False,
            preloaded_weights=self._weights,
        )
        ttnn.synchronize_device(mesh_device)

        recv_socket = pipeline_block.get_downstream_socket()

        self._state = {
            "d": d,
            "attn_semaphores": attn_semaphores,
            "moe_semaphores": moe_semaphores,
            "reduce_semaphores": reduce_semaphores,
            "reduce_root_coord": reduce_root_coord,
            "recv_socket": recv_socket,
            "downstream_socket": downstream_socket,
        }

        if persistent_next_iter_semaphore is not None:
            self._state["persistent_next_iter_semaphore"] = persistent_next_iter_semaphore

        logger.info(f"[rank={my_mesh_id}] DenseBlockStage setup complete")

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        d = self._state["d"]
        DecoderBlock.op(
            d["input_tensor_mesh"],
            d["gamma_overlapped"],
            d["matmul_weights_overlapped"],
            d["rmsnorm2_gamma_overlapped"],
            d["matmul2_weights_overlapped"],
            d["matmul3_weights_overlapped"],
            d["ttnn_qrope_sin"],
            d["ttnn_qrope_cos"],
            d["ttnn_trans_mat"],
            d["ttnn_krope_cos"],
            d["ttnn_krope_sin"],
            d["dkv_matmul_weights_overlapped"],
            d["dkv_rmsnorm_gamma_overlapped"],
            d["ttnn_kv_cache"],
            d["ttnn_position_ids"],
            d["scale"],
            d["sdpa_kv_cache_buffer"],
            d["sdpa_out_interm_buffer"],
            d["sender_coord"],
            d["kv_b2_overlapped"],
            d["o_proj_overlapped"],
            None,
            None,
            None,
            d["ttnn_sdpa_intermediate_recv"],
            d["ttnn_sdpa_forwarder_scratch"],
            d["device_chunk_size"],
            d["ttnn_attention_block_output"],
            attention_block_semaphores=self._state["attn_semaphores"],
            shared_residual_mcast_src_tensor=d["ttnn_residual_mcast_src"],
            gate_mm_weights_tensor=None,
            gate_bias_tensor=None,
            gate_indices_tensor=None,
            gate_output_scores_tensor=None,
            gate_output_indices_tensor=None,
            gate_proj_weights_tensor=d["gate_proj_weights"],
            up_proj_weights_tensor=d["up_proj_weights"],
            down_proj_weights_tensor=d["down_proj_weights"],
            moe_final_output_tensor=None,
            rmsnorm_gamma_tensor=d["ffn_norm_overlapped"],
            shared_gate_weights_overlapped=d["shared_gate_weights_overlapped"],
            shared_up_weights_overlapped=d["shared_up_weights_overlapped"],
            shared_down_weights_tensor=d["shared_down_weights_tensor"],
            shared_k_parallel=d["shared_k_parallel"],
            shared_n_parallel=d["shared_n_parallel"],
            moe_semaphores=self._state["moe_semaphores"],
            reduce_intermediate_tensors=d["reduce_intermediate_tensors"],
            reduce_output_tensor=d["reduce_output_tensor"],
            reduce_semaphores=self._state["reduce_semaphores"],
            reduce_root_coord=self._state["reduce_root_coord"],
            enable_routing=False,
            use_hardcoded_expert_index=False,
            bcast_cluster_axis=0,
            bcast_secondary_cluster_axis=1,
            reduce_cluster_axis=1,
            sdpa_cluster_axis=0,
            sdpa_scale_fp32=d["scale"],
            num_links=1,
            skip_ccl=False,
            upstream_socket=self._state["recv_socket"],
            downstream_socket=self._state["downstream_socket"],
            persistent_next_iter_semaphore=self._state.get("persistent_next_iter_semaphore"),
            persistent_mode=self._persistent_mode,
            is_torus=self._is_torus,
        )
