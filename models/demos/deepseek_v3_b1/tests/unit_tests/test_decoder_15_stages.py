# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Two-stage pipeline integration test for socket-fed bcast + DecoderBlock fused op + reduce-to-one.

Stage 0:
  host token -> fused embedding (HostInterface) -> cross-stage D2D
  D2H loopback: read aggregated reduce output from pipeline
Stage 1:
  entry D2D receiver -> moe sender core socket input -> bcast + fused DecoderBlock
  -> reduce-to-one -> D2D_0 aggregator -> pipeline exit
Stage 2+ (if applicable):
  passive forwarding, no downstream op
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.stage import StageContext, StageKind
from models.demos.deepseek_v3_b1.fused_ops.attention_block.op import AttentionBlock
from models.demos.deepseek_v3_b1.fused_ops.decoder_block.op import DecoderBlock
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.tests.unit_tests.test_decoder_block import create_decoder_block_tensors
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import (
    ROUTED_EXPERT_LAYER_IDX,
    RoutedExpert,
    create_routed_expert_tensors,
    create_shared_expert_tensors,
)


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def build_worker_grid_excluding_core(device_grid_size, excluded_core):
    max_x = device_grid_size.x - 1
    max_y = device_grid_size.y - 1
    ranges = []

    if excluded_core.y > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max_x, excluded_core.y - 1)))
    if excluded_core.y < max_y:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, excluded_core.y + 1), ttnn.CoreCoord(max_x, max_y)))
    if excluded_core.x > 0:
        ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(0, excluded_core.y), ttnn.CoreCoord(excluded_core.x - 1, excluded_core.y))
        )
    if excluded_core.x < max_x:
        ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(excluded_core.x + 1, excluded_core.y), ttnn.CoreCoord(max_x, excluded_core.y))
        )

    return ttnn.CoreRangeSet(ranges)


class DecoderBlockStage(StageKind):
    """Decoder block compute stage: bcast + fused attention + MoE + reduce-to-one."""

    PIPELINE_CORE = ttnn.CoreCoord(12, 8)
    MOE_SENDER_CORE = ttnn.CoreCoord(12, 9)
    M = 1
    K = 7168
    EMBEDDING_SIZE_BYTES = K * 2  # bfloat16
    EMBEDDING_FIFO_SIZE = EMBEDDING_SIZE_BYTES * 2
    TOKEN_SIZE_BYTES = 64

    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        layer_idx: int = 4,
        num_routed_experts: int = 8,
        position_id: int = 0,
        max_seq_len: int = 32 * 1024,
        persistent_mode: bool = True,
        use_hardcoded_expert_index: bool = True,
        enable_routing: bool = True,
        is_torus: bool = True,
    ) -> None:
        self._state_dict = state_dict
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
            is_moe=True,
            num_routed_experts=self._num_routed_experts,
            validate_debug_tensors=False,
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


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_router_config": create_fabric_router_config(15232),
            "worker_l1_size": 1439536,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_dim", [7168])
@pytest.mark.parametrize("iterations", [8192])
@pytest.mark.timeout(120000)
def test_persistent_decoder_15_stages(
    mesh_device, embedding_dim, iterations, device_params, get_reference_model_state_dict
):
    """
    Persistent-mode 15-stage decoder block pipeline test.

    Pipeline topology:
      Stage 0  : H2D embedding -> downstream D2D, D2H loopback <- pipeline
      Stage 1-14 (15 decoder stages): socket bcast -> fused DecoderBlock -> reduce-to-one -> pipeline exit
      Stage 15 : passive forwarding back to stage 0

    The decoder kernel on stages 1-14 runs in a while(true) loop.  Stage 0 drives
    the pipeline by writing *iterations* tokens and reading back each D2H result.
    Validates every result is non-zero.
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs < 2:
        pytest.skip(f"Requires at least 2 distributed processes, got {num_procs}")

    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for MoE (need >= 13x10)")

    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
    assert len(pipeline_config) == num_procs + 1

    is_torus = device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_2D_TORUS_Y

    is_stage0 = my_mesh_id == 0
    is_moe_stage = my_mesh_id >= 1

    K = embedding_dim
    pipeline_core = DecoderBlockStage.PIPELINE_CORE
    token_size_bytes = DecoderBlockStage.TOKEN_SIZE_BYTES
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2

    layer_idx = 4
    num_routed_experts = 8

    torch.manual_seed(42)
    print("Create torch embedding")
    torch_embedding = torch.randn(iterations, 1, 1, K, dtype=torch.bfloat16)
    print("Torch embedding created")

    state_dict = get_reference_model_state_dict(
        layer_idx=layer_idx,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=num_routed_experts,
    )

    ctx = StageContext(mesh_device=mesh_device, pipeline_config=pipeline_config, my_mesh_id=my_mesh_id)
    decoder_stage = None

    pipeline_block = None
    try:
        # ── Pipeline block setup (collective — all hosts must participate simultaneously) ────
        if is_stage0:
            print("Write embedding to tensor")
            embedding_tensor = ttnn.from_torch(
                torch_embedding,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            print("Embedding tensor written to device")
            pipeline_block = PipelineBlock(
                mesh_device,
                pipeline_core,
                upstream_d2d_socket_fifo_size=embedding_fifo_size,
                downstream_d2d_socket_fifo_size=embedding_fifo_size,
                upstream_d2d_socket_page_size=embedding_size_bytes,
                downstream_d2d_socket_page_size=embedding_size_bytes,
                h2d_socket_fifo_size=token_size_bytes * 2,
                d2h_socket_fifo_size=embedding_fifo_size,
                d2h_socket_page_size=embedding_size_bytes,
                embedding_tensor=embedding_tensor,
            )
        else:
            decoder_stage = DecoderBlockStage(
                state_dict,
                layer_idx=layer_idx,
                num_routed_experts=num_routed_experts,
                persistent_mode=True,
                use_hardcoded_expert_index=True,
                enable_routing=True,
                is_torus=is_torus,
            )
            pipeline_block = decoder_stage.create_pipeline_block(ctx)
            decoder_stage.setup(ctx, pipeline_block)

        logger.info(f"[rank={my_mesh_id}] pipeline block created")

        # ── Launch pipeline ──
        pipeline_block.run()
        logger.info(f"[rank={my_mesh_id}] pipeline launched")

        # ── Decoder stages: submit persistent kernel ──
        if is_moe_stage:
            logger.info(f"[rank={my_mesh_id}] submitting persistent decoder kernel")
            decoder_stage.launch_compute(ctx, pipeline_block)
            logger.info(f"[rank={my_mesh_id}] persistent decoder kernel submitted")
        ttnn.distributed_context_barrier()

        # ── Stage 0: drive pipeline with multiple tokens ──
        if is_stage0:
            token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
            num_elements = embedding_size_bytes // 2
            torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
            torch_token[0, 0] = 0
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            start_time = time.time()
            for iteration in range(iterations):
                pipeline_block.write_token(token_tensor)
                logger.info(f"[rank=0] token {iteration} injected")

                d2h_output_tensor = ttnn.from_torch(
                    torch.zeros(1, num_elements, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                print(f"[rank={my_mesh_id}] iteration {iteration} waiting for D2H result")
                pipeline_block.read_output(d2h_output_tensor)
                print(f"[rank={my_mesh_id}] iteration {iteration} D2H result read")
                d2h_result = ttnn.to_torch(d2h_output_tensor)

                d2h_nonzero = torch.count_nonzero(d2h_result)
                logger.info(
                    f"[rank={my_mesh_id}] iteration {iteration}: non-zero={d2h_nonzero}/{d2h_result.numel()}, "
                    f"first 5={d2h_result[0, :5]}"
                )
                assert (
                    d2h_nonzero > 0
                ), f"D2H output is all zeros at iteration {iteration} — persistent MoE 15-stage pipeline failed"
            end_time = time.time()
            print(f"[rank=0] time taken to move {iterations} tokens: {end_time - start_time} seconds")

            logger.info(f"[rank={my_mesh_id}] all {iterations} iterations passed")

        logger.info(f"[rank={my_mesh_id}] waiting for barrier")
        ttnn.distributed_context_barrier()
        logger.info(f"[rank={my_mesh_id}] barrier completed")

    finally:
        pass

    logger.info(f"[rank={my_mesh_id}] persistent 15-stage MoE test PASSED")


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_dim", [7168])
@pytest.mark.parametrize("iterations", [4000])
@pytest.mark.timeout(120000)
def test_persistent_moe_multi_token(
    mesh_device, embedding_dim, iterations, device_params, get_reference_model_state_dict
):
    """
    Persistent-mode 15-stage MoE pipeline test.

    Pipeline topology:
      Stage 0  : H2D embedding → downstream D2D, D2H loopback ← pipeline
      Stage 1-14 (15 MoE stages): socket bcast → fused MoE → reduce-to-one → pipeline exit
      Stage 15 : passive forwarding back to stage 0

    The MoE kernel on stages 1-14 runs in a while(true) loop.  Stage 0 drives
    the pipeline by writing *iterations* tokens and reading back each D2H result.
    Validates every result is non-zero.
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs < 2:
        pytest.skip(f"Requires at least 2 distributed processes, got {num_procs}")

    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for MoE (need >= 13x10)")

    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
    assert len(pipeline_config) == num_procs + 1

    is_torus = device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_2D_TORUS_Y

    is_stage0 = my_mesh_id == 0
    is_moe_stage = my_mesh_id >= 1

    M = 1
    K = embedding_dim
    pipeline_core = ttnn.CoreCoord(12, 8)
    moe_sender_core = ttnn.CoreCoord(12, 9)
    moe_worker_core_grid = build_worker_grid_excluding_core(device_grid, pipeline_core)

    token_size_bytes = 64
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2

    torch.manual_seed(42)
    print("Create torch embedding")
    torch_embedding = torch.randn(iterations, 1, 1, K, dtype=torch.bfloat16)
    print("Torch embedding created")

    reduce_root_coord = ttnn.MeshCoordinate(0, 0)

    stage_entry_device = None
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
    shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
    aggregator_core = shard_cores_list[0]

    if is_moe_stage:
        stage_entry_device = pipeline_config[my_mesh_id].entry_node_coord
        reduce_root_coord = pipeline_config[my_mesh_id].exit_node_coord
        logger.info(f"[rank={my_mesh_id}] stage entry device: {stage_entry_device}")
        logger.info(f"[rank={my_mesh_id}] reduce aggregator core: {aggregator_core}")

    # ── Pipeline block setup (collective) ──
    pipeline_block = None
    try:
        # ── Pipeline block setup (collective — all hosts must participate simultaneously) ────
        if is_stage0:
            print("Write embedding to tensor")
            embedding_tensor = ttnn.from_torch(
                torch_embedding,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            print("Embedding tensor written to device")
            pipeline_block = PipelineBlock(
                mesh_device,
                pipeline_core,
                upstream_d2d_socket_fifo_size=embedding_fifo_size,
                downstream_d2d_socket_fifo_size=embedding_fifo_size,
                upstream_d2d_socket_page_size=embedding_size_bytes,
                downstream_d2d_socket_page_size=embedding_size_bytes,
                h2d_socket_fifo_size=token_size_bytes * 2,
                d2h_socket_fifo_size=embedding_fifo_size,
                d2h_socket_page_size=embedding_size_bytes,
                embedding_tensor=embedding_tensor,
            )
        else:
            pipeline_block = PipelineBlock(
                mesh_device,
                pipeline_core,
                upstream_d2d_socket_fifo_size=embedding_fifo_size,
                downstream_d2d_socket_fifo_size=embedding_fifo_size,
                upstream_d2d_socket_page_size=embedding_size_bytes,
                downstream_d2d_socket_page_size=embedding_size_bytes,
                entry_node_downstream=ttnn.MeshCoreCoord(stage_entry_device, moe_sender_core),
                exit_node_upstream=ttnn.MeshCoreCoord(reduce_root_coord, aggregator_core),
            )

        logger.info(f"[rank={my_mesh_id}] pipeline block created")

        downstream_socket = None
        if my_mesh_id >= 1:
            downstream_socket = pipeline_block.exit_socket_interface.get_upstream_socket()

        # ── MoE tensors (MoE stages only) ──
        state_dict = get_reference_model_state_dict(
            layer_idx=ROUTED_EXPERT_LAYER_IDX,
            is_moe=True,
            seed=RoutedExpert.SEED,
            num_routed_experts=256,
            include_global=False,
        )

        r = None
        s = None
        if is_moe_stage:
            mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            r = create_routed_expert_tensors(
                mesh_device,
                use_hardcoded_expert_index=True,
                mesh_mapper=mesh_mapper,
                state_dict=state_dict,
                is_moe=True,
                layer_idx=ROUTED_EXPERT_LAYER_IDX,
            )
            mcast_grid = moe_worker_core_grid
            s = create_shared_expert_tensors(
                mesh_device,
                M,
                K,
                mcast_grid,
                mesh_mapper=mesh_mapper,
                state_dict=state_dict,
                is_moe=True,
                layer_idx=ROUTED_EXPERT_LAYER_IDX,
            )
            logger.info(f"[rank={my_mesh_id}] MoE tensors created")

            kv_cache_shard_height = 256
            kvpe_dim = 576
            num_mcast_cores = mcast_grid.num_cores()
            kv_cache_shard_spec = ttnn.ShardSpec(
                mcast_grid, (kv_cache_shard_height, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR
            )
            sdpa_kv_cache_buffer = ttnn.from_torch(
                torch.zeros((kv_cache_shard_height * num_mcast_cores, kvpe_dim), dtype=torch.bfloat16),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
                ),
            )
            sdpa_out_interm_shard_height = 40
            sdpa_out_interm_shard_width = 544
            num_worker_cores = moe_worker_core_grid.num_cores()
            sdpa_out_interm_shard_spec = ttnn.ShardSpec(
                moe_worker_core_grid,
                (sdpa_out_interm_shard_height, sdpa_out_interm_shard_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            sdpa_out_interm_buffer = ttnn.from_torch(
                torch.zeros(
                    (sdpa_out_interm_shard_height * num_worker_cores, sdpa_out_interm_shard_width),
                    dtype=torch.bfloat16,
                ),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_out_interm_shard_spec
                ),
                tile=ttnn.Tile([8, 32]),
            )

            tile_1x32 = ttnn.Tile([1, 32])
            final_output_total_width = r.final_output_total_width

            reduce_mesh_mapper_config = ttnn.MeshMapperConfig(
                [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape
            )
            reduce_mesh_mapper = ttnn.create_mesh_mapper(mesh_device, reduce_mesh_mapper_config)

            final_output_mem_config = r.final_output_mem_config
            # Single intermediate tensor with 3x shard width for all 3 reduction rounds
            intermediate_shard_spec = final_output_mem_config.shard_spec
            intermediate_shard_shape = [intermediate_shard_spec.shape[0], intermediate_shard_spec.shape[1] * 3]
            intermediate_mem_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.types.BufferType.L1,
                ttnn.ShardSpec(
                    intermediate_shard_spec.grid,
                    intermediate_shard_shape,
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            reduce_intermediate_tensors = ttnn.from_torch(
                torch.zeros([4, 2, final_output_total_width * 3], dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=intermediate_mem_config,
                tile=tile_1x32,
                mesh_mapper=reduce_mesh_mapper,
            )

            compute_grid = mesh_device.compute_with_storage_grid_size()
            reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(aggregator_core, aggregator_core)})
            reduce_output_shard_spec = ttnn.ShardSpec(
                reduce_output_shard_grid,
                (1, final_output_total_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            reduce_output_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, reduce_output_shard_spec
            )
            reduce_output_data = torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16)
            reduce_output_tensor = ttnn.from_torch(
                reduce_output_data,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=reduce_output_mem_config,
                tile=tile_1x32,
                mesh_mapper=reduce_mesh_mapper,
            )

            num_cores = compute_grid.x * compute_grid.y
            reduce_available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
            reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, reduce_available_cores, 0) for _ in range(4)]

            available_cores = moe_worker_core_grid
            bcast_semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(3)]
            moe_semaphores = MoeOp.create_semaphores(mesh_device)

            device_grid_size = mesh_device.compute_with_storage_grid_size()
            worker_crs = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
            )

            persistent_next_iter_semaphore = ttnn.create_global_semaphore(mesh_device, worker_crs, 1)

            input_core_grid = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid
            bcast_shard_spec = ttnn.ShardSpec(input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR)
            bcast_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, bcast_shard_spec
            )
            bcast_input_tensor = ttnn.from_torch(
                torch.zeros((M, K), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=bcast_mem_config,
                tile=tile_1x32,
                mesh_mapper=mesh_mapper,
            )
            bcast_intermediate_tensor = r.ttnn_residual_mcast_src

            recv_socket = pipeline_block.get_downstream_socket()
            bcast_sender_coord = stage_entry_device
            logger.info(f"[rank={my_mesh_id}] MoE setup complete")

        # ── Launch pipeline ──
        pipeline_block.run()
        logger.info(f"[rank={my_mesh_id}] pipeline launched")

        # ── MoE stages: submit persistent kernel ──
        if my_mesh_id >= 1:
            logger.info(f"[rank={my_mesh_id}] submitting persistent MoE kernel")
            MoeOp.op(
                r.ttnn_residual_mcast_src,
                gate_mm_weights_tensor=r.ttnn_gate_mm_weights,
                gate_bias_tensor=r.ttnn_gate_bias,
                gate_indices_tensor=r.ttnn_gate_indices,
                gate_output_scores_tensor=r.gate_output_scores_tensor,
                gate_output_indices_tensor=r.gate_output_indices_tensor,
                gate_proj_weights_tensor=r.gate_proj_weights,
                up_proj_weights_tensor=r.up_proj_weights,
                down_proj_weights_tensor=r.down_proj_weights,
                final_output_tensor=r.final_output_tensor,
                rmsnorm_gamma_tensor=r.ttnn_rmsnorm_gamma,
                shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
                shared_up_weights_overlapped=s.shared_up_weights_overlapped,
                shared_down_weights_tensor=s.ttnn_down_weights,
                shared_k_parallel=s.k_parallel,
                shared_n_parallel=s.n_parallel,
                use_hardcoded_expert_index=True,
                sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
                sdpa_out_interm_buffer=sdpa_out_interm_buffer,
                num_iterations=1,
                persistent_mode=True,
                persistent_next_iter_semaphore=persistent_next_iter_semaphore,
                reduce_intermediate_tensors=reduce_intermediate_tensors,
                reduce_output_tensor=reduce_output_tensor,
                reduce_semaphores=reduce_semaphores,
                reduce_root_coord=reduce_root_coord,
                bcast_input_tensor=bcast_input_tensor,
                bcast_intermediate_tensor=bcast_intermediate_tensor,
                bcast_semaphores=bcast_semaphores,
                bcast_sender_coord=bcast_sender_coord,
                socket=recv_socket,
                semaphores=moe_semaphores,
                worker_core_grid=moe_worker_core_grid,
                is_torus=is_torus,
                downstream_socket=downstream_socket,
            )
            logger.info(f"[rank={my_mesh_id}] persistent MoE kernel submitted")
        ttnn.distributed_context_barrier()

        # ── Stage 0: drive pipeline with multiple tokens ──
        if is_stage0:
            token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
            num_elements = embedding_size_bytes // 2
            torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
            torch_token[0, 0] = 0
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            start_time = time.time()
            num_tokens_in_flight = 64
            for iteration in range(num_tokens_in_flight):
                pipeline_block.write_token(token_tensor)
                logger.info(f"[rank=0] token {iteration} injected")
            for iter_ in range(num_tokens_in_flight):
                d2h_output_tensor = ttnn.from_torch(
                    torch.zeros(1, num_elements, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                print(f"[rank={my_mesh_id}] iteration {iter_} waiting for D2H result")
                pipeline_block.read_output(d2h_output_tensor)
                print(f"[rank={my_mesh_id}] iteration {iter_} D2H result read")
                d2h_result = ttnn.to_torch(d2h_output_tensor)

                d2h_nonzero = torch.count_nonzero(d2h_result)
                logger.info(
                    f"[rank={my_mesh_id}] iteration {iter_}: non-zero={d2h_nonzero}/{d2h_result.numel()}, "
                    f"first 5={d2h_result[0, :5]}"
                )
                assert (
                    d2h_nonzero > 0
                ), f"D2H output is all zeros at iteration {iter_} — persistent MoE 15-stage pipeline failed"
            end_time = time.time()
            print(f"[rank=0] time taken to move {num_tokens_in_flight} tokens: {end_time - start_time} seconds")

            logger.info(f"[rank={my_mesh_id}] all {num_tokens_in_flight} iterations passed")

        logger.info(f"[rank={my_mesh_id}] waiting for barrier")
        ttnn.distributed_context_barrier()
        logger.info(f"[rank={my_mesh_id}] barrier completed")

    finally:
        pass

    logger.info(f"[rank={my_mesh_id}] persistent 15-stage MoE test PASSED")
