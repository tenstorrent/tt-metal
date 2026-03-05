# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Two-stage pipeline integration test for socket-fed bcast + MoE fused op + reduce-to-one.

Stage 0:
  host token -> fused embedding (HostInterface) -> cross-stage D2D
  D2H loopback: read aggregated reduce output from pipeline
Stage 1:
  entry D2D receiver -> moe sender core socket input -> bcast + fused MoE
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
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.demo.stage import StageContext, StageKind
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3MoELayerWeights,
    MoERoutedExpertWeights,
    create_gate_indices_tensor,
    prepare_attention_weights,
    prepare_routed_expert_weights,
    prepare_shared_expert_weights,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import ROUTED_EXPERT_LAYER_IDX, RoutedExpert


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


def _create_moe_weights(
    mesh_device: ttnn.MeshDevice,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    num_routed_experts: int = 256,
) -> DeepSeekV3MoELayerWeights:
    """Build DeepSeekV3MoELayerWeights from a state dict with tensors moved to device."""
    bdw = BlitzDecodeWeights(mesh_device)
    attn = prepare_attention_weights(bdw, state_dict, layer_idx, is_moe=True, move_to_device=True)
    shared = prepare_shared_expert_weights(bdw, state_dict, layer_idx, is_moe=True, move_to_device=True)
    routed = prepare_routed_expert_weights(
        bdw, state_dict, layer_idx, is_moe=True, num_routed_experts=num_routed_experts, move_to_device=True
    )
    assert isinstance(routed, MoERoutedExpertWeights)
    return DeepSeekV3MoELayerWeights(
        q_a_proj=attn.q_a_proj,
        q_b_proj=attn.q_b_proj,
        kv_a_proj=attn.kv_a_proj,
        o_proj=attn.o_proj,
        gate_mm=attn.gate_mm,
        attn_norm=attn.attn_norm,
        q_norm=attn.q_norm,
        kv_norm=attn.kv_norm,
        ffn_norm=attn.ffn_norm,
        gate_bias=attn.gate_bias,
        kv_b1_proj=attn.kv_b1_proj,
        kv_b2_proj=attn.kv_b2_proj,
        shared_gate_proj=shared.shared_gate_proj,
        shared_up_proj=shared.shared_up_proj,
        shared_down_proj=shared.shared_down_proj,
        routed_gate_proj=routed.routed_gate_proj,
        routed_up_proj=routed.routed_up_proj,
        routed_down_proj=routed.routed_down_proj,
    )


class MoEComputeStage(StageKind):
    """MoE compute stage: bcast + fused MoE + reduce-to-one."""

    PIPELINE_CORE = ttnn.CoreCoord(12, 8)
    MOE_SENDER_CORE = ttnn.CoreCoord(12, 9)
    M = 1
    K = 7168
    EMBEDDING_SIZE_BYTES = K * 2  # bfloat16
    EMBEDDING_FIFO_SIZE = EMBEDDING_SIZE_BYTES * 2
    TOKEN_SIZE_BYTES = 64
    FINAL_OUTPUT_WIDTH_PER_CORE = 32 * 32  # 1024
    SHARED_K_PARALLEL = 8
    SHARED_N_PARALLEL = 8
    KV_CACHE_SHARD_HEIGHT = 256
    KVPE_DIM = 576
    SDPA_OUT_INTERM_SHARD_HEIGHT = 40
    SDPA_OUT_INTERM_SHARD_WIDTH = 544

    def __init__(
        self,
        weights: DeepSeekV3MoELayerWeights,
        *,
        persistent_mode: bool = True,
        use_hardcoded_expert_index: bool = True,
        is_torus: bool = True,
    ) -> None:
        self._weights = weights
        self._persistent_mode = persistent_mode
        self._use_hardcoded_expert_index = use_hardcoded_expert_index
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
            h2d_socket_fifo_size=self.TOKEN_SIZE_BYTES * 2,
            d2h_socket_fifo_size=self.EMBEDDING_FIFO_SIZE,
            d2h_socket_page_size=self.EMBEDDING_SIZE_BYTES,
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        mesh_device = ctx.mesh_device
        pipeline_config = ctx.pipeline_config
        my_mesh_id = ctx.my_mesh_id
        device_grid = mesh_device.compute_with_storage_grid_size()

        moe_worker_core_grid = build_worker_grid_excluding_core(device_grid, self.PIPELINE_CORE)
        input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(self.MOE_SENDER_CORE, self.MOE_SENDER_CORE)])
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        tile_1x32 = ttnn.Tile([1, 32])

        stage_entry_device = pipeline_config[my_mesh_id].entry_node_coord
        reduce_root_coord = pipeline_config[my_mesh_id].exit_node_coord

        # Gate indices tensor
        gate_indices_tensor = create_gate_indices_tensor(mesh_device, input_core_grid, mesh_mapper=mesh_mapper)

        # Gate output buffers (scores and indices on sender core)
        tile_1x16 = ttnn.Tile((1, 16))
        gate_output_shard_spec = ttnn.ShardSpec(input_core_grid, (1, 16), ttnn.ShardOrientation.ROW_MAJOR)
        gate_output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gate_output_shard_spec
        )
        gate_output_scores_tensor = ttnn.from_torch(
            torch.zeros((1, 16), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=gate_output_mem_config,
            tile=tile_1x16,
            mesh_mapper=mesh_mapper,
        )
        gate_output_indices_tensor = ttnn.from_torch(
            torch.zeros((1, 16), dtype=torch.uint16),
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=gate_output_mem_config,
            tile=tile_1x16,
            mesh_mapper=mesh_mapper,
        )

        # Residual / bcast intermediate tensor (input to MoE, on sender core)
        residual_shard_spec = ttnn.ShardSpec(input_core_grid, (self.M, self.K), ttnn.ShardOrientation.ROW_MAJOR)
        residual_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, residual_shard_spec
        )
        torch.manual_seed(RoutedExpert.SEED)
        torch_input = torch.randn((self.M, self.K), dtype=torch.bfloat16)
        residual_mcast_src = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=residual_mem_config,
            tile=tile_1x32,
            mesh_mapper=mesh_mapper,
        )

        # Final output tensor
        gate_proj_noc = ttnn.NOC.NOC_0
        gate_proj_worker_cores = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
        num_gate_proj_cores = len(gate_proj_worker_cores)
        gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
        final_output_total_width = self.FINAL_OUTPUT_WIDTH_PER_CORE * num_gate_proj_cores
        final_output_shard_spec = ttnn.ShardSpec(
            gate_proj_core_ranges, (1, self.FINAL_OUTPUT_WIDTH_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR
        )
        final_output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, final_output_shard_spec
        )
        final_output_tensor = ttnn.from_torch(
            torch.zeros([1, 1, 1, final_output_total_width]).bfloat16().float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=final_output_mem_config,
            tile=tile_1x32,
            mesh_mapper=mesh_mapper,
        )

        # SDPA buffers for CB memory overlap
        mcast_grid = moe_worker_core_grid
        num_mcast_cores = mcast_grid.num_cores()
        kv_cache_shard_spec = ttnn.ShardSpec(
            mcast_grid, (self.KV_CACHE_SHARD_HEIGHT, self.KVPE_DIM), ttnn.ShardOrientation.ROW_MAJOR
        )
        sdpa_kv_cache_buffer = ttnn.from_torch(
            torch.zeros((self.KV_CACHE_SHARD_HEIGHT * num_mcast_cores, self.KVPE_DIM), dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
            ),
        )
        num_worker_cores = moe_worker_core_grid.num_cores()
        sdpa_out_interm_shard_spec = ttnn.ShardSpec(
            moe_worker_core_grid,
            (self.SDPA_OUT_INTERM_SHARD_HEIGHT, self.SDPA_OUT_INTERM_SHARD_WIDTH),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sdpa_out_interm_buffer = ttnn.from_torch(
            torch.zeros(
                (self.SDPA_OUT_INTERM_SHARD_HEIGHT * num_worker_cores, self.SDPA_OUT_INTERM_SHARD_WIDTH),
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

        # Reduce-to-one tensors
        reduce_mesh_mapper_config = ttnn.MeshMapperConfig(
            [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape
        )
        reduce_mesh_mapper = ttnn.create_mesh_mapper(mesh_device, reduce_mesh_mapper_config)

        reduce_intermediate_tensors = []
        for _ in range(3):
            intermediate_data = torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16)
            intermediate_tensor = ttnn.from_torch(
                intermediate_data,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=final_output_mem_config,
                tile=tile_1x32,
                mesh_mapper=reduce_mesh_mapper,
            )
            reduce_intermediate_tensors.append(intermediate_tensor)

        compute_grid = mesh_device.compute_with_storage_grid_size()
        reduce_output_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
        reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(reduce_output_core, reduce_output_core)})
        reduce_output_shard_spec = ttnn.ShardSpec(
            reduce_output_shard_grid, (1, final_output_total_width), ttnn.ShardOrientation.ROW_MAJOR
        )
        reduce_output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, reduce_output_shard_spec
        )
        reduce_output_tensor = ttnn.from_torch(
            torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16),
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

        # Bcast + MoE semaphores
        bcast_semaphores = [ttnn.create_global_semaphore(mesh_device, moe_worker_core_grid, 0) for _ in range(3)]
        moe_semaphores = MoeOp.create_semaphores(mesh_device)

        # Bcast input tensor
        bcast_shard_spec = ttnn.ShardSpec(input_core_grid, (self.M, self.K), ttnn.ShardOrientation.ROW_MAJOR)
        bcast_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, bcast_shard_spec
        )
        bcast_input_tensor = ttnn.from_torch(
            torch.zeros((self.M, self.K), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=bcast_mem_config,
            tile=tile_1x32,
            mesh_mapper=mesh_mapper,
        )

        # Sockets
        recv_socket = pipeline_block.get_downstream_socket()
        downstream_socket = pipeline_block.get_upstream_socket()

        self._state = {
            "residual_mcast_src": residual_mcast_src,
            "gate_indices_tensor": gate_indices_tensor,
            "gate_output_scores_tensor": gate_output_scores_tensor,
            "gate_output_indices_tensor": gate_output_indices_tensor,
            "final_output_tensor": final_output_tensor,
            "sdpa_kv_cache_buffer": sdpa_kv_cache_buffer,
            "sdpa_out_interm_buffer": sdpa_out_interm_buffer,
            "reduce_intermediate_tensors": reduce_intermediate_tensors,
            "reduce_output_tensor": reduce_output_tensor,
            "reduce_semaphores": reduce_semaphores,
            "reduce_root_coord": reduce_root_coord,
            "bcast_input_tensor": bcast_input_tensor,
            "bcast_semaphores": bcast_semaphores,
            "bcast_sender_coord": stage_entry_device,
            "recv_socket": recv_socket,
            "moe_semaphores": moe_semaphores,
            "moe_worker_core_grid": moe_worker_core_grid,
            "downstream_socket": downstream_socket,
        }

        if self._persistent_mode:
            device_grid_size = mesh_device.compute_with_storage_grid_size()
            worker_crs = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
            )
            self._state["persistent_next_iter_semaphore"] = ttnn.create_global_semaphore(mesh_device, worker_crs, 1)

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        d = self._state
        w = self._weights
        MoeOp.op(
            d["residual_mcast_src"],
            gate_mm_weights_tensor=w.gate_mm,
            gate_bias_tensor=w.gate_bias,
            gate_indices_tensor=d["gate_indices_tensor"],
            gate_output_scores_tensor=d["gate_output_scores_tensor"],
            gate_output_indices_tensor=d["gate_output_indices_tensor"],
            gate_proj_weights_tensor=w.routed_gate_proj[0],
            up_proj_weights_tensor=w.routed_up_proj[0],
            down_proj_weights_tensor=w.routed_down_proj[0],
            final_output_tensor=d["final_output_tensor"],
            rmsnorm_gamma_tensor=w.ffn_norm,
            shared_gate_weights_overlapped=w.shared_gate_proj,
            shared_up_weights_overlapped=w.shared_up_proj,
            shared_down_weights_tensor=w.shared_down_proj,
            shared_k_parallel=self.SHARED_K_PARALLEL,
            shared_n_parallel=self.SHARED_N_PARALLEL,
            use_hardcoded_expert_index=self._use_hardcoded_expert_index,
            sdpa_kv_cache_buffer=d["sdpa_kv_cache_buffer"],
            sdpa_out_interm_buffer=d["sdpa_out_interm_buffer"],
            num_iterations=1,
            persistent_mode=self._persistent_mode,
            persistent_next_iter_semaphore=d.get("persistent_next_iter_semaphore"),
            reduce_intermediate_tensors=d["reduce_intermediate_tensors"],
            reduce_output_tensor=d["reduce_output_tensor"],
            reduce_semaphores=d["reduce_semaphores"],
            reduce_root_coord=d["reduce_root_coord"],
            bcast_input_tensor=d["bcast_input_tensor"],
            bcast_intermediate_tensor=d["residual_mcast_src"],
            bcast_semaphores=d["bcast_semaphores"],
            bcast_sender_coord=d["bcast_sender_coord"],
            socket=d["recv_socket"],
            semaphores=d["moe_semaphores"],
            worker_core_grid=d["moe_worker_core_grid"],
            is_torus=self._is_torus,
            downstream_socket=d["downstream_socket"],
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
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("vocab_size, embedding_dim", [(64, 7168)])
@pytest.mark.parametrize("token_id", [0])
@pytest.mark.timeout(1200)
def test_bcast_moe_two_stage_pipeline(
    mesh_device, vocab_size, embedding_dim, token_id, device_params, get_reference_model_state_dict
):
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

    K = embedding_dim
    pipeline_core = MoEComputeStage.PIPELINE_CORE
    token_size_bytes = MoEComputeStage.TOKEN_SIZE_BYTES
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2

    torch_embedding = torch.arange(vocab_size * K, dtype=torch.float32).reshape(1, 1, vocab_size, K).to(torch.bfloat16)

    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=256,
        include_global=False,
    )

    ctx = StageContext(mesh_device=mesh_device, pipeline_config=pipeline_config, my_mesh_id=my_mesh_id)
    moe_stage = None

    # ── Pipeline block setup (collective — all hosts must participate simultaneously) ────
    if is_stage0:
        embedding_tensor = ttnn.from_torch(torch_embedding, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
        weights = _create_moe_weights(mesh_device, state_dict, ROUTED_EXPERT_LAYER_IDX, num_routed_experts=256)
        moe_stage = MoEComputeStage(weights, persistent_mode=False, is_torus=is_torus)
        pipeline_block = moe_stage.create_pipeline_block(ctx)
        moe_stage.setup(ctx, pipeline_block)
        logger.info(f"[rank={my_mesh_id}] MoE stage setup complete")

    logger.info(f"[rank={my_mesh_id}] pipeline block created")

    # ── Launch pipeline programs ──────────────────────────────────────────────
    pipeline_block.run()
    logger.info(f"[rank={my_mesh_id}] pipeline programs launched")

    ttnn.distributed_context_barrier()

    if is_stage0:
        token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
        torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
        torch_token[0, 0] = token_id
        token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pipeline_block.write_token(token_tensor)
        logger.info(f"[rank=0] token {token_id} injected")

    ttnn.distributed_context_barrier()

    if my_mesh_id >= 1:
        logger.info(f"[rank={my_mesh_id}] launching MoE bcast + reduce (num_iterations=1)")
        moe_stage.launch_compute(ctx, pipeline_block)
        logger.info(f"[rank={my_mesh_id}] MoE + reduce completed")

    # ── Stage 0: D2H loopback read + golden validation ───────────────────────
    if is_stage0:
        logger.info(f"[rank=0] waiting for D2H result from pipeline loopback")
        num_elements = embedding_size_bytes // 2
        received_tensor_torch = torch.zeros(1, num_elements, dtype=torch.bfloat16)
        d2h_output_tensor = ttnn.from_torch(received_tensor_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        pipeline_block.read_output(d2h_output_tensor)
        d2h_result_torch = ttnn.to_torch(d2h_output_tensor)
        logger.info(f"[rank=0] D2H read complete, shape={d2h_result_torch.shape}")
        logger.info(f"[rank=0] D2H first 5 values: {d2h_result_torch[0, :5]}")

        d2h_nonzero = torch.count_nonzero(d2h_result_torch)
        logger.info(f"[rank=0] D2H non-zero elements: {d2h_nonzero}/{d2h_result_torch.numel()}")
        assert d2h_nonzero > 0, "D2H output is all zeros — reduce or D2D0 pipeline failed"

    ttnn.distributed_context_barrier()

    # ── Pipeline teardown ───────────────────────────────────────────────────
    logger.info(f"[rank={my_mesh_id}] waiting for pipeline block termination")
    pipeline_block.terminate()
    logger.info(f"[rank={my_mesh_id}] programs terminated")

    logger.info(f"[rank={my_mesh_id}] test PASSED")


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
def test_persistent_moe_15_stages(
    mesh_device, embedding_dim, iterations, device_params, get_reference_model_state_dict
):
    """
    Persistent-mode 15-stage MoE pipeline test.

    Pipeline topology:
      Stage 0  : H2D embedding -> downstream D2D, D2H loopback <- pipeline
      Stage 1-14 (15 MoE stages): socket bcast -> fused MoE -> reduce-to-one -> pipeline exit
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

    K = embedding_dim
    pipeline_core = MoEComputeStage.PIPELINE_CORE
    token_size_bytes = MoEComputeStage.TOKEN_SIZE_BYTES
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2

    torch.manual_seed(42)
    torch_embedding = torch.randn(iterations, 1, 1, K, dtype=torch.bfloat16)

    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=256,
        include_global=False,
    )

    ctx = StageContext(mesh_device=mesh_device, pipeline_config=pipeline_config, my_mesh_id=my_mesh_id)
    moe_stage = None

    pipeline_block = None
    try:
        # ── Pipeline block setup (collective — all hosts must participate simultaneously) ────
        if is_stage0:
            embedding_tensor = ttnn.from_torch(torch_embedding, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
            weights = _create_moe_weights(mesh_device, state_dict, ROUTED_EXPERT_LAYER_IDX, num_routed_experts=256)
            moe_stage = MoEComputeStage(weights, persistent_mode=True, is_torus=is_torus)
            pipeline_block = moe_stage.create_pipeline_block(ctx)
            moe_stage.setup(ctx, pipeline_block)
            logger.info(f"[rank={my_mesh_id}] MoE stage setup complete")

        logger.info(f"[rank={my_mesh_id}] pipeline block created")

        # ── Launch pipeline ──
        pipeline_block.run()
        logger.info(f"[rank={my_mesh_id}] pipeline launched")

        # ── MoE stages: submit persistent kernel ──
        if is_moe_stage:
            logger.info(f"[rank={my_mesh_id}] submitting persistent MoE kernel")
            moe_stage.launch_compute(ctx, pipeline_block)
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
