# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage kinds for the pod pipeline: Embedding, LMHead, MoECompute, Passthrough.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.lm_head_sampling.op import LMHeadSampling
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    create_gate_indices_tensor,
)

# Global constants used by multiple stage kinds (and exported to pipeline/cli)
TOKEN_PAGE_SIZE_BYTES = 64
TOKEN_FIFO_SIZE = 1024
ACTIVATION_DIM = 7168
ACTIVATION_PAGE_SIZE_BYTES = ACTIVATION_DIM * 2
ACTIVATION_FIFO_SIZE = ACTIVATION_PAGE_SIZE_BYTES * 4
PIPELINE_CORE_COORD = ttnn.CoreCoord(12, 8)  # FYI: LM head was previously on (11, 0)


@dataclass
class StageContext:
    """Bundles arguments passed to StageKind methods."""

    mesh_device: ttnn.MeshDevice
    pipeline_config: list
    my_mesh_id: int


class StageKind(ABC):
    """Abstract stage kind: controls PipelineBlock creation, setup, and compute launch."""

    @abstractmethod
    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        """Create and return the PipelineBlock for this stage."""

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        """Post-creation setup (tensor allocation, etc). Default: no-op."""

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        """Launch compute kernels after pipeline_block.run(). Default: no-op."""


class EmbeddingStage(StageKind):
    """Stage 0: H2D + embedding lookup, forwards activation; loopback receives token."""

    def __init__(self, weights: DeepSeekV3EmbeddingLayerWeights) -> None:
        self._weights = weights

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=TOKEN_FIFO_SIZE,
            downstream_d2d_socket_fifo_size=ACTIVATION_FIFO_SIZE,
            upstream_d2d_socket_page_size=TOKEN_PAGE_SIZE_BYTES,
            downstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
            h2d_socket_fifo_size=TOKEN_FIFO_SIZE,
            d2h_socket_fifo_size=TOKEN_FIFO_SIZE,
            d2h_socket_page_size=TOKEN_PAGE_SIZE_BYTES,
            embedding_tensor=self._weights.embedding,
        )


class PassthroughPayload(Enum):
    ACTIVATION = "activation"
    TOKEN = "token"


class PassthroughStage(StageKind):
    """Forward-only stage: activation or token passthrough."""

    def __init__(self, payload: PassthroughPayload) -> None:
        self._payload = payload

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        if self._payload == PassthroughPayload.ACTIVATION:
            up_fifo = down_fifo = ACTIVATION_FIFO_SIZE
            up_page = down_page = ACTIVATION_PAGE_SIZE_BYTES
        else:
            up_fifo = down_fifo = TOKEN_FIFO_SIZE
            up_page = down_page = TOKEN_PAGE_SIZE_BYTES
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=up_fifo,
            downstream_d2d_socket_fifo_size=down_fifo,
            upstream_d2d_socket_page_size=up_page,
            downstream_d2d_socket_page_size=down_page,
            d2h_socket_fifo_size=down_fifo,
            d2h_socket_page_size=down_page,
        )


class MoEDecoderStage(StageKind):
    """Decoder stage that runs an MoE layer; activation in, activation out. Compute stubbed for now."""

    def __init__(self, weights: DeepSeekV3MoELayerWeights) -> None:
        self._weights = weights

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=ACTIVATION_FIFO_SIZE,
            downstream_d2d_socket_fifo_size=ACTIVATION_FIFO_SIZE,
            upstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
            downstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        pass

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        pass


class DenseDecoderStage(StageKind):
    """Decoder stage that runs a dense layer; activation in, activation out. Compute stubbed for now."""

    def __init__(self, weights: DeepSeekV3DenseLayerWeights) -> None:
        self._weights = weights

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=ACTIVATION_FIFO_SIZE,
            downstream_d2d_socket_fifo_size=ACTIVATION_FIFO_SIZE,
            upstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
            downstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        pass

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        pass


class MoEComputeStage(StageKind):
    """MoE compute stage: bcast + fused MoE + reduce-to-one."""

    PIPELINE_CORE = ttnn.CoreCoord(12, 8)
    MOE_SENDER_CORE = ttnn.CoreCoord(12, 9)
    M = 1
    K = 7168
    EMBEDDING_SIZE_BYTES = K * 2  # bfloat16
    EMBEDDING_FIFO_SIZE = EMBEDDING_SIZE_BYTES * 4
    TOKEN_SIZE_BYTES = 64
    FINAL_OUTPUT_WIDTH_PER_CORE = 32 * 32  # 1024
    SHARED_K_PARALLEL = 8
    SHARED_N_PARALLEL = 8
    KV_CACHE_SHARD_HEIGHT = 256
    KVPE_DIM = 576
    SDPA_OUT_INTERM_SHARD_HEIGHT = 40
    SDPA_OUT_INTERM_SHARD_WIDTH = 544

    @staticmethod
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
                ttnn.CoreRange(
                    ttnn.CoreCoord(excluded_core.x + 1, excluded_core.y), ttnn.CoreCoord(max_x, excluded_core.y)
                )
            )

        return ttnn.CoreRangeSet(ranges)

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

        moe_worker_core_grid = self.build_worker_grid_excluding_core(device_grid, self.PIPELINE_CORE)
        input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(self.MOE_SENDER_CORE, self.MOE_SENDER_CORE)])
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        tile_1x32 = ttnn.Tile([1, 32])

        stage_entry_device = pipeline_config[my_mesh_id].entry_node_coord
        reduce_root_coord = pipeline_config[my_mesh_id].exit_node_coord

        gate_indices_tensor = create_gate_indices_tensor(mesh_device, input_core_grid, mesh_mapper=mesh_mapper)

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

        residual_shard_spec = ttnn.ShardSpec(input_core_grid, (self.M, self.K), ttnn.ShardOrientation.ROW_MAJOR)
        residual_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, residual_shard_spec
        )
        residual_mcast_src = ttnn.from_torch(
            torch.zeros((self.M, self.K), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=residual_mem_config,
            tile=tile_1x32,
            mesh_mapper=mesh_mapper,
        )

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

        bcast_semaphores = [ttnn.create_global_semaphore(mesh_device, moe_worker_core_grid, 0) for _ in range(3)]
        moe_semaphores = MoeOp.create_semaphores(mesh_device)

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


class LMHeadStage(StageKind):
    """LMHead+Sampling stage: receive activation, run op, send token downstream."""

    # LMHead-stage-specific constants (tiles, core coords, matmul layout)
    M = 1
    K = ACTIVATION_DIM
    NUM_MATMUL_CORES = 101
    N_PER_CORE = 160
    N_TOTAL = NUM_MATMUL_CORES * N_PER_CORE
    A_TILE = ttnn.Tile([1, 32])
    B_TILE = ttnn.Tile([32, 32])
    OUT_TILE = ttnn.Tile([1, 32])
    ARGMAX_FINAL_CORE = ttnn.CoreCoord(0, 0)
    LMHEAD_INPUT_CORE = ttnn.CoreCoord(10, 9)
    """LMHead+Sampling stage: receive activation, run op, send token downstream."""

    def __init__(
        self,
        weights: DeepSeekV3LMHeadWeights,
        *,
        lm_head_fp32_dest_acc_en: bool = True,
        lm_head_persistent_mode: bool = True,
    ) -> None:
        self._weights = weights
        self._lm_head_fp32_dest_acc_en = lm_head_fp32_dest_acc_en
        self._lm_head_persistent_mode = lm_head_persistent_mode
        self._lmhead_state: dict[str, Any] = {}

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        my_mesh_id = ctx.my_mesh_id
        pipeline_config = ctx.pipeline_config
        lmhead_entry_core = ttnn.MeshCoreCoord(
            pipeline_config[my_mesh_id].entry_node_coord, LMHeadStage.LMHEAD_INPUT_CORE
        )
        lmhead_exit_core = ttnn.MeshCoreCoord(
            pipeline_config[my_mesh_id].exit_node_coord, LMHeadStage.ARGMAX_FINAL_CORE
        )
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=ACTIVATION_FIFO_SIZE,
            downstream_d2d_socket_fifo_size=TOKEN_FIFO_SIZE,
            upstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
            downstream_d2d_socket_page_size=TOKEN_PAGE_SIZE_BYTES,
            entry_node_downstream=lmhead_entry_core,
            exit_node_upstream=lmhead_exit_core,
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        mesh_device = ctx.mesh_device
        my_mesh_id = ctx.my_mesh_id
        pipeline_config = ctx.pipeline_config
        torch_a = torch.zeros((LMHeadStage.M, LMHeadStage.K), dtype=torch.bfloat16)

        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
        sender_coord = pipeline_config[my_mesh_id].entry_node_coord
        num_devices = mesh_rows * mesh_cols

        mcast_core_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(LMHeadStage.LMHEAD_INPUT_CORE, LMHeadStage.LMHEAD_INPUT_CORE)]
        )
        matmul_core_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
            ]
        )
        argmax_final_core_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(LMHeadStage.ARGMAX_FINAL_CORE, LMHeadStage.ARGMAX_FINAL_CORE)]
        )

        input_a_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(mcast_core_grid, (LMHeadStage.M, LMHeadStage.K), ttnn.ShardOrientation.ROW_MAJOR),
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(matmul_core_grid, (LMHeadStage.M, LMHeadStage.N_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR),
        )
        indices_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(matmul_core_grid, (LMHeadStage.M, LMHeadStage.N_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR),
        )
        output_index_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(argmax_final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
        )

        device_inputs = []
        device_intermediate = []
        for r in range(mesh_rows):
            for c in range(mesh_cols):
                if r == sender_coord[0] and c == sender_coord[1]:
                    device_inputs.append(torch_a)
                else:
                    device_inputs.append(torch.zeros_like(torch_a))
                device_intermediate.append(torch.zeros_like(torch_a))
        mesh_input = torch.cat(device_inputs, dim=0)
        mesh_intermediate = torch.cat(device_intermediate, dim=0)
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)

        input_tensor_mesh = ttnn.from_torch(
            mesh_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            tile=LMHeadStage.A_TILE,
            dtype=ttnn.bfloat16,
            memory_config=input_a_mem_config,
            mesh_mapper=mesh_mapper,
        )
        intermediate_tensor_mesh = ttnn.from_torch(
            mesh_intermediate,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            tile=LMHeadStage.A_TILE,
            dtype=ttnn.bfloat16,
            memory_config=input_a_mem_config,
            mesh_mapper=mesh_mapper,
        )
        ttnn_gamma = self._weights.final_norm
        ttnn_b = self._weights.lm_head
        torch_indices_flat = torch.arange(LMHeadStage.N_TOTAL, dtype=torch.int32).reshape(1, LMHeadStage.N_TOTAL)
        ttnn_indices = ttnn.from_torch(
            torch_indices_flat.repeat(num_devices, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=indices_mem_config,
            mesh_mapper=mesh_mapper,
        )
        ttnn_scores = ttnn.from_torch(
            torch.zeros((LMHeadStage.M, LMHeadStage.N_TOTAL), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=output_mem_config,
            tile=LMHeadStage.OUT_TILE,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn_output_index = ttnn.from_torch(
            torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=output_index_mem_config,
            mesh_mapper=mesh_mapper,
        )
        winner_page_bytes = 16
        scratch_shape_per_device = (
            1,
            ((mesh_rows + mesh_cols) * winner_page_bytes) // 4,
        )
        scratch_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                argmax_final_core_grid,
                scratch_shape_per_device,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        scratch_buffer = ttnn.from_torch(
            torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=scratch_mem_config,
            mesh_mapper=mesh_mapper,
        )
        lmhead_input_socket = pipeline_block.get_downstream_socket()
        lmhead_output_socket = pipeline_block.get_upstream_socket()

        device_grid_size = mesh_device.compute_with_storage_grid_size()
        worker_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1),
                )
            }
        )

        out_ready_semaphore = ttnn.create_global_semaphore(mesh_device, worker_crs, 0)
        barrier_semaphore = ttnn.create_global_semaphore(mesh_device, worker_crs, 0)
        secondary_sync_semaphore = ttnn.create_global_semaphore(mesh_device, worker_crs, 0)
        global_semaphore = ttnn.create_global_semaphore(mesh_device, argmax_final_core_grid, 0)
        global_stage2_semaphore = ttnn.create_global_semaphore(mesh_device, argmax_final_core_grid, 0)
        self._lmhead_state = {
            "input_tensor_mesh": input_tensor_mesh,
            "intermediate_tensor_mesh": intermediate_tensor_mesh,
            "ttnn_gamma": ttnn_gamma,
            "ttnn_b": ttnn_b,
            "ttnn_scores": ttnn_scores,
            "ttnn_indices": ttnn_indices,
            "ttnn_output_index": ttnn_output_index,
            "scratch_buffer": scratch_buffer,
            "lmhead_input_socket": lmhead_input_socket,
            "lmhead_output_socket": lmhead_output_socket,
            "out_ready_semaphore": out_ready_semaphore,
            "barrier_semaphore": barrier_semaphore,
            "secondary_sync_semaphore": secondary_sync_semaphore,
            "global_semaphore": global_semaphore,
            "global_stage2_semaphore": global_stage2_semaphore,
        }
        if self._lm_head_persistent_mode:
            persistent_next_iter_semaphore = ttnn.create_global_semaphore(mesh_device, worker_crs, 1)
            self._lmhead_state["persistent_next_iter_semaphore"] = persistent_next_iter_semaphore

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        d = self._lmhead_state
        pipeline_config = ctx.pipeline_config
        my_mesh_id = ctx.my_mesh_id

        print("my mesh id", my_mesh_id)
        print("input socket", d["lmhead_input_socket"])
        print("output socket", d["lmhead_output_socket"])
        print("persistent_next_iter_semaphore", d["persistent_next_iter_semaphore"])
        print("persistent_mode", self._lm_head_persistent_mode)
        print("sender_coord", pipeline_config[my_mesh_id].entry_node_coord)
        print("exit_node_coord", pipeline_config[my_mesh_id].exit_node_coord)

        LMHeadSampling.op(
            d["input_tensor_mesh"],
            d["intermediate_tensor_mesh"],
            d["ttnn_gamma"],
            d["ttnn_b"],
            d["ttnn_scores"],
            sender_coord=pipeline_config[my_mesh_id].entry_node_coord,
            indices_tensor=d["ttnn_indices"],
            output_index_tensor=d["ttnn_output_index"],
            argmax_final_core_coord=LMHeadStage.ARGMAX_FINAL_CORE,
            argmax_final_mesh_coord=pipeline_config[my_mesh_id].exit_node_coord,
            semaphores=[
                d["out_ready_semaphore"],
                d["barrier_semaphore"],
                d["secondary_sync_semaphore"],
            ],
            global_semaphore=d["global_semaphore"],
            global_stage2_semaphore=d["global_stage2_semaphore"],
            fabric_scratch_tensor=d["scratch_buffer"],
            fp32_dest_acc_en=self._lm_head_fp32_dest_acc_en,
            skip_ccl=False,
            socket_input=d["lmhead_input_socket"],
            socket_output=d["lmhead_output_socket"],
            persistent_mode=self._lm_head_persistent_mode,
            persistent_next_iter_semaphore=d.get("persistent_next_iter_semaphore"),
        )
