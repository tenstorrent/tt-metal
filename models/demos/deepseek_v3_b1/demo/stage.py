# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage kinds for the pod pipeline: Embedding, LMHead, Passthrough.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

import ttnn
from models.demos.deepseek_v3_b1.demo.weight_provider import LogicalModelDimensions
from models.demos.deepseek_v3_b1.fused_ops.lm_head_sampling.op import LMHeadSampling
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import MeshWrapper, SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MTPWeights,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import build_broadcast_test_inputs

# Global constants used by multiple stage kinds (and exported to pipeline/cli)
TOKEN_PAGE_SIZE_BYTES = 64
TOKEN_FIFO_SIZE = 1024
ACTIVATION_DIM = 7168

ACTIVATION_PAGE_SIZE_BYTES = ACTIVATION_DIM * 2
ACTIVATION_FIFO_SIZE = ACTIVATION_PAGE_SIZE_BYTES * 1
PIPELINE_CORE_COORD = ttnn.CoreCoord(11, 0)

# Embedding core coords for the combined SpecLMHead+Embedding stage (column 12, outside mcast grid)
EMBEDDING_H2D_CORE_COORD = ttnn.CoreCoord(12, 0)
EMBEDDING_D2H_CORE_COORD = ttnn.CoreCoord(12, 1)
ARGMAX_RELAY_CORE = ttnn.CoreCoord(12, 2)

# MTP constants
embedding_dim = 7168
mtp_output_dim = 7168
num_dram_banks = 8
METADATA_NUM_ELEMS = 32
mtp_n_per_core = mtp_output_dim // num_dram_banks
mtp_padded_dim = num_dram_banks * mtp_n_per_core

# Token metadata payload: just token info (id, type, pos) — same physical size as TOKEN.
TOKEN_META_PAGE_SIZE_BYTES = TOKEN_PAGE_SIZE_BYTES
TOKEN_META_FIFO_SIZE = TOKEN_FIFO_SIZE

# Activation + token metadata payload: logits + 1 metadata tile (token).
ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES = ACTIVATION_PAGE_SIZE_BYTES + TOKEN_PAGE_SIZE_BYTES
ACTIVATION_W_TOKEN_META_FIFO_SIZE = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES


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
        """Post-creation setup (tensor allocation, etc).

        Decoder stages may also compile/build device programs here so ``launch_compute`` only
        enqueues execution. Default: no-op.
        """

    def run_auxiliary_sockets(self) -> None:
        """Start auxiliary (bypass) d2d_exchange kernels. Default: no-op."""

    def terminate_auxiliary(self) -> None:
        """Terminate auxiliary sockets. Default: no-op."""

    def run_auxiliary_sockets(self) -> None:
        """Start auxiliary (bypass) d2d_exchange kernels. Default: no-op."""

    def terminate_auxiliary(self) -> None:
        """Terminate auxiliary sockets. Default: no-op."""

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        """Run stage compute after ``pipeline_block.run()`` (execute pre-built programs where applicable). Default: no-op."""


class EmbeddingStage(StageKind):
    """Stage 0: H2D + embedding lookup, forwards activation; loopback receives token."""

    def __init__(self, weights: DeepSeekV3EmbeddingLayerWeights, *, d2h_page_size: int | None = None) -> None:
        self._weights = weights

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        print(f"[STAGE P{ctx.my_mesh_id}] EmbeddingStage.create_pipeline_block", flush=True)
        mesh_device = ctx.mesh_device
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=TOKEN_FIFO_SIZE,
            downstream_d2d_socket_fifo_size=ACTIVATION_W_TOKEN_META_FIFO_SIZE,
            upstream_d2d_socket_page_size=TOKEN_PAGE_SIZE_BYTES,
            downstream_d2d_socket_page_size=ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES,
            h2d_socket_fifo_size=TOKEN_FIFO_SIZE,
            d2h_socket_fifo_size=TOKEN_FIFO_SIZE,
            d2h_socket_page_size=TOKEN_PAGE_SIZE_BYTES,
            embedding_tensor=self._weights.embedding,
        )


class PassthroughPayload(Enum):
    ACTIVATION = "activation"
    TOKEN = "token"
    TOKEN_META = "token_meta"
    ACTIVATION_W_TOKEN_META = "activation_w_token_meta"


class PassthroughStage(StageKind):
    """Forward-only stage: activation or token passthrough."""

    def __init__(self, payload: PassthroughPayload) -> None:
        self._payload = payload
        print(f"[STAGE] PassthroughStage.__init__ payload={payload.value}", flush=True)

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        print(f"[STAGE P{ctx.my_mesh_id}] PassthroughStage.create_pipeline_block", flush=True)
        mesh_device = ctx.mesh_device
        if self._payload == PassthroughPayload.ACTIVATION:
            up_fifo = down_fifo = ACTIVATION_FIFO_SIZE
            up_page = down_page = ACTIVATION_PAGE_SIZE_BYTES
        elif self._payload == PassthroughPayload.ACTIVATION_W_TOKEN_META:
            up_fifo = down_fifo = ACTIVATION_W_TOKEN_META_FIFO_SIZE
            up_page = down_page = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES
        elif self._payload == PassthroughPayload.TOKEN_META:
            up_fifo = down_fifo = TOKEN_META_FIFO_SIZE
            up_page = down_page = TOKEN_META_PAGE_SIZE_BYTES
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
        )


class SpecLMHeadStage(StageKind):
    """MTP LMHead+Sampling+Verification stage: receives base token, runs its own LM head,
    then verifies its speculative token against the base token."""

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

    def __init__(
        self,
        weights: DeepSeekV3LMHeadWeights,
        *,
        fp32_dest_acc_en: bool = True,
        persistent_mode: bool = True,
    ) -> None:
        self._weights = weights
        self._fp32_dest_acc_en = fp32_dest_acc_en
        self._persistent_mode = persistent_mode
        self._state: dict[str, Any] = {}
        print(f"[STAGE] SpecLMHeadStage.__init__ fp32={fp32_dest_acc_en} persistent={persistent_mode}", flush=True)

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        print(f"[STAGE P{ctx.my_mesh_id}] SpecLMHeadStage.create_pipeline_block", flush=True)
        mesh_device = ctx.mesh_device
        my_mesh_id = ctx.my_mesh_id
        pipeline_config = ctx.pipeline_config
        entry_core = ttnn.MeshCoreCoord(
            pipeline_config[my_mesh_id].entry_node_coord,
            SpecLMHeadStage.LMHEAD_INPUT_CORE,
        )
        exit_core = ttnn.MeshCoreCoord(
            pipeline_config[my_mesh_id].exit_node_coord,
            SpecLMHeadStage.ARGMAX_FINAL_CORE,
        )
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=ACTIVATION_W_TOKEN_META_FIFO_SIZE,
            downstream_d2d_socket_fifo_size=TOKEN_META_FIFO_SIZE,
            upstream_d2d_socket_page_size=ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES,
            downstream_d2d_socket_page_size=TOKEN_META_PAGE_SIZE_BYTES,
            entry_node_downstream=entry_core,
            exit_node_upstream=exit_core,
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        print(f"[STAGE P{ctx.my_mesh_id}] SpecLMHeadStage.setup start", flush=True)
        mesh_device = ctx.mesh_device
        my_mesh_id = ctx.my_mesh_id
        pipeline_config = ctx.pipeline_config

        # +32 for metadata (32 * 2 bytes = 64 bytes of metadata)
        torch_a = torch.zeros((SpecLMHeadStage.M, SpecLMHeadStage.K + METADATA_NUM_ELEMS), dtype=torch.bfloat16)
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
        sender_coord = pipeline_config[my_mesh_id].entry_node_coord
        num_devices = mesh_rows * mesh_cols

        cls = SpecLMHeadStage

        mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(cls.LMHEAD_INPUT_CORE, cls.LMHEAD_INPUT_CORE)])
        matmul_core_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
            ]
        )
        argmax_final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(cls.ARGMAX_FINAL_CORE, cls.ARGMAX_FINAL_CORE)])

        input_a_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(mcast_core_grid, (cls.M, cls.K + METADATA_NUM_ELEMS), ttnn.ShardOrientation.ROW_MAJOR),
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(matmul_core_grid, (cls.M, cls.N_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR),
        )
        indices_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(matmul_core_grid, (cls.M, cls.N_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR),
        )
        output_index_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(argmax_final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
        )

        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        bcast_inputs = build_broadcast_test_inputs(
            mesh_device=mesh_device,
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
            sender_coord=ttnn.MeshCoordinate(sender_coord[0], sender_coord[1]),
            output_shape=torch_a.shape,
            input_shard_shape=(SpecLMHeadStage.M, (SpecLMHeadStage.K + METADATA_NUM_ELEMS)),
            tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            layout=ttnn.TILE_LAYOUT,
            input_dtype=ttnn.bfloat16,
            bcast_core=SpecLMHeadStage.LMHEAD_INPUT_CORE,
            input_tensor_torch=torch_a,
            create_output_tensor_mesh=True,
            create_semaphores=True,
            tile=SpecLMHeadStage.A_TILE,
            output_mesh_mapper="shard_dim0",
        )
        input_tensor_mesh = bcast_inputs.input_tensor_mesh
        intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh

        torch_indices_flat = torch.arange(LogicalModelDimensions.VOCAB_SIZE, dtype=torch.int32).reshape(
            1, LogicalModelDimensions.VOCAB_SIZE
        )
        indices_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        ttnn_indices = ttnn.from_torch(
            torch_indices_flat,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=indices_mem_config,
            mesh_mapper=indices_mesh_mapper,
        )
        ttnn_scores = ttnn.from_torch(
            torch.zeros((cls.M, cls.N_TOTAL), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=output_mem_config,
            tile=cls.OUT_TILE,
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
        scratch_shape = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
        scratch_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(argmax_final_core_grid, scratch_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )
        scratch_buffer = ttnn.from_torch(
            torch.zeros((num_devices, *scratch_shape), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=scratch_mem_config,
            mesh_mapper=mesh_mapper,
        )

        # Metadata buffer on argmax_final_core (64 B TOKEN_META page; NCRISC unicast target).
        METADATA_ELEMS = TOKEN_META_PAGE_SIZE_BYTES // 4
        metadata_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(argmax_final_core_grid, (1, METADATA_ELEMS), ttnn.ShardOrientation.ROW_MAJOR),
        )
        metadata_tensor = ttnn.from_torch(
            torch.zeros((num_devices, 1, METADATA_ELEMS), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=metadata_mem_config,
            mesh_mapper=mesh_mapper,
        )

        device_grid_size = mesh_device.compute_with_storage_grid_size()
        worker_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1),
                )
            }
        )

        self._state = {
            "input_tensor_mesh": input_tensor_mesh,
            "intermediate_tensor_mesh": intermediate_tensor_mesh,
            "ttnn_gamma": self._weights.final_norm,
            "ttnn_b": self._weights.lm_head,
            "ttnn_scores": ttnn_scores,
            "ttnn_indices": ttnn_indices,
            "ttnn_output_index": ttnn_output_index,
            "scratch_buffer": scratch_buffer,
            "metadata_tensor": metadata_tensor,
            "lmhead_input_socket": pipeline_block.get_downstream_socket(),
            "lmhead_output_socket": pipeline_block.get_upstream_socket(),
            "bcast_semaphores": bcast_inputs.semaphores,
            "global_semaphore": ttnn.create_global_semaphore(mesh_device, argmax_final_core_grid, 0),
            "global_stage2_semaphore": ttnn.create_global_semaphore(mesh_device, argmax_final_core_grid, 0),
        }
        if self._persistent_mode:
            self._state["persistent_next_iter_semaphore"] = ttnn.create_global_semaphore(mesh_device, worker_crs, 1)
        print(f"[STAGE P{ctx.my_mesh_id}] SpecLMHeadStage.setup done", flush=True)

    def run_auxiliary_sockets(self) -> None:
        pass

    def terminate_auxiliary(self) -> None:
        pass

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        pass


class SpecLMHeadStage(StageKind):
    """MTP LMHead+Sampling+Verification stage: receives base token, runs its own LM head,
    then verifies its speculative token against the base token."""

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

    def __init__(
        self,
        weights: DeepSeekV3LMHeadWeights,
        *,
        fp32_dest_acc_en: bool = True,
        persistent_mode: bool = True,
    ) -> None:
        self._weights = weights
        self._fp32_dest_acc_en = fp32_dest_acc_en
        self._persistent_mode = persistent_mode
        self._state: dict[str, Any] = {}
        print(f"[STAGE] SpecLMHeadStage.__init__ fp32={fp32_dest_acc_en} persistent={persistent_mode}", flush=True)

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        print(f"[STAGE P{ctx.my_mesh_id}] SpecLMHeadStage.create_pipeline_block", flush=True)
        mesh_device = ctx.mesh_device
        my_mesh_id = ctx.my_mesh_id
        pipeline_config = ctx.pipeline_config
        entry_core = ttnn.MeshCoreCoord(
            pipeline_config[my_mesh_id].entry_node_coord,
            SpecLMHeadStage.LMHEAD_INPUT_CORE,
        )
        exit_core = ttnn.MeshCoreCoord(
            pipeline_config[my_mesh_id].exit_node_coord,
            SpecLMHeadStage.ARGMAX_FINAL_CORE,
        )
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=ACTIVATION_W_TOKEN_META_FIFO_SIZE,
            downstream_d2d_socket_fifo_size=TOKEN_META_FIFO_SIZE,
            upstream_d2d_socket_page_size=ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES,
            downstream_d2d_socket_page_size=TOKEN_META_PAGE_SIZE_BYTES,
            entry_node_downstream=entry_core,
            exit_node_upstream=exit_core,
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        print(f"[STAGE P{ctx.my_mesh_id}] SpecLMHeadStage.setup start", flush=True)
        mesh_device = ctx.mesh_device
        my_mesh_id = ctx.my_mesh_id
        pipeline_config = ctx.pipeline_config

        # +32 for metadata (32 * 2 bytes = 64 bytes of metadata)
        torch_a = torch.zeros((SpecLMHeadStage.M, SpecLMHeadStage.K + METADATA_NUM_ELEMS), dtype=torch.bfloat16)
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
        sender_coord = pipeline_config[my_mesh_id].entry_node_coord
        num_devices = mesh_rows * mesh_cols

        cls = SpecLMHeadStage

        mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(cls.LMHEAD_INPUT_CORE, cls.LMHEAD_INPUT_CORE)])
        matmul_core_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
            ]
        )
        argmax_final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(cls.ARGMAX_FINAL_CORE, cls.ARGMAX_FINAL_CORE)])

        input_a_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(mcast_core_grid, (cls.M, cls.K + METADATA_NUM_ELEMS), ttnn.ShardOrientation.ROW_MAJOR),
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(matmul_core_grid, (cls.M, cls.N_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR),
        )
        indices_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(matmul_core_grid, (cls.M, cls.N_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR),
        )
        output_index_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(argmax_final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
        )

        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        bcast_inputs = build_broadcast_test_inputs(
            mesh_device=mesh_device,
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
            sender_coord=ttnn.MeshCoordinate(sender_coord[0], sender_coord[1]),
            output_shape=torch_a.shape,
            input_shard_shape=(SpecLMHeadStage.M, (SpecLMHeadStage.K + METADATA_NUM_ELEMS)),
            tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            layout=ttnn.TILE_LAYOUT,
            input_dtype=ttnn.bfloat16,
            bcast_core=SpecLMHeadStage.LMHEAD_INPUT_CORE,
            input_tensor_torch=torch_a,
            create_output_tensor_mesh=True,
            create_semaphores=True,
            tile=SpecLMHeadStage.A_TILE,
            output_mesh_mapper="shard_dim0",
        )
        input_tensor_mesh = bcast_inputs.input_tensor_mesh
        intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh

        torch_indices_flat = torch.arange(LogicalModelDimensions.VOCAB_SIZE, dtype=torch.int32).reshape(
            1, LogicalModelDimensions.VOCAB_SIZE
        )
        indices_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        ttnn_indices = ttnn.from_torch(
            torch_indices_flat,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=indices_mem_config,
            mesh_mapper=indices_mesh_mapper,
        )
        ttnn_scores = ttnn.from_torch(
            torch.zeros((cls.M, cls.N_TOTAL), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=output_mem_config,
            tile=cls.OUT_TILE,
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
        scratch_shape = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
        scratch_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(argmax_final_core_grid, scratch_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )
        scratch_buffer = ttnn.from_torch(
            torch.zeros((num_devices, *scratch_shape), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=scratch_mem_config,
            mesh_mapper=mesh_mapper,
        )

        # Metadata buffer on argmax_final_core (64 B TOKEN_META page; NCRISC unicast target).
        METADATA_ELEMS = TOKEN_META_PAGE_SIZE_BYTES // 4
        metadata_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(argmax_final_core_grid, (1, METADATA_ELEMS), ttnn.ShardOrientation.ROW_MAJOR),
        )
        metadata_tensor = ttnn.from_torch(
            torch.zeros((num_devices, 1, METADATA_ELEMS), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=metadata_mem_config,
            mesh_mapper=mesh_mapper,
        )

        device_grid_size = mesh_device.compute_with_storage_grid_size()
        worker_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1),
                )
            }
        )

        self._state = {
            "input_tensor_mesh": input_tensor_mesh,
            "intermediate_tensor_mesh": intermediate_tensor_mesh,
            "ttnn_gamma": self._weights.final_norm,
            "ttnn_b": self._weights.lm_head,
            "ttnn_scores": ttnn_scores,
            "ttnn_indices": ttnn_indices,
            "ttnn_output_index": ttnn_output_index,
            "scratch_buffer": scratch_buffer,
            "metadata_tensor": metadata_tensor,
            "lmhead_input_socket": pipeline_block.get_downstream_socket(),
            "lmhead_output_socket": pipeline_block.get_upstream_socket(),
            "bcast_semaphores": bcast_inputs.semaphores,
            "global_semaphore": ttnn.create_global_semaphore(mesh_device, argmax_final_core_grid, 0),
            "global_stage2_semaphore": ttnn.create_global_semaphore(mesh_device, argmax_final_core_grid, 0),
        }
        if self._persistent_mode:
            self._state["persistent_next_iter_semaphore"] = ttnn.create_global_semaphore(mesh_device, worker_crs, 1)
        print(f"[STAGE P{ctx.my_mesh_id}] SpecLMHeadStage.setup done", flush=True)

    def run_auxiliary_sockets(self) -> None:
        pass

    def terminate_auxiliary(self) -> None:
        pass

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        print(f"[STAGE P{ctx.my_mesh_id}] SpecLMHeadStage.launch_compute calling LMHeadSampling.op", flush=True)
        d = self._state
        pipeline_config = ctx.pipeline_config
        my_mesh_id = ctx.my_mesh_id
        LMHeadSampling.op(
            d["input_tensor_mesh"],
            d["intermediate_tensor_mesh"],
            d["ttnn_gamma"],
            d["ttnn_b"],
            d["ttnn_scores"],
            sender_coord=pipeline_config[my_mesh_id].entry_node_coord,
            indices_tensor=d["ttnn_indices"],
            output_index_tensor=d["ttnn_output_index"],
            argmax_final_core_coord=SpecLMHeadStage.ARGMAX_FINAL_CORE,
            argmax_final_mesh_coord=pipeline_config[my_mesh_id].exit_node_coord,
            bcast_semaphores=d["bcast_semaphores"],
            global_semaphore=d["global_semaphore"],
            global_stage2_semaphore=d["global_stage2_semaphore"],
            fabric_scratch_tensor=d["scratch_buffer"],
            fp32_dest_acc_en=self._fp32_dest_acc_en,
            skip_ccl=False,
            socket_input=d["lmhead_input_socket"],
            socket_output=d["lmhead_output_socket"],
            persistent_mode=self._persistent_mode,
            persistent_next_iter_semaphore=d.get("persistent_next_iter_semaphore"),
            is_mtp_base_stage=False,
            is_mtp_verify_stage=True,
            metadata_tensor=d["metadata_tensor"],
        )
        print(f"[STAGE P{ctx.my_mesh_id}] SpecLMHeadStage.launch_compute done", flush=True)


class BaseLMHeadStage(StageKind):
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
    ARGMAX_FINAL_CORE = ttnn.CoreCoord(0, 1)  # Changed from (0, 1) to (0, 0)
    LMHEAD_INPUT_CORE = ttnn.CoreCoord(10, 9)

    def __init__(
        self,
        weights: DeepSeekV3LMHeadWeights,
        *,
        fp32_dest_acc_en: bool = True,
        persistent_mode: bool = True,
        mtp_weights: DeepSeekV3MTPWeights | None = None,
        send_mtp_output_downstream: bool = False,
    ) -> None:
        self._weights = weights
        self._fp32_dest_acc_en = fp32_dest_acc_en
        self._persistent_mode = persistent_mode
        self._mtp_weights = mtp_weights
        self._enable_mtp = mtp_weights is not None
        self._send_mtp_output_downstream = send_mtp_output_downstream and self._enable_mtp
        self._lmhead_state: dict[str, Any] = {}
        print(
            f"[STAGE] BaseLMHeadStage.__init__ fp32={fp32_dest_acc_en} persistent={persistent_mode} mtp={self._enable_mtp} send_mtp_down={self._send_mtp_output_downstream}",
            flush=True,
        )

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        print(f"[STAGE P{ctx.my_mesh_id}] BaseLMHeadStage.create_pipeline_block", flush=True)
        mesh_device = ctx.mesh_device
        my_mesh_id = ctx.my_mesh_id
        pipeline_config = ctx.pipeline_config
        lmhead_entry_core = ttnn.MeshCoreCoord(
            pipeline_config[my_mesh_id].entry_node_coord, BaseLMHeadStage.LMHEAD_INPUT_CORE
        )
        lmhead_exit_core = ttnn.MeshCoreCoord(
            pipeline_config[my_mesh_id].exit_node_coord, BaseLMHeadStage.ARGMAX_FINAL_CORE
        )
        down_page = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES
        down_fifo = ACTIVATION_W_TOKEN_META_FIFO_SIZE
        # Flag here we are creating socket page size of
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=ACTIVATION_W_TOKEN_META_FIFO_SIZE,
            downstream_d2d_socket_fifo_size=down_fifo,
            upstream_d2d_socket_page_size=ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES,
            downstream_d2d_socket_page_size=down_page,
            entry_node_downstream=lmhead_entry_core,
            exit_node_upstream=lmhead_exit_core,
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        print(f"[STAGE P{ctx.my_mesh_id}] BaseLMHeadStage.setup start", flush=True)
        mesh_device = ctx.mesh_device
        my_mesh_id = ctx.my_mesh_id
        pipeline_config = ctx.pipeline_config
        torch_a = torch.zeros((BaseLMHeadStage.M, BaseLMHeadStage.K + METADATA_NUM_ELEMS), dtype=torch.bfloat16)

        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
        sender_coord = pipeline_config[my_mesh_id].entry_node_coord
        num_devices = mesh_rows * mesh_cols

        mcast_core_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(BaseLMHeadStage.LMHEAD_INPUT_CORE, BaseLMHeadStage.LMHEAD_INPUT_CORE)]
        )
        matmul_core_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
            ]
        )
        argmax_final_core_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(BaseLMHeadStage.ARGMAX_FINAL_CORE, BaseLMHeadStage.ARGMAX_FINAL_CORE)]
        )

        input_a_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(mcast_core_grid, (BaseLMHeadStage.M, BaseLMHeadStage.K), ttnn.ShardOrientation.ROW_MAJOR),
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                matmul_core_grid, (BaseLMHeadStage.M, BaseLMHeadStage.N_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR
            ),
        )
        indices_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                matmul_core_grid, (BaseLMHeadStage.M, BaseLMHeadStage.N_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR
            ),
        )
        output_index_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(argmax_final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
        )

        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        bcast_inputs = build_broadcast_test_inputs(
            mesh_device=mesh_device,
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
            sender_coord=ttnn.MeshCoordinate(sender_coord[0], sender_coord[1]),
            output_shape=torch_a.shape,
            input_shard_shape=(BaseLMHeadStage.M, (BaseLMHeadStage.K + METADATA_NUM_ELEMS)),
            tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            layout=ttnn.TILE_LAYOUT,
            input_dtype=ttnn.bfloat16,
            bcast_core=BaseLMHeadStage.LMHEAD_INPUT_CORE,
            input_tensor_torch=torch_a,
            create_output_tensor_mesh=True,
            create_semaphores=True,
            tile=BaseLMHeadStage.A_TILE,
            output_mesh_mapper="shard_dim0",
        )
        input_tensor_mesh = bcast_inputs.input_tensor_mesh
        intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
        ttnn_gamma = self._weights.final_norm
        ttnn_b = self._weights.lm_head
        torch_indices_flat = torch.arange(LogicalModelDimensions.VOCAB_SIZE, dtype=torch.int32).reshape(
            1, LogicalModelDimensions.VOCAB_SIZE
        )
        indices_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        ttnn_indices = ttnn.from_torch(
            torch_indices_flat,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=indices_mem_config,
            mesh_mapper=indices_mesh_mapper,
        )
        ttnn_scores = ttnn.from_torch(
            torch.zeros((BaseLMHeadStage.M, BaseLMHeadStage.N_TOTAL), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=output_mem_config,
            tile=BaseLMHeadStage.OUT_TILE,
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

        METADATA_ELEMS = TOKEN_META_PAGE_SIZE_BYTES // 4
        metadata_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(argmax_final_core_grid, (1, METADATA_ELEMS), ttnn.ShardOrientation.ROW_MAJOR),
        )
        metadata_tensor = ttnn.from_torch(
            torch.zeros((num_devices, 1, METADATA_ELEMS), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=metadata_mem_config,
            mesh_mapper=mesh_mapper,
        )

        # MTP output tensor allocation
        ttnn_mtp_output = None
        if self._enable_mtp:
            compute_cores = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
            compute_core_grid = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
            )
            mtp_output_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(compute_core_grid, (BaseLMHeadStage.M, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
            )
            ttnn_mtp_output = ttnn.from_torch(
                torch.zeros((num_devices, BaseLMHeadStage.M, mtp_padded_dim), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=mtp_output_mem_config,
                tile=BaseLMHeadStage.OUT_TILE,
                mesh_mapper=mesh_mapper,
            )

        sender = BaseLMHeadStage.LMHEAD_INPUT_CORE
        matmul_bbox = matmul_core_grid.bounding_box()
        mcast_end_x = max(matmul_bbox.end.x, sender.x)
        mcast_end_y = max(matmul_bbox.end.y, sender.y)
        mcast_receiver_ranges = []
        if sender.y > 0:
            mcast_receiver_ranges.append(
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(mcast_end_x, sender.y - 1))
            )
        if sender.x > 0:
            mcast_receiver_ranges.append(
                ttnn.CoreRange(ttnn.CoreCoord(0, sender.y), ttnn.CoreCoord(sender.x - 1, sender.y))
            )
        if sender.x < mcast_end_x:
            mcast_receiver_ranges.append(
                ttnn.CoreRange(ttnn.CoreCoord(sender.x + 1, sender.y), ttnn.CoreCoord(mcast_end_x, sender.y))
            )
        if sender.y < mcast_end_y:
            mcast_receiver_ranges.append(
                ttnn.CoreRange(ttnn.CoreCoord(0, sender.y + 1), ttnn.CoreCoord(mcast_end_x, mcast_end_y))
            )

        eh_gather_output_buf = None

        if self._enable_mtp:
            eh_out_w_per_core = mtp_n_per_core // BaseLMHeadStage.OUT_TILE.tile_shape[1]
            eh_gather_total_tiles = num_dram_banks * eh_out_w_per_core + 1
            out_tile_h = BaseLMHeadStage.OUT_TILE.tile_shape[0]
            out_tile_w = BaseLMHeadStage.OUT_TILE.tile_shape[1]
            print(
                f"[STAGE P{ctx.my_mesh_id}] eh_gather_total_tiles={eh_gather_total_tiles}, out_tile_h={out_tile_h}, out_tile_w={out_tile_w}",
                flush=True,
            )
            eh_gather_shard_shape = (eh_gather_total_tiles * out_tile_h, out_tile_w)
            eh_gather_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    argmax_final_core_grid,
                    eh_gather_shard_shape,
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            eh_gather_output_buf = ttnn.from_torch(
                torch.zeros((num_devices * eh_gather_total_tiles * out_tile_h, out_tile_w), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                tile=BaseLMHeadStage.OUT_TILE,
                device=mesh_device,
                memory_config=eh_gather_mem_config,
                mesh_mapper=mesh_mapper,
            )
            print(
                f"[STAGE P{ctx.my_mesh_id}] eh_gather_output_buf shape={eh_gather_output_buf.shape}, memory_config={eh_gather_mem_config}",
                flush=True,
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
            "metadata_tensor": metadata_tensor,
            "lmhead_input_socket": lmhead_input_socket,
            "lmhead_output_socket": lmhead_output_socket,
            "bcast_semaphores": bcast_inputs.semaphores,
            "global_semaphore": global_semaphore,
            "global_stage2_semaphore": global_stage2_semaphore,
            "eh_gather_output_buf": eh_gather_output_buf,
        }
        if self._enable_mtp:
            self._lmhead_state["ttnn_mtp_output"] = ttnn_mtp_output
            self._lmhead_state["ttnn_embedding"] = self._mtp_weights.embedding
            self._lmhead_state["ttnn_h_gamma"] = self._mtp_weights.h_gamma
            self._lmhead_state["ttnn_e_gamma"] = self._mtp_weights.e_gamma
            self._lmhead_state["ttnn_eh_proj"] = self._mtp_weights.eh_projection
        if self._persistent_mode:
            persistent_next_iter_semaphore = ttnn.create_global_semaphore(mesh_device, worker_crs, 1)
            self._lmhead_state["persistent_next_iter_semaphore"] = persistent_next_iter_semaphore
        print(f"[STAGE P{ctx.my_mesh_id}] BaseLMHeadStage.setup done", flush=True)

    def run_auxiliary_sockets(self) -> None:
        pass

    def terminate_auxiliary(self) -> None:
        pass

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        print(
            f"[STAGE P{ctx.my_mesh_id}] BaseLMHeadStage.launch_compute calling LMHeadSampling.op (mtp={self._enable_mtp})",
            flush=True,
        )
        d = self._lmhead_state
        pipeline_config = ctx.pipeline_config
        my_mesh_id = ctx.my_mesh_id
        LMHeadSampling.op(
            d["input_tensor_mesh"],
            d["intermediate_tensor_mesh"],
            d["ttnn_gamma"],
            d["ttnn_b"],
            d["ttnn_scores"],
            sender_coord=pipeline_config[my_mesh_id].entry_node_coord,
            output_mtp_tensor=d.get("ttnn_mtp_output"),
            embedding_tensor=d.get("ttnn_embedding"),
            h_gamma_tensor=d.get("ttnn_h_gamma"),
            e_gamma_tensor=d.get("ttnn_e_gamma"),
            eh_projection_tensor=d.get("ttnn_eh_proj"),
            indices_tensor=d["ttnn_indices"],
            output_index_tensor=d["ttnn_output_index"],
            argmax_final_core_coord=BaseLMHeadStage.ARGMAX_FINAL_CORE,
            argmax_final_mesh_coord=pipeline_config[my_mesh_id].exit_node_coord,
            bcast_semaphores=d["bcast_semaphores"],
            global_semaphore=d["global_semaphore"],
            global_stage2_semaphore=d["global_stage2_semaphore"],
            fabric_scratch_tensor=d["scratch_buffer"],
            fp32_dest_acc_en=self._fp32_dest_acc_en,
            skip_ccl=False,
            socket_input=d["lmhead_input_socket"],
            socket_output=d["lmhead_output_socket"],
            persistent_mode=self._persistent_mode,
            persistent_next_iter_semaphore=d.get("persistent_next_iter_semaphore"),
            is_mtp_base_stage=True,
            eh_gather_output_buf_tensor=d.get("eh_gather_output_buf"),
            metadata_tensor=d.get("metadata_tensor"),
        )
        print(f"[STAGE P{ctx.my_mesh_id}] BaseLMHeadStage.launch_compute done", flush=True)


class _CombinedPipelineBlock:
    """Pipeline block for combined SpecLMHead + Embedding stage.

    Wires four independent socket paths on the same mesh:
    - H2D (token from host) -> fused embedding -> exit D2D (activation to P1)
    - Entry D2D (ACTIVATION_W_TOKEN_META from P3) -> SpecLMHead input (LMHEAD_INPUT_CORE)
    - SpecLMHead output (ARGMAX_FINAL_CORE) -> D2D relay (ARGMAX_RELAY_CORE) -> D2H (TOKEN_META to host)

    The argmax -> D2H path deliberately routes through an intermediate
    SocketInterface relay so that the argmax kernel always uses the well-tested
    socket_mode=2 (D2D / NOC write) code path, matching the original pipeline
    where ARGMAX_FINAL_CORE writes to a d2d_exchange relay on the same exit
    device.

    Implements the same interface as PipelineBlock (run, terminate, write_token,
    read_output, get_downstream_socket, get_upstream_socket) so that Pipeline
    can use it as a drop-in replacement.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        embedding_tensor,
        lmhead_input_core: ttnn.CoreCoord,
        argmax_final_core: ttnn.CoreCoord,
    ) -> None:
        my_mesh_id = mesh_device.get_system_mesh_id()
        num_procs = int(ttnn.distributed_context_get_size())
        pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
        assert len(pipeline_config) == num_procs + 1, "Pipeline config must include loopback entry"

        h2d_device_coord = pipeline_config[my_mesh_id].entry_node_coord
        exit_node_coord = pipeline_config[my_mesh_id].exit_node_coord
        loopback_entry_coord = pipeline_config[num_procs].entry_node_coord
        loopback_exit_coord = pipeline_config[num_procs].exit_node_coord
        next_stage_entry_coord = pipeline_config[my_mesh_id + 1].entry_node_coord
        prev_stage_exit_coord = pipeline_config[num_procs - 1].exit_node_coord

        embedding_size_bytes = embedding_tensor.shape[-1] * 2  # bfloat16
        assert ACTIVATION_PAGE_SIZE_BYTES == embedding_size_bytes

        # -- H2D path (embedding) --
        self.h2d_socket = ttnn.H2DSocket(
            mesh_device,
            ttnn.MeshCoreCoord(h2d_device_coord, EMBEDDING_H2D_CORE_COORD),
            ttnn.BufferType.L1,
            TOKEN_FIFO_SIZE,
            ttnn.H2DMode.HOST_PUSH,
        )

        self.h2d_host_io = HostInterface(
            self.h2d_socket,
            None,
            TOKEN_PAGE_SIZE_BYTES,
            0,
            core_to_core_socket_buffer_size=ACTIVATION_FIFO_SIZE,
            h2d_downstream_core=ttnn.MeshCoreCoord(exit_node_coord, PIPELINE_CORE_COORD),
            embedding_tensor=embedding_tensor,
        )

        self.exit_socket_interface = SocketInterface(
            ACTIVATION_PAGE_SIZE_BYTES,
            ACTIVATION_FIFO_SIZE,
            ACTIVATION_PAGE_SIZE_BYTES,
            ttnn.MeshCoreCoord(exit_node_coord, PIPELINE_CORE_COORD),
            ttnn.MeshCoreCoord(next_stage_entry_coord, PIPELINE_CORE_COORD),
            upstream_socket=self.h2d_host_io.get_downstream_socket(),
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=my_mesh_id + 1),
        )

        # -- SpecLMHead input path (loopback entry from P3) --
        spec_root_device_coord = pipeline_config[my_mesh_id].entry_node_coord
        self.entry_socket_interface = SocketInterface(
            ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES,
            ACTIVATION_W_TOKEN_META_FIFO_SIZE,
            ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES,
            ttnn.MeshCoreCoord(prev_stage_exit_coord, PIPELINE_CORE_COORD),
            ttnn.MeshCoreCoord(loopback_entry_coord, PIPELINE_CORE_COORD),
            downstream_core_coord=ttnn.MeshCoreCoord(spec_root_device_coord, lmhead_input_core),
            sender_mesh=MeshWrapper(mesh_id=num_procs - 1),
            receiver_mesh=MeshWrapper(mesh_device),
        )

        # -- D2H path (verification result via intermediate D2D relay) --
        spec_exit_device_coord = pipeline_config[my_mesh_id].exit_node_coord
        relay_send_device = spec_exit_device_coord
        relay_recv_device = loopback_exit_coord

        self.d2h_socket = ttnn.D2HSocket(
            mesh_device,
            ttnn.MeshCoreCoord(loopback_exit_coord, EMBEDDING_D2H_CORE_COORD),
            TOKEN_FIFO_SIZE,
        )

        self.d2h_host_io = HostInterface(
            None,
            self.d2h_socket,
            0,
            TOKEN_META_PAGE_SIZE_BYTES,
            core_to_core_socket_buffer_size=TOKEN_META_FIFO_SIZE,
            d2h_upstream_core=ttnn.MeshCoreCoord(relay_recv_device, ARGMAX_RELAY_CORE),
        )

        self.argmax_relay = SocketInterface(
            TOKEN_META_PAGE_SIZE_BYTES,
            TOKEN_META_FIFO_SIZE,
            TOKEN_META_PAGE_SIZE_BYTES,
            ttnn.MeshCoreCoord(relay_send_device, ARGMAX_RELAY_CORE),
            ttnn.MeshCoreCoord(relay_recv_device, ARGMAX_RELAY_CORE),
            upstream_core_coord=ttnn.MeshCoreCoord(spec_exit_device_coord, argmax_final_core),
            downstream_socket=self.d2h_host_io.get_upstream_socket(),
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_device),
        )

        print(
            f"[COMBINED P{my_mesh_id}] _CombinedPipelineBlock created: "
            f"h2d_dev={h2d_device_coord} exit_dev={exit_node_coord} "
            f"spec_root={spec_root_device_coord} spec_exit={spec_exit_device_coord} "
            f"relay_send={relay_send_device} relay_recv={relay_recv_device} "
            f"d2h_dev={loopback_exit_coord}",
            flush=True,
        )

    def run(self) -> None:
        self.h2d_host_io.run()
        self.d2h_host_io.run()
        self.argmax_relay.run()
        self.exit_socket_interface.run()
        self.entry_socket_interface.run()

    def terminate(self) -> None:
        ttnn.distributed_context_barrier()
        self.h2d_host_io.terminate(False)
        self.entry_socket_interface.terminate(False)
        self.exit_socket_interface.terminate(False)
        self.argmax_relay.terminate(False)
        self.d2h_host_io.terminate(True)

    def is_first_pipeline_stage(self) -> bool:
        return True

    def write_token(self, token_tensor) -> None:
        self.h2d_socket.write_tensor(token_tensor)

    def read_output(self, output_tensor) -> None:
        self.d2h_socket.read_tensor(output_tensor)

    def get_downstream_socket(self):
        """SpecLMHead reads activation+metadata from this socket (loopback entry downstream)."""
        return self.entry_socket_interface.get_downstream_socket()

    def get_upstream_socket(self):
        """SpecLMHead writes verification result into this D2D socket (to relay, then D2H)."""
        return self.argmax_relay.get_upstream_socket()


class SpecLMHeadWithEmbeddingStage(SpecLMHeadStage):
    """Combined SpecLMHead + Embedding on the same mesh.

    SpecLMHead occupies (0,0)-(10,9).  Embedding I/O uses column 12:
      H2D at EMBEDDING_H2D_CORE_COORD, D2H at EMBEDDING_D2H_CORE_COORD,
      argmax relay at ARGMAX_RELAY_CORE.  Exit D2D relay uses PIPELINE_CORE_COORD.

    Pipeline topology:
      P0(this) -> P1(BaseLMHead+MTP) -> P2(Passthrough) -> P3(Passthrough) -> back to P0
    """

    def __init__(
        self,
        weights: DeepSeekV3LMHeadWeights,
        embedding_weights: DeepSeekV3EmbeddingLayerWeights,
        *,
        fp32_dest_acc_en: bool = True,
        persistent_mode: bool = True,
    ) -> None:
        super().__init__(weights, fp32_dest_acc_en=fp32_dest_acc_en, persistent_mode=persistent_mode)
        self._embedding_weights = embedding_weights
        print(
            f"[STAGE] SpecLMHeadWithEmbeddingStage.__init__ fp32={fp32_dest_acc_en} persistent={persistent_mode}",
            flush=True,
        )

    def create_pipeline_block(self, ctx: StageContext) -> _CombinedPipelineBlock:
        print(f"[STAGE P{ctx.my_mesh_id}] SpecLMHeadWithEmbeddingStage.create_pipeline_block", flush=True)
        return _CombinedPipelineBlock(
            ctx.mesh_device,
            self._embedding_weights.embedding,
            SpecLMHeadStage.LMHEAD_INPUT_CORE,
            SpecLMHeadStage.ARGMAX_FINAL_CORE,
        )
