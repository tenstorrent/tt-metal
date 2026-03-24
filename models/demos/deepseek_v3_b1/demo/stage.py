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
from models.demos.deepseek_v3_b1.fused_ops.lm_head_sampling.op import LMHeadSampling
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    DeepSeekV3MTPWeights,
)

# Global constants used by multiple stage kinds (and exported to pipeline/cli)
TOKEN_PAGE_SIZE_BYTES = 64
TOKEN_FIFO_SIZE = 1024
ACTIVATION_DIM = 7168
ACTIVATION_PAGE_SIZE_BYTES = ACTIVATION_DIM * 2
ACTIVATION_FIFO_SIZE = ACTIVATION_PAGE_SIZE_BYTES * 4
PIPELINE_CORE_COORD = ttnn.CoreCoord(11, 0)

# MTP constants
embedding_dim = 7168
mtp_output_dim = 7168
num_dram_banks = 8
mtp_n_per_core = mtp_output_dim // num_dram_banks
mtp_padded_dim = num_dram_banks * mtp_n_per_core

# Token metadata payload: just token info (id, type, pos) — same physical size as TOKEN.
TOKEN_META_PAGE_SIZE_BYTES = TOKEN_PAGE_SIZE_BYTES
TOKEN_META_FIFO_SIZE = TOKEN_FIFO_SIZE

# Activation + token metadata payload: gathered EH logits + 1 metadata tile (token).
# Tile [1,32] bf16 → tile_size bytes per tile; total = (num_cores * tiles_per_core + 1) * tile_size.
_ACT_W_META_TILE = ttnn.Tile([1, 32])
_ACT_W_META_TILE_SIZE = _ACT_W_META_TILE.get_tile_size(ttnn.bfloat16)
_ACT_W_META_TILES_PER_CORE = mtp_n_per_core // _ACT_W_META_TILE.tile_shape[1]
_ACT_W_META_TOTAL_TILES = num_dram_banks * _ACT_W_META_TILES_PER_CORE + 1
ACTIVATION_W_TOKEN_META_TILE_SIZE_BYTES = _ACT_W_META_TILE_SIZE
ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES = _ACT_W_META_TOTAL_TILES * _ACT_W_META_TILE_SIZE
ACTIVATION_W_TOKEN_META_FIFO_SIZE = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES * 2
ACTIVATION_W_TOKEN_META_LOGITS_BYTES = (_ACT_W_META_TOTAL_TILES - 1) * _ACT_W_META_TILE_SIZE


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

    def run_auxiliary_sockets(self) -> None:
        """Start auxiliary (bypass) d2d_exchange kernels. Default: no-op."""

    def terminate_auxiliary(self) -> None:
        """Terminate auxiliary sockets. Default: no-op."""

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        """Launch compute kernels after pipeline_block.run(). Default: no-op."""


class EmbeddingStage(StageKind):
    """Stage 0: H2D + embedding lookup, forwards activation; loopback receives token."""

    def __init__(self, weights: DeepSeekV3EmbeddingLayerWeights, *, d2h_page_size: int | None = None) -> None:
        self._weights = weights
        self._d2h_page_size = d2h_page_size or TOKEN_PAGE_SIZE_BYTES

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        d2h_page = self._d2h_page_size
        d2h_fifo = max(d2h_page * 2, TOKEN_FIFO_SIZE)
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=d2h_fifo,
            downstream_d2d_socket_fifo_size=ACTIVATION_FIFO_SIZE,
            upstream_d2d_socket_page_size=d2h_page,
            downstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
            h2d_socket_fifo_size=TOKEN_FIFO_SIZE,
            d2h_socket_fifo_size=d2h_fifo,
            d2h_socket_page_size=d2h_page,
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

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
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

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        my_mesh_id = ctx.my_mesh_id
        pipeline_config = ctx.pipeline_config
        entry_core = ttnn.MeshCoreCoord(
            pipeline_config[my_mesh_id].entry_node_coord,
            MTPVerificationLMHeadStage.ARGMAX_FINAL_CORE,
        )
        exit_core = ttnn.MeshCoreCoord(
            pipeline_config[my_mesh_id].exit_node_coord,
            MTPVerificationLMHeadStage.ARGMAX_FINAL_CORE,
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
        mesh_device = ctx.mesh_device
        my_mesh_id = ctx.my_mesh_id
        pipeline_config = ctx.pipeline_config

        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
        sender_coord = pipeline_config[my_mesh_id].entry_node_coord
        num_devices = mesh_rows * mesh_cols

        cls = MTPVerificationLMHeadStage

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
            ttnn.ShardSpec(mcast_core_grid, (cls.M, cls.K), ttnn.ShardOrientation.ROW_MAJOR),
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

        torch_a = torch.zeros((cls.M, cls.K), dtype=torch.bfloat16)
        device_inputs = []
        device_intermediate = []
        for r in range(mesh_rows):
            for c in range(mesh_cols):
                if r == sender_coord[0] and c == sender_coord[1]:
                    device_inputs.append(torch_a)
                else:
                    device_inputs.append(torch.zeros_like(torch_a))
                device_intermediate.append(torch.zeros_like(torch_a))
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)

        input_tensor_mesh = ttnn.from_torch(
            torch.cat(device_inputs, dim=0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            tile=cls.A_TILE,
            dtype=ttnn.bfloat16,
            memory_config=input_a_mem_config,
            mesh_mapper=mesh_mapper,
        )
        intermediate_tensor_mesh = ttnn.from_torch(
            torch.cat(device_intermediate, dim=0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            tile=cls.A_TILE,
            dtype=ttnn.bfloat16,
            memory_config=input_a_mem_config,
            mesh_mapper=mesh_mapper,
        )
        ttnn_indices = ttnn.from_torch(
            torch.arange(cls.N_TOTAL, dtype=torch.int32).reshape(1, cls.N_TOTAL).repeat(num_devices, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=indices_mem_config,
            mesh_mapper=mesh_mapper,
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

        # Base token tensors — all on argmax_final_core.
        # base_token_tensor is 64 bytes (TOKEN_META landing buffer for NCRISC NOC write).
        _SENTINEL = 0xFFFFFFFF
        _TOKEN_META_ELEMS = TOKEN_META_PAGE_SIZE_BYTES // 4
        token_meta_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(argmax_final_core_grid, (1, _TOKEN_META_ELEMS), ttnn.ShardOrientation.ROW_MAJOR),
        )
        base_token_tensor = ttnn.from_torch(
            torch.full((num_devices, 1, _TOKEN_META_ELEMS), _SENTINEL, dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=token_meta_mem_config,
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
            "base_token_tensor": base_token_tensor,
            "lmhead_input_socket": pipeline_block.get_downstream_socket(),
            "lmhead_output_socket": pipeline_block.get_upstream_socket(),
            "out_ready_semaphore": ttnn.create_global_semaphore(mesh_device, worker_crs, 0),
            "barrier_semaphore": ttnn.create_global_semaphore(mesh_device, worker_crs, 0),
            "secondary_sync_semaphore": ttnn.create_global_semaphore(mesh_device, worker_crs, 0),
            "global_semaphore": ttnn.create_global_semaphore(mesh_device, argmax_final_core_grid, 0),
            "global_stage2_semaphore": ttnn.create_global_semaphore(mesh_device, argmax_final_core_grid, 0),
        }
        if self._persistent_mode:
            self._state["persistent_next_iter_semaphore"] = ttnn.create_global_semaphore(mesh_device, worker_crs, 1)

    def run_auxiliary_sockets(self) -> None:
        pass

    def terminate_auxiliary(self) -> None:
        pass

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
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
            argmax_final_core_coord=MTPVerificationLMHeadStage.ARGMAX_FINAL_CORE,
            argmax_final_mesh_coord=pipeline_config[my_mesh_id].exit_node_coord,
            semaphores=[
                d["out_ready_semaphore"],
                d["barrier_semaphore"],
                d["secondary_sync_semaphore"],
            ],
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
            base_token_tensor=d["base_token_tensor"],
        )


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
    ARGMAX_FINAL_CORE = ttnn.CoreCoord(0, 0)
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
        if self._send_mtp_output_downstream:
            down_page = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES
            down_fifo = ACTIVATION_W_TOKEN_META_FIFO_SIZE
        else:
            down_page = TOKEN_PAGE_SIZE_BYTES
            down_fifo = TOKEN_FIFO_SIZE
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=ACTIVATION_FIFO_SIZE,
            downstream_d2d_socket_fifo_size=down_fifo,
            upstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
            downstream_d2d_socket_page_size=down_page,
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
            ((mesh_rows + mesh_cols) * winner_page_bytes + (256 + 8 if self._enable_mtp else 0)) // 4,
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
                ttnn.ShardSpec(compute_core_grid, (LMHeadStage.M, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
            )
            ttnn_mtp_output = ttnn.from_torch(
                torch.zeros((num_devices, LMHeadStage.M, mtp_padded_dim), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=mtp_output_mem_config,
                tile=LMHeadStage.OUT_TILE,
                mesh_mapper=mesh_mapper,
            )

        sender = LMHeadStage.LMHEAD_INPUT_CORE
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

        mcast_receiver_grid = ttnn.CoreRangeSet(mcast_receiver_ranges)
        num_receiver_cores = (mcast_end_x + 1) * (mcast_end_y + 1) - 1

        mcast_dst_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(mcast_receiver_grid, (LMHeadStage.M, LMHeadStage.K), ttnn.ShardOrientation.ROW_MAJOR),
        )
        mcast_dst_working_buf = ttnn.from_torch(
            torch.zeros((num_devices * num_receiver_cores, LMHeadStage.K), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=LMHeadStage.A_TILE,
            device=mesh_device,
            memory_config=mcast_dst_mem_config,
            mesh_mapper=mesh_mapper,
        )

        mcast_eh_dst_working_buf = None
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
            "mcast_dst_working_buf": mcast_dst_working_buf,
            "mcast_eh_dst_working_buf": mcast_eh_dst_working_buf,
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

    def run_auxiliary_sockets(self) -> None:
        pass

    def terminate_auxiliary(self) -> None:
        pass

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
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
            mcast_dst_working_buf_tensor=d.get("mcast_dst_working_buf"),
            mcast_eh_dst_working_buf_tensor=d.get("mcast_eh_dst_working_buf"),
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
            fp32_dest_acc_en=self._fp32_dest_acc_en,
            skip_ccl=False,
            socket_input=d["lmhead_input_socket"],
            socket_output=d["lmhead_output_socket"],
            persistent_mode=self._persistent_mode,
            persistent_next_iter_semaphore=d.get("persistent_next_iter_semaphore"),
            enable_mtp=self._enable_mtp,
            eh_subblock_k=d.get("eh_subblock_k"),
        )
