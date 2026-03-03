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
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
)

# Constants used by stage kinds (same as pipeline module)
token_page_size_bytes = 64
token_fifo_size = 1024
activation_dim = 7168
activation_page_size_bytes = activation_dim * 2
activation_fifo_size = activation_page_size_bytes * 4

M = 1
K = activation_dim
num_matmul_cores = 101
n_per_core = 160
n_total = num_matmul_cores * n_per_core

a_tile = ttnn.Tile([1, 32])
b_tile = ttnn.Tile([32, 32])
out_tile = ttnn.Tile([1, 32])
pipeline_core_coord = ttnn.CoreCoord(11, 0)
argmax_final_core = ttnn.CoreCoord(0, 0)
lmhead_input_core = ttnn.CoreCoord(10, 9)


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
            pipeline_core_coord,
            upstream_d2d_socket_fifo_size=token_fifo_size,
            downstream_d2d_socket_fifo_size=activation_fifo_size,
            upstream_d2d_socket_page_size=token_page_size_bytes,
            downstream_d2d_socket_page_size=activation_page_size_bytes,
            h2d_socket_fifo_size=token_fifo_size,
            d2h_socket_fifo_size=token_fifo_size,
            d2h_socket_page_size=token_page_size_bytes,
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
            up_fifo = down_fifo = activation_fifo_size
            up_page = down_page = activation_page_size_bytes
        else:
            up_fifo = down_fifo = token_fifo_size
            up_page = down_page = token_page_size_bytes
        return PipelineBlock(
            mesh_device,
            pipeline_core_coord,
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
            pipeline_core_coord,
            upstream_d2d_socket_fifo_size=activation_fifo_size,
            downstream_d2d_socket_fifo_size=activation_fifo_size,
            upstream_d2d_socket_page_size=activation_page_size_bytes,
            downstream_d2d_socket_page_size=activation_page_size_bytes,
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        pass

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        pass


class LMHeadStage(StageKind):
    """LMHead+Sampling stage: receive activation, run op, send token downstream."""

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
        self._lmhead_state: dict[str, Any] = {}

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        my_mesh_id = ctx.my_mesh_id
        pipeline_config = ctx.pipeline_config
        lmhead_entry_core = ttnn.MeshCoreCoord(pipeline_config[my_mesh_id].entry_node_coord, lmhead_input_core)
        lmhead_exit_core = ttnn.MeshCoreCoord(pipeline_config[my_mesh_id].exit_node_coord, argmax_final_core)
        return PipelineBlock(
            mesh_device,
            pipeline_core_coord,
            upstream_d2d_socket_fifo_size=activation_fifo_size,
            downstream_d2d_socket_fifo_size=token_fifo_size,
            upstream_d2d_socket_page_size=activation_page_size_bytes,
            downstream_d2d_socket_page_size=token_page_size_bytes,
            entry_node_downstream=lmhead_entry_core,
            exit_node_upstream=lmhead_exit_core,
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        mesh_device = ctx.mesh_device
        my_mesh_id = ctx.my_mesh_id
        pipeline_config = ctx.pipeline_config
        torch_a = torch.zeros((M, K), dtype=torch.bfloat16)

        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
        sender_coord = pipeline_config[my_mesh_id].entry_node_coord
        num_devices = mesh_rows * mesh_cols

        mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(lmhead_input_core, lmhead_input_core)])
        matmul_core_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
            ]
        )
        argmax_final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)])

        input_a_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
        )
        indices_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
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
            tile=a_tile,
            dtype=ttnn.bfloat16,
            memory_config=input_a_mem_config,
            mesh_mapper=mesh_mapper,
        )
        intermediate_tensor_mesh = ttnn.from_torch(
            mesh_intermediate,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            tile=a_tile,
            dtype=ttnn.bfloat16,
            memory_config=input_a_mem_config,
            mesh_mapper=mesh_mapper,
        )
        ttnn_gamma = self._weights.final_norm
        ttnn_b = self._weights.lm_head
        torch_indices_flat = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
        ttnn_indices = ttnn.from_torch(
            torch_indices_flat.repeat(num_devices, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=indices_mem_config,
            mesh_mapper=mesh_mapper,
        )
        ttnn_scores = ttnn.from_torch(
            torch.zeros((M, n_total), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=output_mem_config,
            tile=out_tile,
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
        if self._persistent_mode:
            persistent_next_iter_semaphore = ttnn.create_global_semaphore(mesh_device, worker_crs, 1)
            self._lmhead_state["persistent_next_iter_semaphore"] = persistent_next_iter_semaphore

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
            indices_tensor=d["ttnn_indices"],
            output_index_tensor=d["ttnn_output_index"],
            argmax_final_core_coord=argmax_final_core,
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
        )
