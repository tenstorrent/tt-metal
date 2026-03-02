# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pod pipeline orchestration with StageKind interface (Embedding, LMHead, Passthrough).
Shared by the demo CLI and persistent-mode tests (4-stage and 16-stage).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.lm_head_sampling.op import LMHeadSampling
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import ttnn_dtype_from_torch_dtype
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock

# Module-level constants (from persistent mode tests)
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


def create_fabric_router_config(max_payload_size: int) -> Any:
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@dataclass
class LMHeadWeights:
    """Torch tensors for LMHead stage (RMS norm gamma, weight matrix, vocab indices)."""

    gamma: torch.Tensor  # [M, K]
    weight_matrix: torch.Tensor  # [K, n_total]
    indices: torch.Tensor  # [1, n_total]


def create_synthetic_weights(
    iterations: int,
) -> tuple[torch.Tensor, LMHeadWeights, torch.Tensor]:
    """
    Build deterministic synthetic weights and expected output indices.
    Returns (embedding_tensor, lmhead_weights, expected_indices).
    """
    torch_gamma = torch.ones((M, K), dtype=torch.bfloat16)
    row_indices = torch.arange(iterations, dtype=torch.int64) % K
    torch_embedding_table = torch.zeros((iterations, K), dtype=torch.bfloat16)
    torch_embedding_table[torch.arange(iterations), row_indices] = 1
    winner_per_row = torch.arange(K, dtype=torch.int64) % n_total
    torch_b = torch.full((K, n_total), fill_value=-1.0, dtype=torch.bfloat16)
    torch_b[torch.arange(K), winner_per_row] = 1
    torch_indices_flat = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
    torch_expected_indices = torch.stack(
        [
            LMHeadSampling.golden(
                torch_embedding_table[iteration : iteration + 1].float(),
                torch_gamma.float(),
                torch_b.float().unsqueeze(0),
                indices=torch_indices_flat,
                k=1,
                p=1.0,
            ).to(torch.uint32)
            for iteration in range(iterations)
        ],
        dim=0,
    )
    embedding_tensor = torch_embedding_table.reshape(iterations, 1, 1, K)
    lmhead_weights = LMHeadWeights(
        gamma=torch_gamma,
        weight_matrix=torch_b,
        indices=torch_indices_flat,
    )
    return embedding_tensor, lmhead_weights, torch_expected_indices


def create_random_weights_single_iteration(
    seed: int = 5449,
) -> tuple[torch.Tensor, LMHeadWeights, torch.Tensor]:
    """
    Build random weights for a single token iteration (e.g. 4-stage single-galaxy test).
    Returns (embedding_tensor [1,1,1,K], lmhead_weights, expected_idx from golden).
    """
    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_flat = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
    torch_expected_idx = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0),
        indices=torch_indices_flat,
        k=1,
        p=1.0,
    ).to(torch.uint32)
    embedding_tensor = torch_a.reshape(1, 1, 1, K)
    lmhead_weights = LMHeadWeights(
        gamma=torch_gamma,
        weight_matrix=torch_b,
        indices=torch_indices_flat,
    )
    return embedding_tensor, lmhead_weights, torch_expected_idx


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

    def __init__(self, embedding_tensor: torch.Tensor) -> None:
        self._embedding_tensor = embedding_tensor

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        embedding_ttnn = ttnn.from_torch(
            self._embedding_tensor,
            dtype=ttnn_dtype_from_torch_dtype(self._embedding_tensor.dtype),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        embedding_ttnn = ttnn.to_device(embedding_ttnn, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
            embedding_tensor=embedding_ttnn,
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


class LMHeadStage(StageKind):
    """LMHead+Sampling stage: receive activation, run op, send token downstream."""

    def __init__(
        self,
        weights: LMHeadWeights,
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
        torch_gamma = self._weights.gamma
        torch_b = self._weights.weight_matrix
        torch_indices_flat = self._weights.indices

        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
        sender_coord = pipeline_config[my_mesh_id].entry_node_coord

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
        width_shard_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
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

        num_devices = mesh_rows * mesh_cols
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
        ttnn_gamma = ttnn.from_torch(
            torch_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=input_a_mem_config,
            tile=a_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn_b = ttnn.from_torch(
            torch_b,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=width_shard_mem_config,
            tile=b_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
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
        ttnn_indices = ttnn.from_torch(
            torch_indices_flat.repeat(num_devices, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=indices_mem_config,
            mesh_mapper=mesh_mapper,
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


def create_stage_kind(
    my_mesh_id: int,
    lmhead_stage: int,
    *,
    embedding_tensor: torch.Tensor | None = None,
    lmhead_weights: LMHeadWeights | None = None,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
) -> StageKind:
    """Factory: return the StageKind for the given mesh_id and topology."""
    if my_mesh_id == 0:
        if embedding_tensor is None:
            raise ValueError("Stage 0 requires embedding_tensor")
        return EmbeddingStage(embedding_tensor)
    elif my_mesh_id == lmhead_stage:
        if lmhead_weights is None:
            raise ValueError(f"LMHead stage ({lmhead_stage}) requires lmhead_weights")
        return LMHeadStage(
            weights=lmhead_weights,
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
        )
    elif my_mesh_id > lmhead_stage:
        return PassthroughStage(PassthroughPayload.TOKEN)
    else:
        return PassthroughStage(PassthroughPayload.ACTIVATION)


class PodPipeline:
    """Orchestrator for one pipeline stage; delegates to StageKind."""

    def __init__(self, mesh_device: ttnn.MeshDevice, stage_kind: StageKind) -> None:
        self._mesh_device = mesh_device
        self._stage_kind = stage_kind
        self._my_mesh_id = mesh_device.get_system_mesh_id()
        self._pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
        self._ctx = StageContext(
            mesh_device=mesh_device,
            pipeline_config=self._pipeline_config,
            my_mesh_id=self._my_mesh_id,
        )
        self._pipeline_block: PipelineBlock | None = None

    @property
    def my_mesh_id(self) -> int:
        return self._my_mesh_id

    def setup(self) -> None:
        self._pipeline_block = self._stage_kind.create_pipeline_block(self._ctx)
        self._stage_kind.setup(self._ctx, self._pipeline_block)

    def run(self) -> None:
        if self._pipeline_block is None:
            raise RuntimeError("PodPipeline.setup() must be called before run()")
        self._pipeline_block.run()
        self._stage_kind.launch_compute(self._ctx, self._pipeline_block)

    def write_token(self, token_tensor: ttnn.Tensor) -> None:
        if self._pipeline_block is None:
            raise RuntimeError("PodPipeline.setup() must be called first")
        self._pipeline_block.write_token(token_tensor)

    def read_output(self, output_tensor: ttnn.Tensor) -> None:
        if self._pipeline_block is None:
            raise RuntimeError("PodPipeline.setup() must be called first")
        self._pipeline_block.read_output(output_tensor)

    def barrier(self) -> None:
        ttnn.distributed_context_barrier()

    def terminate(self) -> None:
        """Terminate the pipeline block if it was created (e.g. for one-shot tests)."""
        if self._pipeline_block is not None:
            self._pipeline_block.terminate()
