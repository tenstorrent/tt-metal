# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.lm_head_sampling.op import LMHeadSampling
from models.demos.deepseek_v3_b1.metadata.metadata import (
    METADATA_TENSOR_BYTES,
    METADATA_TENSOR_NUM_BF16,
    METADATA_TENSOR_NUM_UINT32,
)
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import MeshWrapper, SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.persistent_loop.op import PersistentLoop
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import (
    HostIoPlacement,
    LoopbackConfig,
    PipelineBlock,
    PipelineBlockKind,
    StageMetadata,
)
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import build_broadcast_test_inputs
from models.demos.deepseek_v3_b1.utils import get_pinned_optimal_dram_bank_to_logical_worker_assignment
from models.demos.deepseek_v3_b1.weights.prepare import (
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MTPWeights,
    DeepSeekV3SpecWeights,
)

# Global constants used by multiple stage kinds (and exported to pipeline/cli)
TOKEN_FIFO_NUM_PAGES = 64
ACTIVATION_DIM = 7168
DEFAULT_ACTIVATION_FIFO_PAGES = 1
SINGLE_BUFFER_FIFO_PAGES = 1


def activation_fifo_size_bytes(page_size_bytes: int, fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES) -> int:
    if fifo_pages < 1:
        raise ValueError(f"fifo_pages must be >= 1, got {fifo_pages}")
    return page_size_bytes * fifo_pages


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _mesh_sampling_scratch_shapes(mesh_rows: int, mesh_cols: int) -> tuple[tuple[int, int], tuple[int, int]]:
    topk_min_alignment = 32
    bf16_tile_size = 2 * 32 * 32
    uint32_tile_size = 4 * 32 * 32
    stage1_tiles = (mesh_rows * topk_min_alignment + 1023) // 1024
    stage2_tiles = (mesh_cols * topk_min_alignment + 1023) // 1024
    total_tiles = stage1_tiles + stage2_tiles
    scores_bytes = total_tiles * bf16_tile_size
    indices_bytes = total_tiles * uint32_tile_size
    scores_width = _round_up(scores_bytes, 2) // 2
    indices_width = _round_up(indices_bytes, 4) // 4
    return (1, scores_width), (1, indices_width)


ACTIVATION_PAGE_SIZE_BYTES = ACTIVATION_DIM * 2
ACTIVATION_FIFO_SIZE = activation_fifo_size_bytes(ACTIVATION_PAGE_SIZE_BYTES)
PIPELINE_CORE_COORD = ttnn.CoreCoord(12, 8)
SECOND_PIPELINE_CORE_COORD = ttnn.CoreCoord(12, 7)

# Embedding D2H core coord for the combined SpecLMHead+Embedding stage (column 12, outside mcast grid)
EMBEDDING_D2H_CORE_COORD = ttnn.CoreCoord(12, 1)

# MTP constants
num_dram_banks = 8
# Number of bf16 elements appended to each activation shard to carry the full
# DeepseekMetadata struct (header + p_indices + p_scores). The source unicasts
# the whole struct from the LM-head input core to the sampling final core; only
# the first `aligned_size_bytes()` bytes (the header) come from upstream — the
# trailing region stays zero on the source and is overwritten on the destination
# by sampling.hpp after top-P.
METADATA_NUM_ELEMS = METADATA_TENSOR_NUM_BF16
mtp_n_per_core = ACTIVATION_DIM // num_dram_banks
mtp_padded_dim = num_dram_banks * mtp_n_per_core

# Token metadata payload: full DeepseekMetadata struct (header + p_indices + p_scores).
# This is the per-iteration metadata that flows through the model: every decoder
# stage carries it, the BaseLMHeadStage produces the trailing p_indices / p_scores
# arrays inside it, and the SpecLMHeadStage forwards it. Spec-LM output and the
# embedding input are the only paths that don't carry the full struct.
# FIFO depth is kept at TOKEN_FIFO_NUM_PAGES so buffering capacity (in pages) is
# unchanged from the prior 64 B layout — total L1 footprint scales with page size.
TOKEN_META_PAGE_SIZE_BYTES = METADATA_TENSOR_BYTES
TOKEN_META_FIFO_SIZE = TOKEN_META_PAGE_SIZE_BYTES * TOKEN_FIFO_NUM_PAGES

# Activation + metadata payload: logits + full DeepseekMetadata.
ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES = ACTIVATION_PAGE_SIZE_BYTES + TOKEN_META_PAGE_SIZE_BYTES
ACTIVATION_W_TOKEN_META_FIFO_SIZE = activation_fifo_size_bytes(ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES)


@dataclass
class StageContext:
    """Bundles arguments passed to StageKind methods."""

    mesh_device: ttnn.MeshDevice
    pipeline_config: list
    my_stage_idx: int
    stages_metadata: dict[int, StageMetadata] | None = None

    @property
    def my_mesh_id(self) -> int:
        return self.my_stage_idx


class StageKind(ABC):
    """Abstract stage kind: controls PipelineBlock creation, setup, and compute launch."""

    @abstractmethod
    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlockKind:
        """Create and return the pipeline block for this stage."""

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlockKind) -> None:
        """Post-creation setup (tensor allocation, etc).

        Decoder stages may also compile/build device programs here so ``launch_compute`` only
        enqueues execution. Default: no-op.
        """

    def run_auxiliary_sockets(self) -> None:
        """Start auxiliary (bypass) d2d_exchange kernels. Default: no-op."""

    def terminate_auxiliary(self) -> None:
        """Terminate auxiliary sockets. Default: no-op."""

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlockKind) -> None:
        """Run stage compute after ``pipeline_block.run()`` (execute pre-built programs where applicable). Default: no-op."""

    def terminate(self, ctx: StageContext, pipeline_block: PipelineBlockKind) -> None:
        """Signal the stage's persistent compute kernels to exit on the next iteration.

        Concrete stages that launch persistent compute kernels (e.g. LMHead, Decoder)
        override this to set a termination semaphore. The pipeline orchestrator then
        pushes a dummy token through the pipeline so each stage can complete its final
        iteration and break naturally at the top-of-loop termination check. Default: no-op.
        """


class PassthroughPayload(Enum):
    ACTIVATION = "activation"
    TOKEN = "token"
    TOKEN_META = "token_meta"
    ACTIVATION_W_TOKEN_META = "activation_w_token_meta"


class EmbeddingStage(StageKind):
    """Stage 0: H2D + embedding lookup with configurable downstream/loopback payloads."""

    def __init__(
        self,
        weights: DeepSeekV3EmbeddingLayerWeights,
        *,
        loopback_payload: PassthroughPayload = PassthroughPayload.TOKEN,
        d2h_page_size: int | None = None,
        host_loopback: bool = False,
    ) -> None:
        self._weights = weights
        self._loopback_payload = loopback_payload
        self._host_loopback = host_loopback
        self._d2h_page_size = d2h_page_size

    @staticmethod
    def _payload_sizes(payload: PassthroughPayload) -> tuple[int, int]:
        if payload == PassthroughPayload.ACTIVATION:
            return ACTIVATION_FIFO_SIZE, ACTIVATION_PAGE_SIZE_BYTES
        if payload == PassthroughPayload.ACTIVATION_W_TOKEN_META:
            return ACTIVATION_W_TOKEN_META_FIFO_SIZE, ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES
        if payload == PassthroughPayload.TOKEN_META:
            return TOKEN_META_FIFO_SIZE, TOKEN_META_PAGE_SIZE_BYTES
        return TOKEN_META_FIFO_SIZE, TOKEN_META_PAGE_SIZE_BYTES

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        my_stage_idx = ctx.my_stage_idx
        activation_fifo_size = ACTIVATION_W_TOKEN_META_FIFO_SIZE
        activation_page_size = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES
        pipeline_config = ctx.pipeline_config
        if self._d2h_page_size is not None:
            size_to_payload = {
                TOKEN_META_PAGE_SIZE_BYTES: PassthroughPayload.TOKEN,
                TOKEN_META_PAGE_SIZE_BYTES: PassthroughPayload.TOKEN_META,
                ACTIVATION_PAGE_SIZE_BYTES: PassthroughPayload.ACTIVATION,
                ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES: PassthroughPayload.ACTIVATION_W_TOKEN_META,
            }
            if self._d2h_page_size not in size_to_payload:
                raise ValueError(f"Unsupported d2h_page_size: {self._d2h_page_size}")
            loopback_payload = size_to_payload[self._d2h_page_size]
        else:
            loopback_payload = self._loopback_payload

        up_fifo, up_page = self._payload_sizes(loopback_payload)
        d2h_fifo, d2h_page = up_fifo, up_page

        num_procs = len(pipeline_config) - 1
        host_io_placement = self._create_host_io_placement(pipeline_config, num_procs)
        if self._host_loopback:
            loopback = LoopbackConfig.host_loopback(host_io_placement=host_io_placement)
        else:
            loopback = LoopbackConfig.fabric_loopback(host_io_placement=host_io_placement)

        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=up_fifo,
            downstream_d2d_socket_fifo_size=activation_fifo_size,
            upstream_d2d_socket_page_size=up_page,
            downstream_d2d_socket_page_size=activation_page_size,
            h2d_socket_fifo_size=TOKEN_META_FIFO_SIZE,
            d2h_socket_fifo_size=d2h_fifo,
            d2h_socket_page_size=d2h_page,
            embedding_tensor=self._weights.embedding,
            forward_metadata=True,
            loopback=loopback,
            my_stage_idx=my_stage_idx,
            stages_metadata=ctx.stages_metadata,
            pipeline_config=pipeline_config,
        )

    @staticmethod
    def _create_host_io_placement(pipeline_config, num_procs) -> HostIoPlacement:
        """Resolve per-socket core coords for the four stage-0 kernels.

        When the H2D chip and the forward D2D chip are the same device, two
        persistent BRISC kernels would land on the same Tensix core.  We move
        H2D (not D2D) to the alt core so the D2D send core stays at
        PIPELINE_CORE_COORD — other stages build their entry socket configs
        using pipeline_core_coord and must see the same sender core.  The same
        logic applies to D2H vs. the loopback D2D entry.
        """
        h2d_chip = pipeline_config[0].entry_node_coord
        fwd_d2d_chip = pipeline_config[0].exit_node_coord
        lb_d2d_chip = pipeline_config[num_procs].entry_node_coord
        d2h_chip = pipeline_config[num_procs].exit_node_coord

        def _same(a, b):
            return a[0] == b[0] and a[1] == b[1]

        h2d_core = SECOND_PIPELINE_CORE_COORD if _same(h2d_chip, fwd_d2d_chip) else PIPELINE_CORE_COORD
        d2h_core = SECOND_PIPELINE_CORE_COORD if _same(d2h_chip, lb_d2d_chip) else PIPELINE_CORE_COORD

        return HostIoPlacement(
            h2d_core=h2d_core,
            d2h_core=d2h_core,
            fwd_d2d_core=PIPELINE_CORE_COORD,
            lb_d2d_core=PIPELINE_CORE_COORD,
        )


class PassthroughStage(StageKind):
    """Forward-only stage: activation or token passthrough."""

    def __init__(
        self,
        payload: PassthroughPayload,
        *,
        host_loopback: bool = False,
        upstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        downstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
    ) -> None:
        self._payload = payload
        self._upstream_fifo_pages = upstream_fifo_pages
        self._downstream_fifo_pages = downstream_fifo_pages
        self._host_loopback = host_loopback

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        my_stage_idx = ctx.my_stage_idx
        if self._payload == PassthroughPayload.ACTIVATION:
            up_fifo = activation_fifo_size_bytes(ACTIVATION_PAGE_SIZE_BYTES, self._upstream_fifo_pages)
            down_fifo = activation_fifo_size_bytes(ACTIVATION_PAGE_SIZE_BYTES, self._downstream_fifo_pages)
            up_page = down_page = ACTIVATION_PAGE_SIZE_BYTES
        elif self._payload == PassthroughPayload.ACTIVATION_W_TOKEN_META:
            up_fifo = activation_fifo_size_bytes(ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES, self._upstream_fifo_pages)
            down_fifo = activation_fifo_size_bytes(ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES, self._downstream_fifo_pages)
            up_page = down_page = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES
        elif self._payload == PassthroughPayload.TOKEN_META:
            up_fifo = down_fifo = TOKEN_META_FIFO_SIZE
            up_page = down_page = TOKEN_META_PAGE_SIZE_BYTES
        else:
            up_fifo = down_fifo = TOKEN_META_FIFO_SIZE
            up_page = down_page = TOKEN_META_PAGE_SIZE_BYTES
        if self._host_loopback:
            # d2h_core must differ from PIPELINE_CORE_COORD: both land on the same chip
            # (no_loopback sets exit_node_coord = entry_node_coord), so using the same
            # core would dispatch two persistent kernels to the same Tensix.
            loopback = LoopbackConfig.host_loopback(
                HostIoPlacement(
                    h2d_core=PIPELINE_CORE_COORD,
                    d2h_core=SECOND_PIPELINE_CORE_COORD,
                    fwd_d2d_core=PIPELINE_CORE_COORD,
                    lb_d2d_core=PIPELINE_CORE_COORD,
                )
            )
        else:
            loopback = LoopbackConfig.fabric_loopback(HostIoPlacement.default(PIPELINE_CORE_COORD))
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=up_fifo,
            downstream_d2d_socket_fifo_size=down_fifo,
            upstream_d2d_socket_page_size=up_page,
            downstream_d2d_socket_page_size=down_page,
            d2h_socket_fifo_size=up_fifo if self._host_loopback else None,
            d2h_socket_page_size=up_page if self._host_loopback else None,
            loopback=loopback,
            my_stage_idx=my_stage_idx,
            stages_metadata=ctx.stages_metadata,
            pipeline_config=ctx.pipeline_config,
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
    OUT_TILE = ttnn.Tile([1, 32])
    ARGMAX_FINAL_CORE = ttnn.CoreCoord(0, 0)
    LMHEAD_INPUT_CORE = ttnn.CoreCoord(10, 9)

    def __init__(
        self,
        *,
        fp32_dest_acc_en: bool = True,
        persistent_mode: bool = True,
        spec_weights: DeepSeekV3SpecWeights | None = None,
    ) -> None:
        self._fp32_dest_acc_en = fp32_dest_acc_en
        self._persistent_mode = persistent_mode
        self._spec_weights = spec_weights
        self._state: dict[str, Any] = {}

    def _get_sender_coord(self, ctx: StageContext, pipeline_block):
        """Return the broadcast sender MeshCoordinate for SpecLMHead."""
        return ctx.pipeline_config[ctx.my_mesh_id].entry_node_coord

    def _get_exit_coord(self, ctx: StageContext, pipeline_block):
        """Return the argmax final device MeshCoordinate for SpecLMHead."""
        return ctx.pipeline_config[ctx.my_mesh_id].exit_node_coord

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        my_stage_idx = ctx.my_stage_idx
        pipeline_config = ctx.pipeline_config
        entry_core = ttnn.MeshCoreCoord(
            pipeline_config[my_stage_idx].entry_node_coord,
            SpecLMHeadStage.LMHEAD_INPUT_CORE,
        )
        exit_core = ttnn.MeshCoreCoord(
            pipeline_config[my_stage_idx].exit_node_coord,
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
            my_stage_idx=my_stage_idx,
            stages_metadata=ctx.stages_metadata,
            pipeline_config=ctx.pipeline_config,
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        mesh_device = ctx.mesh_device
        my_stage_idx = ctx.my_stage_idx
        pipeline_config = ctx.pipeline_config

        # +32 for metadata (32 * 2 bytes = 64 bytes of metadata)
        torch_a = torch.zeros((SpecLMHeadStage.M, SpecLMHeadStage.K + METADATA_NUM_ELEMS), dtype=torch.bfloat16)
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
        sender_coord = self._get_sender_coord(ctx, pipeline_block)
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

        scores_scratch_shape, indices_scratch_shape = _mesh_sampling_scratch_shapes(mesh_rows, mesh_cols)
        scores_scratch_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(argmax_final_core_grid, scores_scratch_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )
        indices_scratch_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(argmax_final_core_grid, indices_scratch_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )
        scores_scratch_tensor = ttnn.from_torch(
            torch.zeros((num_devices, *scores_scratch_shape), dtype=torch.uint32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=scores_scratch_mem_config,
            mesh_mapper=mesh_mapper,
        )
        indices_scratch_tensor = ttnn.from_torch(
            torch.zeros((num_devices, *indices_scratch_shape), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=indices_scratch_mem_config,
            mesh_mapper=mesh_mapper,
        )

        # Metadata buffer on argmax_final_core (sized for the full DeepseekMetadata
        # struct: 64 B header from upstream unicast + 192 B for sampling.hpp's
        # local p_indices / p_scores writes).
        METADATA_ELEMS = METADATA_TENSOR_NUM_UINT32
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
        spec_b = self._spec_weights.lm_head
        spec_gamma = None
        self._state = {
            "input_tensor_mesh": input_tensor_mesh,
            "intermediate_tensor_mesh": intermediate_tensor_mesh,
            "ttnn_gamma": spec_gamma,
            "ttnn_b": spec_b,
            "ttnn_scores": ttnn_scores,
            "ttnn_indices": ttnn_indices,
            "ttnn_output_index": ttnn_output_index,
            "scores_scratch_tensor": scores_scratch_tensor,
            "indices_scratch_tensor": indices_scratch_tensor,
            "metadata_tensor": metadata_tensor,
            "lmhead_input_socket": pipeline_block.get_downstream_socket(),
            "lmhead_output_socket": pipeline_block.get_upstream_socket(),
            "bcast_semaphores": bcast_inputs.semaphores,
            "global_semaphore": ttnn.create_global_semaphore(mesh_device, argmax_final_core_grid, 0),
            "global_stage2_semaphore": ttnn.create_global_semaphore(mesh_device, argmax_final_core_grid, 0),
        }
        self._persistent_loop = PersistentLoop(mesh_device, worker_crs, self._persistent_mode)
        if self._persistent_mode:
            self._state["persistent_next_iter_semaphore"] = self._persistent_loop.next_iter_semaphore
            self._state["termination_semaphore"] = self._persistent_loop.termination_semaphore

    def run_auxiliary_sockets(self) -> None:
        pass

    def terminate_auxiliary(self) -> None:
        pass

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        d = self._state
        LMHeadSampling.op(
            d["input_tensor_mesh"],
            d["intermediate_tensor_mesh"],
            d["ttnn_gamma"],
            d["ttnn_b"],
            d["ttnn_scores"],
            sender_coord=self._get_sender_coord(ctx, pipeline_block),
            indices_tensor=d["ttnn_indices"],
            output_index_tensor=d["ttnn_output_index"],
            argmax_final_core_coord=SpecLMHeadStage.ARGMAX_FINAL_CORE,
            argmax_final_mesh_coord=self._get_exit_coord(ctx, pipeline_block),
            bcast_semaphores=d["bcast_semaphores"],
            global_semaphore=d["global_semaphore"],
            global_stage2_semaphore=d["global_stage2_semaphore"],
            scores_scratch_tensor=d["scores_scratch_tensor"],
            indices_scratch_tensor=d["indices_scratch_tensor"],
            fp32_dest_acc_en=self._fp32_dest_acc_en,
            skip_ccl=False,
            socket_input=d["lmhead_input_socket"],
            socket_output=d["lmhead_output_socket"],
            persistent_mode=self._persistent_mode,
            persistent_next_iter_semaphore=d.get("persistent_next_iter_semaphore"),
            termination_semaphore=d.get("termination_semaphore"),
            is_mtp_base_stage=False,
            is_mtp_verify_stage=True,
            metadata_tensor=d["metadata_tensor"],
            k=1,
        )

    def terminate(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        self._persistent_loop.terminate()


class BaseLMHeadStage(StageKind):
    """LMHead+Sampling stage: receive activation, run op, send token downstream."""

    # LMHead-stage-specific constants (tiles, core coords, matmul layout)
    M = 1
    K = ACTIVATION_DIM
    NUM_MATMUL_CORES = 101
    N_PER_CORE = 160
    N_TOTAL = NUM_MATMUL_CORES * N_PER_CORE
    A_TILE = ttnn.Tile([1, 32])
    OUT_TILE = ttnn.Tile([1, 32])
    ARGMAX_FINAL_CORE = ttnn.CoreCoord(0, 1)
    LMHEAD_INPUT_CORE = ttnn.CoreCoord(10, 9)

    def __init__(
        self,
        weights: DeepSeekV3LMHeadWeights,
        *,
        fp32_dest_acc_en: bool = True,
        persistent_mode: bool = True,
        mtp_weights: DeepSeekV3MTPWeights | None = None,
        embedding_weights: DeepSeekV3EmbeddingLayerWeights | None = None,
        send_mtp_output_downstream: bool = False,
        seed: int = 2005,
        upstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        downstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
    ) -> None:
        self._weights = weights
        self._fp32_dest_acc_en = fp32_dest_acc_en
        self._persistent_mode = persistent_mode
        self._mtp_weights = mtp_weights
        self._embedding_weights = embedding_weights
        self._enable_mtp = mtp_weights is not None
        if self._enable_mtp and self._embedding_weights is None:
            raise ValueError("embedding_weights are required when mtp_weights are provided")
        self._upstream_fifo_pages = upstream_fifo_pages
        self._downstream_fifo_pages = downstream_fifo_pages
        self._send_mtp_output_downstream = send_mtp_output_downstream and self._enable_mtp
        self._seed = seed
        self._lmhead_state: dict[str, Any] = {}

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        my_stage_idx = ctx.my_stage_idx
        pipeline_config = ctx.pipeline_config
        lmhead_entry_core = ttnn.MeshCoreCoord(
            pipeline_config[my_stage_idx].entry_node_coord, BaseLMHeadStage.LMHEAD_INPUT_CORE
        )
        lmhead_exit_core = ttnn.MeshCoreCoord(
            pipeline_config[my_stage_idx].exit_node_coord, BaseLMHeadStage.ARGMAX_FINAL_CORE
        )
        up_page = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES
        up_fifo = activation_fifo_size_bytes(up_page, self._upstream_fifo_pages)
        # MTP: forward activation+metadata downstream; non-MTP: only the token result goes downstream.
        if self._send_mtp_output_downstream:
            down_page = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES
            down_fifo = activation_fifo_size_bytes(down_page, self._downstream_fifo_pages)
        else:
            down_page = TOKEN_META_PAGE_SIZE_BYTES
            down_fifo = TOKEN_META_FIFO_SIZE
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=up_fifo,
            downstream_d2d_socket_fifo_size=down_fifo,
            upstream_d2d_socket_page_size=up_page,
            downstream_d2d_socket_page_size=down_page,
            entry_node_downstream=lmhead_entry_core,
            exit_node_upstream=lmhead_exit_core,
            my_stage_idx=my_stage_idx,
            stages_metadata=ctx.stages_metadata,
            pipeline_config=ctx.pipeline_config,
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
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
        ttnn_gamma = None
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
        scores_scratch_shape, indices_scratch_shape = _mesh_sampling_scratch_shapes(mesh_rows, mesh_cols)
        scores_scratch_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                argmax_final_core_grid,
                scores_scratch_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        indices_scratch_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                argmax_final_core_grid,
                indices_scratch_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        scores_scratch_tensor = ttnn.from_torch(
            torch.zeros((num_devices, *scores_scratch_shape), dtype=torch.uint32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=scores_scratch_mem_config,
            mesh_mapper=mesh_mapper,
        )
        indices_scratch_tensor = ttnn.from_torch(
            torch.zeros((num_devices, *indices_scratch_shape), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=indices_scratch_mem_config,
            mesh_mapper=mesh_mapper,
        )

        base_token_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(mcast_core_grid, (1, 8), ttnn.ShardOrientation.ROW_MAJOR),
        )
        base_token_buffer = ttnn.from_torch(
            torch.zeros((num_devices, 1, 8), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=base_token_mem_config,
            mesh_mapper=mesh_mapper,
        )

        METADATA_ELEMS = METADATA_TENSOR_NUM_UINT32
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

        # MTP fused buffer: single allocation backing CB17 (matmul output), CB20 (reduce
        # intermediate), and CB19 (eh_gather / reduce output).  CB17+CB20 live on compute
        # cores at different offsets (both active during reduce); CB19 lives on the argmax
        # core.  Per-core shard is the max of (cb17+cb20) and cb19.
        eh_mm_fused_buffer = None
        if self._enable_mtp:
            compute_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(mesh_device, ttnn.NOC.NOC_0)
            compute_core_grid = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
            )

            out_tile_w = BaseLMHeadStage.OUT_TILE.tile_shape[1]
            eh_out_w_per_core = mtp_n_per_core // out_tile_w
            eh_gather_total_tiles = num_dram_banks * eh_out_w_per_core + 1

            compute_core_width = mtp_n_per_core + 3 * mtp_n_per_core  # cb17 + cb20
            argmax_core_width = eh_gather_total_tiles * out_tile_w
            eh_mm_shard_width = max(compute_core_width, argmax_core_width)

            eh_mm_fused_grid = compute_core_grid.merge(argmax_final_core_grid)
            num_eh_mm_fused_cores = eh_mm_fused_grid.num_cores()
            eh_mm_fused_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    eh_mm_fused_grid,
                    (BaseLMHeadStage.M, 15360),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            eh_mm_fused_buffer = ttnn.from_torch(
                torch.zeros(
                    (num_devices, BaseLMHeadStage.M, 15360 * num_eh_mm_fused_cores),  # clean up later
                    dtype=torch.bfloat16,
                ),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=eh_mm_fused_mem,
                tile=BaseLMHeadStage.OUT_TILE,
                mesh_mapper=mesh_mapper,
            )

            dg = mesh_device.compute_with_storage_grid_size()
            reduce_sem_crs = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))]
            )
            reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, reduce_sem_crs, 0) for _ in range(4)]

        lmhead_input_socket = pipeline_block.get_downstream_socket() if pipeline_block.has_exit else None
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
            "scores_scratch_tensor": scores_scratch_tensor,
            "indices_scratch_tensor": indices_scratch_tensor,
            "metadata_tensor": metadata_tensor,
            "lmhead_input_socket": lmhead_input_socket,
            "lmhead_output_socket": lmhead_output_socket,
            "bcast_semaphores": bcast_inputs.semaphores,
            "global_semaphore": global_semaphore,
            "global_stage2_semaphore": global_stage2_semaphore,
            "base_token_buffer": base_token_buffer,
        }
        if self._enable_mtp:
            ttnn_h_gamma = None
            ttnn_e_gamma = None
            self._lmhead_state["eh_mm_fused_buffer"] = eh_mm_fused_buffer
            self._lmhead_state["ttnn_embedding"] = self._embedding_weights.embedding  # replicated on each device
            self._lmhead_state["ttnn_eh_proj"] = self._mtp_weights.eh_projection  # width sharded on each device
            self._lmhead_state["reduce_semaphores"] = reduce_semaphores
            self._lmhead_state["ttnn_h_gamma"] = ttnn_h_gamma
            self._lmhead_state["ttnn_e_gamma"] = ttnn_e_gamma
            compute_grid_size = mesh_device.compute_with_storage_grid_size()
            num_cores = compute_grid_size.x * compute_grid_size.y
            available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)
            self._lmhead_state["mtp_bcast_semaphores"] = [ttnn.create_global_semaphore(mesh_device, available_cores, 0)]
        self._persistent_loop = PersistentLoop(mesh_device, worker_crs, self._persistent_mode)
        if self._persistent_mode:
            self._lmhead_state["persistent_next_iter_semaphore"] = self._persistent_loop.next_iter_semaphore
            self._lmhead_state["termination_semaphore"] = self._persistent_loop.termination_semaphore

    def run_auxiliary_sockets(self) -> None:
        pass

    def terminate_auxiliary(self) -> None:
        pass

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        d = self._lmhead_state
        pipeline_config = ctx.pipeline_config
        my_stage_idx = ctx.my_stage_idx
        LMHeadSampling.op(
            d["input_tensor_mesh"],
            d["intermediate_tensor_mesh"],
            d["ttnn_gamma"],
            d["ttnn_b"],
            d["ttnn_scores"],
            sender_coord=pipeline_config[my_stage_idx].entry_node_coord,
            eh_mm_fused_buffer=d.get("eh_mm_fused_buffer"),
            embedding_tensor=d.get("ttnn_embedding"),
            h_gamma_tensor=d.get("ttnn_h_gamma"),
            e_gamma_tensor=d.get("ttnn_e_gamma"),
            eh_projection_tensor=d.get("ttnn_eh_proj"),
            indices_tensor=d["ttnn_indices"],
            output_index_tensor=d["ttnn_output_index"],
            argmax_final_core_coord=BaseLMHeadStage.ARGMAX_FINAL_CORE,
            argmax_final_mesh_coord=pipeline_config[my_stage_idx].exit_node_coord,
            bcast_semaphores=d["bcast_semaphores"],
            global_semaphore=d["global_semaphore"],
            global_stage2_semaphore=d["global_stage2_semaphore"],
            scores_scratch_tensor=d["scores_scratch_tensor"],
            indices_scratch_tensor=d["indices_scratch_tensor"],
            fp32_dest_acc_en=self._fp32_dest_acc_en,
            skip_ccl=False,
            socket_input=d["lmhead_input_socket"],
            socket_output=d["lmhead_output_socket"],
            persistent_mode=self._persistent_mode,
            persistent_next_iter_semaphore=d.get("persistent_next_iter_semaphore"),
            termination_semaphore=d.get("termination_semaphore"),
            is_mtp_base_stage=True,
            metadata_tensor=d.get("metadata_tensor"),
            reduce_semaphores=d.get("reduce_semaphores"),
            mtp_bcast_semaphores=d.get("mtp_bcast_semaphores"),
            base_token_buffer=d.get("base_token_buffer"),
            seed=self._seed,
        )

    def terminate(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        self._persistent_loop.terminate()


class _CombinedPipelineBlock:
    """Pipeline block for combined SpecLMHead + Embedding stage.

    Wires three independent socket paths on the same mesh:
    - H2D (token from host) -> fused embedding -> exit D2D (activation to P1)
    - Entry D2D (ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES from P3) -> SpecLMHead input (LMHEAD_INPUT_CORE)
    - SpecLMHead output (ARGMAX_FINAL_CORE) -> D2H (TOKEN_META to host)

    The broadcast sender and argmax final device are co-located with the
    loopback entry/exit coordinates respectively, avoiding inter-device relays.

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
        *,
        loopback_input_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        my_stage_idx: int | None = None,
        pipeline_config: list | None = None,
        stages_metadata: dict[int, StageMetadata] | None = None,
    ) -> None:
        if my_stage_idx is None:
            my_stage_idx = mesh_device.get_system_mesh_id()
        num_procs = int(ttnn.distributed_context_get_size())
        if pipeline_config is None:
            pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
        assert len(pipeline_config) == num_procs + 1, "Pipeline config must include loopback entry"

        def stage_mesh_id(stage_idx: int) -> int:
            if stages_metadata is not None:
                return stages_metadata[stage_idx].mesh_id
            return stage_idx

        my_mesh_id = stage_mesh_id(my_stage_idx)
        next_stage_mesh_id = stage_mesh_id(my_stage_idx + 1)
        prev_stage_mesh_id = stage_mesh_id(num_procs - 1)

        exit_node_coord = pipeline_config[my_stage_idx].exit_node_coord
        loopback_entry_coord = pipeline_config[num_procs].entry_node_coord
        loopback_exit_coord = pipeline_config[num_procs].exit_node_coord
        next_stage_entry_coord = pipeline_config[my_stage_idx + 1].entry_node_coord
        prev_stage_exit_coord = pipeline_config[num_procs - 1].exit_node_coord

        logger.debug(
            f"[COMBINED P{my_stage_idx}] exit_node_coord={exit_node_coord} loopback_entry_coord={loopback_entry_coord} loopback_exit_coord={loopback_exit_coord} next_stage_entry_coord={next_stage_entry_coord} prev_stage_exit_coord={prev_stage_exit_coord}",
        )

        embedding_size_bytes = embedding_tensor.shape[-1] * 2  # bfloat16
        assert ACTIVATION_PAGE_SIZE_BYTES == embedding_size_bytes

        # -- H2D path (embedding) --
        # H2D socket lives on the exit node so the fused embedding kernel can
        # forward data directly to the next mesh over an inter-mesh MeshSocket,
        # eliminating the previous intra-mesh relay hop.
        self.h2d_socket = ttnn.H2DSocket(
            mesh_device,
            ttnn.MeshCoreCoord(exit_node_coord, PIPELINE_CORE_COORD),
            ttnn.BufferType.L1,
            TOKEN_META_FIFO_SIZE,
            ttnn.H2DMode.HOST_PUSH,
        )

        self.h2d_host_io = HostInterface(
            self.h2d_socket,
            None,
            TOKEN_META_PAGE_SIZE_BYTES,
            0,
            core_to_core_socket_buffer_size=ACTIVATION_W_TOKEN_META_FIFO_SIZE,
            h2d_downstream_core=ttnn.MeshCoreCoord(next_stage_entry_coord, PIPELINE_CORE_COORD),
            embedding_tensor=embedding_tensor,
            metadata_size_bytes=TOKEN_META_PAGE_SIZE_BYTES,
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=next_stage_mesh_id),
        )

        # -- SpecLMHead input path (loopback entry from P3) --
        spec_root_device_coord = loopback_entry_coord
        self.spec_entry_coord = spec_root_device_coord
        loopback_input_fifo_size = activation_fifo_size_bytes(
            ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES, loopback_input_fifo_pages
        )
        self.entry_socket_interface = SocketInterface(
            ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES,
            loopback_input_fifo_size,
            ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES,
            ttnn.MeshCoreCoord(prev_stage_exit_coord, PIPELINE_CORE_COORD),
            ttnn.MeshCoreCoord(loopback_entry_coord, PIPELINE_CORE_COORD),
            downstream_core_coord=ttnn.MeshCoreCoord(spec_root_device_coord, lmhead_input_core),
            sender_mesh=MeshWrapper(mesh_id=prev_stage_mesh_id),
            receiver_mesh=MeshWrapper(mesh_device),
        )

        # -- D2H path (argmax output → D2H on same device, no inter-device relay needed) --
        spec_exit_device_coord = loopback_exit_coord
        self.spec_exit_coord = spec_exit_device_coord

        self.d2h_socket = ttnn.D2HSocket(
            mesh_device,
            ttnn.MeshCoreCoord(loopback_exit_coord, EMBEDDING_D2H_CORE_COORD),
            TOKEN_META_FIFO_SIZE,
        )

        self.d2h_host_io = HostInterface(
            None,
            self.d2h_socket,
            0,
            TOKEN_META_PAGE_SIZE_BYTES,
            core_to_core_socket_buffer_size=TOKEN_META_FIFO_SIZE,
            d2h_upstream_core=ttnn.MeshCoreCoord(spec_exit_device_coord, argmax_final_core),
            metadata_size_bytes=TOKEN_META_PAGE_SIZE_BYTES,
        )

        logger.debug(
            f"[COMBINED P{my_stage_idx}] _CombinedPipelineBlock created: "
            f"exit_dev={exit_node_coord} "
            f"spec_root={spec_root_device_coord} spec_exit={spec_exit_device_coord} "
            f"d2h_dev={loopback_exit_coord}",
        )

    def export_host_socket_descriptors(self, io_socket_descriptor_prefix: str = "deepseek") -> None:
        assert self.h2d_socket is not None, "Expected H2D socket on the first pipeline stage"
        assert self.d2h_socket is not None, "Expected D2H socket on the first pipeline stage"
        self.h2d_socket.export_descriptor(f"{io_socket_descriptor_prefix}_h2d")
        self.d2h_socket.export_descriptor(f"{io_socket_descriptor_prefix}_d2h")

    def run(self) -> None:
        self.h2d_host_io.run()
        self.d2h_host_io.run()
        self.entry_socket_interface.run()

    def terminate(self) -> None:
        ttnn.distributed_context_barrier()
        self.h2d_host_io.terminate(False)
        self.entry_socket_interface.terminate(False)
        self.d2h_host_io.terminate(True)

    def is_first_pipeline_stage(self) -> bool:
        return True

    def write_token(self, token_tensor) -> None:
        self.h2d_socket.write_tensor(token_tensor)

    def read_output(self, output_tensor) -> None:
        self.d2h_socket.read_tensor(output_tensor)

    def push_dummy_token(self) -> None:
        """Push a single zeroed token through the H2D socket. See :meth:`PipelineBlock.push_dummy_token`."""
        page_words = TOKEN_META_PAGE_SIZE_BYTES // 4
        dummy = ttnn.from_torch(
            torch.zeros(1, page_words, dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.h2d_socket.write_tensor(dummy)

    def drain_dummy_output(self) -> None:
        """Drain one page from the D2H socket. See :meth:`PipelineBlock.drain_dummy_output`."""
        page_words = TOKEN_META_PAGE_SIZE_BYTES // 4
        sink = ttnn.from_torch(
            torch.zeros(1, page_words, dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.d2h_socket.read_tensor(sink)

    def get_downstream_socket(self):
        """SpecLMHead reads activation+metadata from this socket (loopback entry downstream)."""
        return self.entry_socket_interface.get_downstream_socket()

    def get_upstream_socket(self):
        """SpecLMHead writes verification result into this socket (directly to D2H)."""
        return self.d2h_host_io.get_upstream_socket()


class SpecLMHeadWithEmbeddingStage(SpecLMHeadStage):
    """Combined SpecLMHead + Embedding on the same mesh.

    SpecLMHead occupies (0,0)-(10,9).  Embedding I/O uses column 12:
      H2D at PIPELINE_CORE_COORD, D2H at EMBEDDING_D2H_CORE_COORD,
      argmax final at ARGMAX_FINAL_CORE.  Exit D2D relay uses PIPELINE_CORE_COORD.

    Pipeline topology:
      P0(this) -> P1(BaseLMHead+MTP) -> P2(Passthrough) -> P3(Passthrough) -> back to P0
    """

    def __init__(
        self,
        embedding_weights: DeepSeekV3EmbeddingLayerWeights,
        *,
        fp32_dest_acc_en: bool = True,
        persistent_mode: bool = True,
        spec_weights: DeepSeekV3SpecWeights | None = None,
        loopback_input_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
    ) -> None:
        super().__init__(
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            spec_weights=spec_weights,
        )
        self._embedding_weights = embedding_weights
        self._loopback_input_fifo_pages = loopback_input_fifo_pages

    def _get_sender_coord(self, ctx: StageContext, pipeline_block):
        return pipeline_block.spec_entry_coord

    def _get_exit_coord(self, ctx: StageContext, pipeline_block):
        return pipeline_block.spec_exit_coord

    def create_pipeline_block(self, ctx: StageContext) -> _CombinedPipelineBlock:
        return _CombinedPipelineBlock(
            ctx.mesh_device,
            self._embedding_weights.embedding,
            SpecLMHeadStage.LMHEAD_INPUT_CORE,
            SpecLMHeadStage.ARGMAX_FINAL_CORE,
            loopback_input_fifo_pages=self._loopback_input_fifo_pages,
            my_stage_idx=ctx.my_stage_idx,
            pipeline_config=ctx.pipeline_config,
            stages_metadata=ctx.stages_metadata,
        )
