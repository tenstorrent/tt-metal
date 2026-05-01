# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage kinds for the pod pipeline: Embedding and Passthrough.

LMHeadStage is intentionally NOT vendored — the surviving tests use
`FakeLMHeadStage` from `_fake_moe_helpers` instead, so the full LMHead
machinery (LMHeadSampling op, broadcast test inputs) is not needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import ttnn
from .pipeline_block import PipelineBlock


# Minimal stand-in for the demo's `DeepSeekV3EmbeddingLayerWeights` —
# `EmbeddingStage` only reads `.embedding`, so a one-field dataclass suffices.
@dataclass
class DeepSeekV3EmbeddingLayerWeights:
    embedding: ttnn.Tensor


# Global constants used by multiple stage kinds (and exported to pipeline/cli)
TOKEN_PAGE_SIZE_BYTES = 64
TOKEN_FIFO_NUM_PAGES = 64
TOKEN_FIFO_SIZE = TOKEN_PAGE_SIZE_BYTES * TOKEN_FIFO_NUM_PAGES
ACTIVATION_DIM = 7168
ACTIVATION_PAGE_SIZE_BYTES = ACTIVATION_DIM * 2
ACTIVATION_FIFO_SIZE = ACTIVATION_PAGE_SIZE_BYTES * 2
PIPELINE_CORE_COORD = ttnn.CoreCoord(12, 8)


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

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        """Run stage compute after ``pipeline_block.run()`` (execute pre-built programs where applicable). Default: no-op."""


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
        )
