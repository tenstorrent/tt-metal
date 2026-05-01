# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline orchestration: PipelineConfiguration + Pipeline + create_fabric_router_config.

Vendored from `models.demos.deepseek_v3_b1.demo.pipeline`. The configuration-builder
helpers (create_single_galaxy_*, create_single_pod_*, create_sp4_*) and their
DenseDecoderStage/MoEDecoderStage/LMHeadStage/WeightProvider deps are intentionally
removed — the surviving tests build their own PipelineConfiguration directly.
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger

import ttnn
from .stage import StageContext, StageKind
from .pipeline_block import PipelineBlock


def create_fabric_router_config(max_payload_size: int) -> Any:
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


class PipelineConfiguration:
    """Maps stage IDs to stage factories. Each host builds only its stage (lazy weights)."""

    def __init__(
        self,
        stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]],
    ) -> None:
        self._stage_factories = stage_factories

    @property
    def num_stages(self) -> int:
        return len(self._stage_factories)

    def build_pipeline(self, mesh_device: ttnn.MeshDevice) -> Pipeline:
        """Create a Pipeline for this process's stage (determined by mesh_id)."""
        my_mesh_id = mesh_device.get_system_mesh_id()
        stage = self._stage_factories[my_mesh_id](mesh_device)
        return Pipeline(mesh_device, stage)


class Pipeline:
    """Orchestrator for one pipeline stage with explicit 4-phase setup."""

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

    def configure_block(self) -> None:
        """Phase 1: Create the PipelineBlock (socket wiring)."""
        self._pipeline_block = self._stage_kind.create_pipeline_block(self._ctx)

    def setup(self) -> None:
        """Phase 2: Allocate tensors, weights, semaphores on device.

        Decoder/dense stages also build :meth:`DecoderBlock.get_program_context` here so
        program construction finishes before :meth:`start_pipeline`.
        """
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.configure_block() must be called before setup()")
        self._stage_kind.setup(self._ctx, self._pipeline_block)

    def start_pipeline(self) -> None:
        """Phase 3: Start pipeline block kernels (socket interfaces)."""
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.configure_block() must be called before start_pipeline()")
        self._pipeline_block.run()

    def start_compute(self) -> None:
        """Phase 4: Launch stage compute (e.g. ``LMHeadSampling.op``, ``DecoderBlock.execute``)."""
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.configure_block() must be called before start_compute()")
        self._stage_kind.launch_compute(self._ctx, self._pipeline_block)

    def setup_and_run(self) -> None:
        """Run all four phases in order."""
        logger.info("Configuring block")
        self.configure_block()

        logger.info("Setting up")
        self.setup()
        logger.info("Pipeline setup complete, waiting for all stages to complete...")
        self.barrier()

        logger.info("Starting pipeline")
        self.start_pipeline()
        logger.info("Pipeline started, waiting for all stages to complete...")
        self.barrier()

        logger.info("Starting compute")
        self.start_compute()
        logger.info("Compute started, waiting for all stages to complete...")
        self.barrier()

    def write_token(self, token_tensor: ttnn.Tensor) -> None:
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.setup_and_run() or configure_block() must be called first")
        self._pipeline_block.write_token(token_tensor)

    def read_output(self, output_tensor: ttnn.Tensor) -> None:
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.setup_and_run() or configure_block() must be called first")
        self._pipeline_block.read_output(output_tensor)

    def export_host_socket_descriptors(self, prefix: str) -> None:
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.setup_and_run() must complete before exporting host socket descriptors")
        self._pipeline_block.export_host_socket_descriptors(prefix)

    def barrier(self) -> None:
        ttnn.distributed_context_barrier()

    def terminate(self) -> None:
        """Terminate the pipeline block if it was created (e.g. for one-shot tests)."""
        if self._pipeline_block is not None:
            self._pipeline_block.terminate()
