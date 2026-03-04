# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline orchestration: configuration, factory functions, Pipeline.
Stage kinds (Embedding, LMHead, Passthrough) live in stage.py.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import ttnn
from models.demos.deepseek_v3_b1.demo.stage import (
    EmbeddingStage,
    LMHeadStage,
    PassthroughPayload,
    PassthroughStage,
    StageContext,
    StageKind,
)
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.prepare_weights import DeepSeekV3EmbeddingLayerWeights, DeepSeekV3LMHeadWeights


class WeightProvider(Protocol):
    """Provides embedding and LM head weights on demand; each host loads only what its stage needs."""

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        ...

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        ...


def create_fabric_router_config(max_payload_size: int) -> Any:
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def create_single_galaxy_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
) -> PipelineConfiguration:
    """4-stage single-galaxy: Embed -> LMHead -> Token fwd -> Token fwd."""

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return EmbeddingStage(weight_provider.load_embedding(device))

    def stage_1(device: ttnn.MeshDevice) -> StageKind:
        return LMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
        )

    return PipelineConfiguration(
        {
            0: stage_0,
            1: stage_1,
            2: lambda d: PassthroughStage(PassthroughPayload.TOKEN),
            3: lambda d: PassthroughStage(PassthroughPayload.TOKEN),
        }
    )


def create_single_pod_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
) -> PipelineConfiguration:
    """16-stage single-pod: Embed -> 13x Activation fwd -> LMHead -> Token fwd."""

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return EmbeddingStage(weight_provider.load_embedding(device))

    def stage_14(device: ttnn.MeshDevice) -> StageKind:
        return LMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
        )

    passthrough_activation = lambda d: PassthroughStage(PassthroughPayload.ACTIVATION)
    stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]] = {
        0: stage_0,
        **{i: passthrough_activation for i in range(1, 14)},
        14: stage_14,
        15: lambda d: PassthroughStage(PassthroughPayload.TOKEN),
    }
    return PipelineConfiguration(stage_factories)


def create_pipeline_configuration_from_num_procs(
    num_procs: int,
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
) -> PipelineConfiguration:
    """Pick topology from process count (4 -> single_galaxy, 16 -> single_pod)."""
    if num_procs == 4:
        return create_single_galaxy_pipeline_configuration(
            weight_provider,
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
        )
    if num_procs == 16:
        return create_single_pod_pipeline_configuration(
            weight_provider,
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
        )
    raise ValueError(f"Unsupported num_procs: {num_procs}")


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
        """Phase 2: Allocate tensors, weights, semaphores on device."""
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.configure_block() must be called before setup()")
        self._stage_kind.setup(self._ctx, self._pipeline_block)

    def start_pipeline(self) -> None:
        """Phase 3: Start pipeline block kernels (socket interfaces)."""
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.configure_block() must be called before start_pipeline()")
        self._pipeline_block.run()

    def start_compute(self) -> None:
        """Phase 4: Launch stage compute (e.g. LMHeadSampling.op)."""
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.configure_block() must be called before start_compute()")
        self._stage_kind.launch_compute(self._ctx, self._pipeline_block)

    def setup_and_run(self) -> None:
        """Run all four phases in order."""
        self.configure_block()
        self.setup()
        self.start_pipeline()
        self.start_compute()

    def write_token(self, token_tensor: ttnn.Tensor) -> None:
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.setup_and_run() or configure_block() must be called first")
        self._pipeline_block.write_token(token_tensor)

    def read_output(self, output_tensor: ttnn.Tensor) -> None:
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.setup_and_run() or configure_block() must be called first")
        self._pipeline_block.read_output(output_tensor)

    def barrier(self) -> None:
        ttnn.distributed_context_barrier()

    def terminate(self) -> None:
        """Terminate the pipeline block if it was created (e.g. for one-shot tests)."""
        if self._pipeline_block is not None:
            self._pipeline_block.terminate()
