# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline orchestration: configuration, factory functions, Pipeline.
Stage kinds (Embedding, LMHead, Passthrough) live in stage.py.
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.demo.monitor import StageMonitor
from models.demos.deepseek_v3_b1.demo.stage import (
    EmbeddingStage,
    LMHeadStage,
    PassthroughPayload,
    PassthroughStage,
    StageContext,
    StageInfo,
    StageKind,
    StagePhase,
)
from models.demos.deepseek_v3_b1.demo.weight_provider import WeightProvider
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.tests.unit_tests.test_decoder_block_api import DecoderBlockStage, DenseBlockStage


def create_fabric_router_config(max_payload_size: int) -> Any:
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def create_single_galaxy_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    lm_head_fp32_dest_acc_en: bool = True,
    lm_head_persistent_mode: bool = True,
) -> PipelineConfiguration:
    """4-stage single-galaxy: Embed -> LMHead -> Token fwd -> Token fwd."""

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return EmbeddingStage(weight_provider.load_embedding(device))

    def stage_1(device: ttnn.MeshDevice) -> StageKind:
        return LMHeadStage(
            weights=weight_provider.load_lm_head(device),
            lm_head_fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode=lm_head_persistent_mode,
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
    lm_head_fp32_dest_acc_en: bool = True,
    lm_head_persistent_mode: bool = True,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
) -> PipelineConfiguration:
    """16-stage single-pod: Embed -> Dense(0,1,2) -> Decoder(3..12) -> LMHead -> Token fwd.

    If dense_layer_id_override is set (e.g. 0), all dense stages use that layer id.
    If moe_layer_id_override is set (e.g. 3), all decoder stages use that layer id.
    """

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return EmbeddingStage(weight_provider.load_embedding(device))

    def stage_14(device: ttnn.MeshDevice) -> StageKind:
        return LMHeadStage(
            weights=weight_provider.load_lm_head(device),
            lm_head_fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode=lm_head_persistent_mode,
        )

    # Same layout as SP4: stage i -> layer_id i-1 for decoder stages; fewer MoE stages (4-13 = layers 3-12)
    def _dense_stage(layer_id: int):
        return lambda d: DenseBlockStage(
            weights=weight_provider.load_dense_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
        )

    def _decoder_stage(layer_id: int):
        return lambda d: DecoderBlockStage(
            weights=weight_provider.load_moe_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
        )

    dense_ids = (dense_layer_id_override,) * 3 if dense_layer_id_override is not None else (0, 1, 2)
    moe_layer_id = moe_layer_id_override if moe_layer_id_override is not None else None

    stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]] = {
        0: stage_0,
        1: _dense_stage(dense_ids[0]),
        2: _dense_stage(dense_ids[1]),
        3: _dense_stage(dense_ids[2]),
        **{i: _decoder_stage(moe_layer_id if moe_layer_id is not None else i - 1) for i in range(4, 14)},
        14: stage_14,
        15: lambda d: PassthroughStage(PassthroughPayload.TOKEN),
    }
    return PipelineConfiguration(stage_factories)


def create_sp4_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    lm_head_fp32_dest_acc_en: bool = True,
    lm_head_persistent_mode: bool = True,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
) -> PipelineConfiguration:
    """64-stage super-pod: Embed -> Dense(0,1,2) -> Decoder(3..60) -> LMHead -> Token fwd.

    If dense_layer_id_override is set (e.g. 0), all dense stages use that layer id.
    If moe_layer_id_override is set (e.g. 3), all decoder stages use that layer id.
    """

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return EmbeddingStage(weight_provider.load_embedding(device))

    def stage_62(device: ttnn.MeshDevice) -> StageKind:
        return LMHeadStage(
            weights=weight_provider.load_lm_head(device),
            lm_head_fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode=lm_head_persistent_mode,
        )

    # Stage i -> layer_id i-1 for decoder stages (stage 1 = layer 0, ..., stage 61 = layer 60)
    def _dense_stage(layer_id: int):
        return lambda d: DenseBlockStage(
            weights=weight_provider.load_dense_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
        )

    def _decoder_stage(layer_id: int):
        return lambda d: DecoderBlockStage(
            weights=weight_provider.load_moe_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
        )

    dense_ids = (dense_layer_id_override,) * 3 if dense_layer_id_override is not None else (0, 1, 2)
    moe_layer_id = moe_layer_id_override if moe_layer_id_override is not None else None

    stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]] = {
        0: stage_0,
        1: _dense_stage(dense_ids[0]),
        2: _dense_stage(dense_ids[1]),
        3: _dense_stage(dense_ids[2]),
        **{i: _decoder_stage(moe_layer_id if moe_layer_id is not None else i - 1) for i in range(4, 62)},
        62: stage_62,
        63: lambda d: PassthroughStage(PassthroughPayload.TOKEN),
    }
    return PipelineConfiguration(stage_factories)


def create_pipeline_configuration_from_num_procs(
    num_procs: int,
    weight_provider: WeightProvider,
    *,
    lm_head_fp32_dest_acc_en: bool = True,
    lm_head_persistent_mode: bool = True,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
) -> PipelineConfiguration:
    """Pick topology from process count (4 -> single_galaxy, 16 -> single_pod, 64 -> sp4)."""
    if num_procs == 4:
        return create_single_galaxy_pipeline_configuration(
            weight_provider,
            lm_head_fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode=lm_head_persistent_mode,
        )
    if num_procs == 16:
        return create_single_pod_pipeline_configuration(
            weight_provider,
            lm_head_fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode=lm_head_persistent_mode,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
        )
    if num_procs == 64:
        return create_sp4_pipeline_configuration(
            weight_provider,
            lm_head_fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode=lm_head_persistent_mode,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
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

    def build_pipeline(self, mesh_device: ttnn.MeshDevice, *, monitor_port: int | None = None) -> Pipeline:
        """Create a Pipeline for this process's stage (determined by mesh_id).

        If *monitor_port* is not None, a lightweight HTTP monitor is started on
        ``monitor_port + mesh_id``.
        """
        my_mesh_id = mesh_device.get_system_mesh_id()
        stage = self._stage_factories[my_mesh_id](mesh_device)
        return Pipeline(mesh_device, stage, monitor_port=monitor_port)


class Pipeline:
    """Orchestrator for one pipeline stage with explicit 4-phase setup."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        stage_kind: StageKind,
        *,
        monitor_port: int | None = None,
    ) -> None:
        self._mesh_device = mesh_device
        self._stage_kind = stage_kind
        self._my_mesh_id = mesh_device.get_system_mesh_id()
        self._pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)

        self._info = StageInfo(
            mesh_id=self._my_mesh_id,
            stage_type=type(stage_kind).__name__,
        )
        self._ctx = StageContext(
            mesh_device=mesh_device,
            pipeline_config=self._pipeline_config,
            my_mesh_id=self._my_mesh_id,
            info=self._info,
        )
        self._pipeline_block: PipelineBlock | None = None

        self._monitor: StageMonitor | None = None
        if monitor_port is not None:
            self._monitor = StageMonitor(self._info, base_port=monitor_port)
            self._monitor.start()

    @property
    def my_mesh_id(self) -> int:
        return self._my_mesh_id

    @property
    def info(self) -> StageInfo:
        return self._info

    def configure_block(self) -> None:
        """Phase 1: Create the PipelineBlock (socket wiring)."""
        self._info.transition(StagePhase.CONFIGURING)
        self._pipeline_block = self._stage_kind.create_pipeline_block(self._ctx)
        self._info.transition(StagePhase.CONFIGURED)

    def setup(self) -> None:
        """Phase 2: Allocate tensors, weights, semaphores on device."""
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.configure_block() must be called before setup()")
        self._info.transition(StagePhase.SETTING_UP)
        self._stage_kind.setup(self._ctx, self._pipeline_block)
        self._info.transition(StagePhase.READY)

    def start_pipeline(self) -> None:
        """Phase 3: Start pipeline block kernels (socket interfaces)."""
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.configure_block() must be called before start_pipeline()")
        self._info.transition(StagePhase.PIPELINE_STARTING)
        self._pipeline_block.run()
        self._info.transition(StagePhase.PIPELINE_RUNNING)

    def start_compute(self) -> None:
        """Phase 4: Launch stage compute (e.g. LMHeadSampling.op)."""
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.configure_block() must be called before start_compute()")
        self._info.transition(StagePhase.COMPUTING)
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

    def barrier(self) -> None:
        ttnn.distributed_context_barrier()

    def terminate(self) -> None:
        """Terminate the pipeline block if it was created (e.g. for one-shot tests)."""
        self._info.transition(StagePhase.TERMINATED)
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
        if self._pipeline_block is not None:
            self._pipeline_block.terminate()
