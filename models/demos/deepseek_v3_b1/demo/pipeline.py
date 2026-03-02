# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline orchestration: configuration, factory functions, Pipeline.
Stage kinds (Embedding, LMHead, Passthrough) live in stage.py.
"""

from __future__ import annotations

from typing import Any

import torch

import ttnn
from models.demos.deepseek_v3_b1.demo.stage import (
    EmbeddingStage,
    K,
    LMHeadStage,
    LMHeadWeights,
    M,
    PassthroughPayload,
    PassthroughStage,
    StageContext,
    StageKind,
    n_total,
)
from models.demos.deepseek_v3_b1.fused_ops.lm_head_sampling.op import LMHeadSampling
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock


def create_fabric_router_config(max_payload_size: int) -> Any:
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


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


def create_single_galaxy_pipeline_configuration(
    *,
    embedding_tensor: torch.Tensor,
    lmhead_weights: LMHeadWeights,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
) -> PipelineConfiguration:
    """4-stage single-galaxy: Embed -> LMHead -> Token fwd -> Token fwd."""
    return PipelineConfiguration(
        {
            0: EmbeddingStage(embedding_tensor),
            1: LMHeadStage(
                weights=lmhead_weights,
                fp32_dest_acc_en=fp32_dest_acc_en,
                persistent_mode=persistent_mode,
            ),
            2: PassthroughStage(PassthroughPayload.TOKEN),
            3: PassthroughStage(PassthroughPayload.TOKEN),
        }
    )


def create_single_pod_pipeline_configuration(
    *,
    embedding_tensor: torch.Tensor,
    lmhead_weights: LMHeadWeights,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
) -> PipelineConfiguration:
    """16-stage single-pod: Embed -> 13x Activation fwd -> LMHead -> Token fwd."""
    stages: dict[int, StageKind] = {0: EmbeddingStage(embedding_tensor)}
    for i in range(1, 14):
        stages[i] = PassthroughStage(PassthroughPayload.ACTIVATION)
    stages[14] = LMHeadStage(
        weights=lmhead_weights,
        fp32_dest_acc_en=fp32_dest_acc_en,
        persistent_mode=persistent_mode,
    )
    stages[15] = PassthroughStage(PassthroughPayload.TOKEN)
    return PipelineConfiguration(stages)


def create_pipeline_configuration_from_num_procs(
    num_procs: int,
    *,
    embedding_tensor: torch.Tensor,
    lmhead_weights: LMHeadWeights,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
) -> PipelineConfiguration:
    """Pick topology from process count (4 -> single_galaxy, 16 -> single_pod)."""
    if num_procs == 4:
        return create_single_galaxy_pipeline_configuration(
            embedding_tensor=embedding_tensor,
            lmhead_weights=lmhead_weights,
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
        )
    if num_procs == 16:
        return create_single_pod_pipeline_configuration(
            embedding_tensor=embedding_tensor,
            lmhead_weights=lmhead_weights,
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
        )
    raise ValueError(f"Unsupported num_procs: {num_procs}")


class PipelineConfiguration:
    """Maps stage IDs to StageKinds. The full pipeline definition."""

    def __init__(self, stages: dict[int, StageKind]) -> None:
        self._stages = stages

    @property
    def num_stages(self) -> int:
        return len(self._stages)

    def __getitem__(self, stage_id: int) -> StageKind:
        return self._stages[stage_id]

    def build_pipeline(self, mesh_device: ttnn.MeshDevice) -> Pipeline:
        """Create a Pipeline for this process's stage (determined by mesh_id)."""
        my_mesh_id = mesh_device.get_system_mesh_id()
        return Pipeline(mesh_device, self._stages[my_mesh_id])


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
