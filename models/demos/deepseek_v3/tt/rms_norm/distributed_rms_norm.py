# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.rms_norm.rms_norm_base import RMSNormBase
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherConfig,
    FromWeightConfig,
    MeshDeviceStub,
    OpConfigBase,
    RMSNormPostAllGatherConfig,
    RMSNormPreAllGatherConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import save_and_get_path
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class DistributedRMSNorm(RMSNormBase):
    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert cls.is_device_supported(mesh_device)

        torch_weight = state_dict["weight"]
        tt_weight = ttnn.as_tensor(
            torch_weight.reshape((1, 1, -1, ttnn.TILE_SIZE)),  # Reshape to tile width sticks for optimal performance
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, -2), mesh_shape=tuple(mesh_device.shape)),
        )

        # Save to disk with standard naming - "rmsnorm" must match the op name used in the model config
        # so that RunConfig can populate it with the actual weight tensors at runtime
        return {
            "rms_norm_post_all_gather": {
                "weight": save_and_get_path(output_path / "rmsnorm.weight", tt_weight),
            }
        }

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        """Generate prefill configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        return cls._model_config(
            hf_config=hf_config, mesh_device=mesh_device, rms_norm_stats_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # type: ignore

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelDecodeConfig containing operator configurations for decode mode
        """
        return cls._model_config(
            hf_config=hf_config,
            mesh_device=mesh_device,
            rms_norm_stats_memory_config=ttnn.create_sharded_memory_config(
                shape=[1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE * tuple(mesh_device.shape)[1]],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
        )  # type: ignore

    @classmethod
    def _model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, rms_norm_stats_memory_config: ttnn.MemoryConfig
    ) -> dict[str, OpConfigBase]:
        """Generate model configuration for RMSNorm."""
        return {
            "rms_norm_pre_all_gather": RMSNormPreAllGatherConfig(
                dtype=ttnn.bfloat16,
            ),
            "all_gather": AllGatherConfig(
                dim=3,
                cluster_axis=1,
                mesh_device=mesh_device,
                memory_config=rms_norm_stats_memory_config,
                topology=ttnn.Topology.Linear,
            ),
            "rms_norm_post_all_gather": RMSNormPostAllGatherConfig(
                epsilon=hf_config.rms_norm_eps,
                weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                dtype=ttnn.bfloat16,
            ),
        }

    @classmethod
    def _rmsnorm_forward(cls, x: ttnn.Tensor, cfg: RunPrefillConfig | RunDecodeConfig) -> ttnn.Tensor:
        """Forward pass of the embedding.

        Args:
            x: Input tensor (token indices)
            cfg: RunConfig containing weights and op configurations

        Returns:
            Output tensor after embedding lookup
        """

        program_config = cls._get_pc(x.memory_config())
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(x, program_config=program_config, **cfg["rms_norm_pre_all_gather"])

        # AllGather stats
        tt_gathered_stats = ttnn.all_gather(tt_stats, **cfg["all_gather"])
        ttnn.deallocate(tt_stats)

        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            x,
            tt_gathered_stats,
            program_config=program_config,
            **cfg["rms_norm_post_all_gather"],
        )
        ttnn.deallocate(tt_stats)

        return tt_out
