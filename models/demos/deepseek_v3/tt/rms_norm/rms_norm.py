# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.rms_norm.rms_norm_base import RMSNormBase
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, MeshDeviceStub, RMSNormConfig
from models.demos.deepseek_v3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_HIFI2, save_and_get_path
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class RMSNorm(RMSNormBase):
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
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Save to disk with standard naming - "rmsnorm" must match the op name used in the model config
        # so that RunConfig can populate it with the actual weight tensors at runtime
        return {
            "weight": save_and_get_path(output_path / "rmsnorm.weight", tt_weight),
        }

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        return RMSNormConfig(
            epsilon=hf_config.rms_norm_eps,
            weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        return RMSNormConfig(
            epsilon=hf_config.rms_norm_eps,
            weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
        )

    @classmethod
    def _rmsnorm_forward(cls, x: ttnn.Tensor, cfg: RunPrefillConfig | RunDecodeConfig) -> ttnn.Tensor:
        """Forward pass of the embedding.

        Args:
            x: Input tensor (token indices)
            cfg: RunConfig containing weights and op configurations

        Returns:
            Output tensor after embedding lookup
        """
        return ttnn.rms_norm(x, program_config=cls._get_pc(x.memory_config()), **cfg)
