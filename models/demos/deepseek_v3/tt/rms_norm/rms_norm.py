# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.rms_norm.rms_norm_base import RMSNormBase
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, MeshDeviceStub, RMSNormConfig
from models.demos.deepseek_v3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_LOFI, get_state_dicts, save_and_get_path
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
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert mesh_device.shape[0] > 0, "RMSNorm does not support 0D devices"

        torch_metaweight = get_state_dicts(state_dicts, "weight", dtype=torch.bfloat16)
        num_shards = torch_metaweight.shape[0]
        assert num_shards == mesh_device.shape[0], "Number of state dicts does not match the number of rows."

        tt_weight = ttnn.as_tensor(
            torch_metaweight.reshape(
                (num_shards, 1, -1, ttnn.TILE_SIZE)
            ),  # Reshape to tile width sticks for optimal performance
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, None)),
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
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        return RMSNormConfig(
            epsilon=hf_config.rms_norm_eps,
            weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
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
        pc = cls._get_pc(x.memory_config())
        return ttnn.rms_norm(x, program_config=pc, **cfg)
