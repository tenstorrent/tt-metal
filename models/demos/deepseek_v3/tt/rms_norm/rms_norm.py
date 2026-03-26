# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.rms_norm.rms_norm_base import RMSNormBase
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, MeshDeviceStub, RMSNormConfig
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI4_NOFP32_ACC,
    get_state_dicts,
    shard_and_save,
)
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


def _has_distinct_buffer(a: ttnn.Tensor, b: ttnn.Tensor) -> bool:
    try:
        return a.buffer_address() != b.buffer_address()
    except Exception:
        return a is not b


class RMSNorm(RMSNormBase):
    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        torch_metaweight = get_state_dicts(state_dicts, "weight", dtype=torch.bfloat16)
        num_shards = torch_metaweight.shape[0]
        assert num_shards == mesh_device.shape[0], "Number of state dicts does not match the number of rows."

        # Save to disk with standard naming - "rmsnorm" must match the op name used in the model config
        # so that RunConfig can populate it with the actual weight tensors at runtime
        return {
            "weight": shard_and_save(
                output_path / "rmsnorm.weight",
                torch_metaweight.reshape(
                    (num_shards, 1, -1, ttnn.TILE_SIZE)
                ),  # Reshape to tile width sticks for optimal performance
                shard_dims=(0, None),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        }

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        return RMSNormConfig(
            epsilon=hf_config.rms_norm_eps,
            weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI4_NOFP32_ACC,
        )

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        return RMSNormConfig(
            epsilon=hf_config.rms_norm_eps,
            weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI4_NOFP32_ACC,
        )

    @classmethod
    def _rmsnorm_forward_decode(
        cls,
        x: ttnn.Tensor,
        cfg: RunDecodeConfig,
        memory_config: ttnn.MemoryConfig,
        output_memory_config: ttnn.MemoryConfig,
        residual: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Forward pass of the RMSNorm for decode mode.

        Args:
            x: Input tensor
            cfg: RunDecodeConfig containing weights and op configurations
            memory_config: Memory configuration for the input tensor
            output_memory_config: Memory configuration for the output tensor
            residual: Optional residual tensor to add (not used in this implementation)
        Returns:
            Output tensor after RMSNorm computation
        """
        tensor_in = ttnn.to_memory_config(x, memory_config)
        tt_out = ttnn.rms_norm(tensor_in, program_config=cls._get_pc(tensor_in.memory_config()), **cfg)
        tt_out = ttnn.to_memory_config(tt_out, output_memory_config)
        if _has_distinct_buffer(x, tensor_in):
            ttnn.deallocate(tensor_in)
        return tt_out, residual

    @classmethod
    def _rmsnorm_forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        """Forward pass of the RMSNorm for prefill mode.

        Args:
            x: Input tensor
            cfg: RunPrefillConfig containing weights and op configurations

        Returns:
            Output tensor after RMSNorm computation
        """
        return ttnn.rms_norm(x, program_config=cls._get_pc(x.memory_config()), **cfg)
