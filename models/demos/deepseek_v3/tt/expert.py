# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import final

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.mlp_1d import MLP1D
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, LinearConfig, MeshDeviceStub, MulConfig
from models.demos.deepseek_v3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_LOFI, even_int_div, save_and_get_path
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class Expert(MLP1D):  # The only difference with the regular Dequantized MLP is the intermediate layer size
    """Expert layer for Mixture-of-Experts (MoE) models."""

    @classmethod
    def _get_model_dims_from_cfg(cls, hf_config: PretrainedConfig) -> tuple[int, int]:
        dim = hf_config.hidden_size
        hidden_dim = hf_config.moe_intermediate_size
        return dim, hidden_dim

    @classmethod
    def _get_num_experts_per_device(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> int:
        """Calculate the number of experts per device based on the total number of experts and the device shape."""
        return even_int_div(hf_config.n_routed_experts, mesh_device.get_num_devices())

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        num_experts_per_device = cls._get_num_experts_per_device(hf_config, mesh_device)

        # Mapping from weight keys to expert names
        weight_key_to_expert = {"gate_proj": "w1_experts", "down_proj": "w2_experts", "up_proj": "w3_experts"}

        # Initialize per-weight-type dictionary of device groups
        weight_groups = {key: [] for key in weight_key_to_expert.keys()}
        num_devices = mesh_device.get_num_devices()

        # Group experts by device
        for device_idx in range(num_devices):
            # Calculate expert range for this device
            start_expert = device_idx * num_experts_per_device
            end_expert = start_expert + num_experts_per_device
            expert_indices = range(start_expert, end_expert)

            # TODO: check for quantized weights
            # Process each weight type
            for weight_key in weight_key_to_expert.keys():
                # Collect tensors for this weight type and device
                weight_tensors = [
                    state_dict[f"experts.{expert_idx}.{weight_key}.weight"] for expert_idx in expert_indices
                ]
                # Stack and permute, then add to the corresponding weight group
                weight_groups[weight_key].append(torch.stack(weight_tensors, dim=0).permute(0, 2, 1))

        # Convert weights for each expert group using compact loop
        return {
            experts_group: {
                "input_tensor_b": save_and_get_path(
                    output_path / f"{experts_group}.input_tensor_b",
                    cls._convert_weight(
                        weight_groups[weight_key],
                        mesh_device,
                    ),
                )
            }
            for weight_key, experts_group in weight_key_to_expert.items()
        }

    @final
    @classmethod
    def _convert_weight(
        cls,
        wx_per_device_state_dict_group: list[torch.Tensor],
        mesh_device: ttnn.Device,
    ) -> ttnn.Tensor:
        """
        Convert a normal (non-quantized) weight tensor to a format suitable for TTNN.

        Args:
            weight_tensor: The weight tensor.
            mesh_device: The mesh device to use for the conversion.

        Returns:
            The converted TTNN tensor.
        """
        multi_dev_host_weights = ttnn.from_host_shards(
            [
                ttnn.from_torch(e.unsqueeze(0), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
                for e in wx_per_device_state_dict_group
            ],
            mesh_device.shape,
        )

        multi_dev_weights = ttnn.to_device(
            multi_dev_host_weights,
            mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return multi_dev_weights

    @classmethod
    def _create_model_config(
        cls, mesh_device: ttnn.Device, mem_config: ttnn.MemoryConfig, num_experts_per_device: int
    ) -> ModelPrefillConfig:
        # Calculate input and output memory configurations
        input_memory_config = mem_config
        output_memory_config = mem_config

        # Construct the config
        return {
            "w1_experts": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=mem_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "w2_experts": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=mem_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "w3_experts": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=mem_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "mul_experts": MulConfig(
                memory_config=mem_config,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            ),
            "input_memory_config": input_memory_config,  # For asserting the input to the MLP
            "output_memory_config": output_memory_config,  # For asserting the output of the MLP
            "num_experts_per_device": num_experts_per_device,
        }

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        num_experts_per_device = cls._get_num_experts_per_device(hf_config, mesh_device)
        return cls._create_model_config(mesh_device, ttnn.L1_MEMORY_CONFIG, num_experts_per_device)

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        """Generate prefill configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        num_experts_per_device = even_int_div(hf_config.n_routed_experts, mesh_device.get_num_devices())
        return cls._create_model_config(mesh_device, ttnn.DRAM_MEMORY_CONFIG, num_experts_per_device)

    @classmethod
    def _forward_util(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"]

        # Gate and up projections
        w1_out = ttnn.linear(x, **cfg["w1_experts"])
        w3_out = ttnn.linear(x, **cfg["w3_experts"])

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul_experts"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Down projection
        output = ttnn.linear(activated, **cfg["w2_experts"])
        ttnn.deallocate(activated)

        assert output.memory_config() == cfg["output_memory_config"]
        return output

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        return cls._forward_util(x, cfg)

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        return cls._forward_util(x, cfg)
