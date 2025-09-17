# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import final

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, LinearConfig, MeshDeviceStub, MulConfig
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_LOFI,
    dequantize,
    even_int_div,
    save_and_get_path,
)
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class Experts(AbstractModule):
    """Experts layer for Mixture-of-Experts (MoE) module."""

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

            # Process each weight type
            for weight_key in weight_key_to_expert.keys():
                # Collect tensors for this weight type and device
                weight_tensors = []
                for expert_idx in expert_indices:
                    weight_tensor = state_dict[f"experts.{expert_idx}.{weight_key}.weight"]
                    inv_scale_key = f"experts.{expert_idx}.{weight_key}.weight_scale_inv"
                    if inv_scale_key in state_dict:
                        # Dequantize the tensor before using it
                        inv_scale_tensor = state_dict[inv_scale_key]
                        weight_tensor = dequantize(
                            weight_tensor, inv_scale_tensor, hf_config.quantization_config["weight_block_size"]
                        )

                    weight_tensors.append(weight_tensor)
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
            [ttnn.from_torch(e.unsqueeze(0), layout=ttnn.ROW_MAJOR_LAYOUT) for e in wx_per_device_state_dict_group],
            mesh_device.shape,
        )
        # This is a solution to fasten the conversion to tile layout and bfloat4_b with multi-threading
        multi_dev_host_weights = ttnn.to_dtype(multi_dev_host_weights, ttnn.bfloat4_b)

        multi_dev_weights = ttnn.to_device(
            multi_dev_host_weights,
            mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return multi_dev_weights

    @classmethod
    def is_device_supported(cls, mesh_device: ttnn.Device) -> bool:
        """
        As we only support 1D tensor parallelism, we only support 1D mesh devices.

        Args:
            mesh_device: The mesh device to check.

        Returns:
            True if the device is supported, False otherwise.
        """
        return tuple(mesh_device.shape) == (1, 8) or tuple(mesh_device.shape) == (4, 8)

    @classmethod
    def _create_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, mode: str
    ) -> ModelPrefillConfig | ModelDecodeConfig:
        num_experts_per_device = cls._get_num_experts_per_device(hf_config, mesh_device)

        # Calculate input and output memory configurations
        if mode == "decode":
            input_memory_config = ttnn.L1_MEMORY_CONFIG
            output_memory_config = ttnn.L1_MEMORY_CONFIG
        else:
            input_memory_config = ttnn.DRAM_MEMORY_CONFIG
            output_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Construct the config
        return {
            "w1_experts": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "w2_experts": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "w3_experts": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "mul_experts": MulConfig(
                memory_config=output_memory_config,
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
        return cls._create_model_config(hf_config, mesh_device, "decode")

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        """Generate prefill configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        return cls._create_model_config(hf_config, mesh_device, "prefill")

    @classmethod
    def _forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"

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
        return cls._forward(x, cfg)

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        return cls._forward(x, cfg)
