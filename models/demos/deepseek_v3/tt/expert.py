# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import final

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.mlp_1d import MLP1D
from models.demos.deepseek_v3.utils.config_helpers import dram_sharded_weight_config, save_and_get_path
from models.demos.deepseek_v3.utils.run_config import WeightConfig


class Expert(MLP1D):  # The only difference with the regular Dequantized MLP is the intermediate layer size
    """Expert layer for Mixture-of-Experts (MoE) models."""

    @classmethod
    def _get_model_dims_from_cfg(cls, hf_config: PretrainedConfig) -> tuple[int, int]:
        dim = hf_config.hidden_size
        hidden_dim = hf_config.moe_intermediate_size
        return dim, hidden_dim

    @classmethod
    def convert_weights_moe(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        # assert cls.is_device_supported(mesh_device)

        # breakpoint()  # Debugging point to inspect the state_dict structure
        num_experts_per_device = hf_config.n_routed_experts // mesh_device.get_num_devices()
        num_groups = hf_config.n_routed_experts // num_experts_per_device

        w1_per_device_state_dict_group = []
        w2_per_device_state_dict_group = []
        w3_per_device_state_dict_group = []

        # Group experts by device
        for device_idx in range(mesh_device.get_num_devices()):
            w1_sub_group = []
            w2_sub_group = []
            w3_sub_group = []

            # Add experts for this device
            start_expert = device_idx * num_experts_per_device
            end_expert = start_expert + num_experts_per_device

            for expert_idx in range(start_expert, end_expert):
                w1_sub_group.append(state_dict[f"experts.{expert_idx}.gate_proj.weight"])
                w2_sub_group.append(state_dict[f"experts.{expert_idx}.down_proj.weight"])
                w3_sub_group.append(state_dict[f"experts.{expert_idx}.up_proj.weight"])

            w1_per_device_state_dict_group.append(torch.stack(w1_sub_group, dim=0).permute(0, 2, 1))
            w2_per_device_state_dict_group.append(torch.stack(w2_sub_group, dim=0).permute(0, 2, 1))
            w3_per_device_state_dict_group.append(torch.stack(w3_sub_group, dim=0).permute(0, 2, 1))

        # Convert weights for each expert group
        return {
            "w1_experts": {
                "input_tensor_b": save_and_get_path(
                    output_path / f"w1_experts.input_tensor_b",
                    cls._convert_weight_moe(hf_config, w1_per_device_state_dict_group, mesh_device, is_w2=False),
                )
            },
            "w2_experts": {
                "input_tensor_b": save_and_get_path(
                    output_path / f"w2_experts.input_tensor_b",
                    cls._convert_weight_moe(hf_config, w2_per_device_state_dict_group, mesh_device, is_w2=True),
                )
            },
            "w3_experts": {
                "input_tensor_b": save_and_get_path(
                    output_path / f"w3_experts.input_tensor_b",
                    cls._convert_weight_moe(hf_config, w3_per_device_state_dict_group, mesh_device, is_w2=False),
                )
            },
        }

    @final
    @classmethod
    def _convert_weight_moe(
        cls,
        hf_config: PretrainedConfig,
        wx_per_device_state_dict_group: list[torch.Tensor],
        mesh_device: ttnn.Device,
        is_w2: bool,
    ) -> ttnn.Tensor:
        """
        Convert a normal (non-quantized) weight tensor to a format suitable for TTNN.

        Args:
            weight_tensor: The weight tensor.
            mesh_device: The mesh device to use for the conversion.

        Returns:
            The converted TTNN tensor.
        """
        dim, hidden_dim = cls._get_model_dims_from_cfg(hf_config)

        per_device_in_features, per_device_out_features = dim, hidden_dim
        if is_w2:
            per_device_in_features, per_device_out_features = hidden_dim, dim

        multi_dev_host_weights = ttnn.from_host_shards(
            [ttnn.from_torch(e, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT) for e in wx_per_device_state_dict_group],
            mesh_device.shape,
        )

        multi_dev_weights = ttnn.to_device(
            multi_dev_host_weights,
            mesh_device,
            memory_config=dram_sharded_weight_config(
                per_device_in_features * 8, per_device_out_features, mesh_device.dram_grid_size()
            ),
        )

        return multi_dev_weights
