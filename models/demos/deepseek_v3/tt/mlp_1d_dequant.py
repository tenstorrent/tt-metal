# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, final

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.mlp_1d import MLP1D
from models.demos.deepseek_v3.utils.config_helpers import dequantize, save_and_get_path
from models.demos.deepseek_v3.utils.run_config import WeightConfig


class MLP1DDequant(MLP1D):
    """
    Base class for MLP modules in DeepSeek V3.
    """

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert cls.is_device_supported(mesh_device)
        return {
            models_name: {
                "input_tensor_b": save_and_get_path(
                    output_path / f"{models_name}.input_tensor_b",
                    cls.convert_quantized_weight(
                        hf_config,
                        state_dict[f"{hf_name}.weight"],
                        state_dict[f"{hf_name}.weight_scale_inv"],
                        mesh_device,
                        is_w2=(models_name == "w2"),
                    ),
                )
            }
            for hf_name, models_name in [
                ("gate_proj", "w1"),
                ("down_proj", "w2"),
                ("up_proj", "w3"),
            ]
        }

    @final
    @classmethod
    def convert_quantized_weight(
        cls,
        hf_config: Any,
        quantized_weight_tensor: torch.Tensor,
        scale_inv_tensor: torch.Tensor,
        mesh_device: ttnn.Device,
        is_w2: bool,
    ) -> ttnn.Tensor:
        """
        Convert the quantized weight tensor to a format suitable for TTNN.

        Args:
            hf_config: The Hugging Face configuration object.
            quantized_weight_tensor: The quantized weight tensor.
            scale_inv_tensor: The scale inverse tensor.
            mesh_device: The mesh device to use for the conversion.

        Returns:
            The converted TTNN tensor.
        """
        return cls.convert_weight(
            hf_config=hf_config,
            weight_tensor=dequantize(
                quantized_weight_tensor, scale_inv_tensor, hf_config.quantization_config.weight_block_size
            ),
            mesh_device=mesh_device,
            is_w2=is_w2,
        )
