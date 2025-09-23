# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import final

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.mlp.mlp_1d import MLP1D
from models.demos.deepseek_v3.utils.config_helpers import dequantize, get_state_dicts, save_and_get_path
from models.demos.deepseek_v3.utils.run_config import WeightConfig


class MLP1DDequant(MLP1D):
    """
    Base class for MLP modules in DeepSeek V3.
    """

    WEIGHT_TORCH_DTYPE = torch.float8_e4m3fn
    WEIGHT_SCALE_INV_TORCH_DTYPE = torch.float32

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        weight_block_height, weight_block_width = hf_config.quantization_config["weight_block_size"]

        return {
            models_name: {
                "input_tensor_b": save_and_get_path(
                    output_path / f"{models_name}.input_tensor_b",
                    cls.convert_quantized_metaweight(
                        get_state_dicts(
                            state_dict,
                            f"{hf_name}.weight",
                            shape=(out_features, in_features),
                            dtype=cls.WEIGHT_TORCH_DTYPE,
                        ),
                        get_state_dicts(
                            state_dict,
                            f"{hf_name}.weight_scale_inv",
                            shape=(
                                ttnn.core.divup(out_features, weight_block_height),
                                ttnn.core.divup(in_features, weight_block_width),
                            ),
                            dtype=cls.WEIGHT_SCALE_INV_TORCH_DTYPE,
                        ),
                        mesh_device,
                        is_w2=is_w2,
                        metaweight_block_size=(1, weight_block_height, weight_block_width),
                    ),
                )
            }
            for hf_name, models_name, is_w2 in [
                ("gate_proj", "w1", False),
                ("down_proj", "w2", True),
                ("up_proj", "w3", False),
            ]
            for in_features, out_features in [cls.get_weight_shape(hf_config, is_w2)]
        }

    @final
    @classmethod
    def convert_quantized_metaweight(
        cls,
        quantized_weight_tensor: torch.Tensor,
        scale_inv_tensor: torch.Tensor,
        mesh_device: ttnn.Device,
        is_w2: bool,
        metaweight_block_size: tuple[int, ...],
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
        return cls.convert_metaweight(
            torch_metaweight_tensor=dequantize(quantized_weight_tensor, scale_inv_tensor, metaweight_block_size),
            mesh_device=mesh_device,
            is_w2=is_w2,
        )
