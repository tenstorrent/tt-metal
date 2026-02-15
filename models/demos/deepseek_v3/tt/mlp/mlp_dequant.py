# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path
from typing import final

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.mlp.mlp import MLP
from models.demos.deepseek_v3.utils.config_helpers import dequantize, get_state_dicts
from models.demos.deepseek_v3.utils.run_config import WeightConfig
from models.demos.deepseek_v3.utils.weight_spec import ModuleWeightSpec, WeightSpec, WeightSpecContext


class MLPDequant(MLP):
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
                "input_tensor_b": cls.convert_quantized_metaweight(
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
            }
            for hf_name, models_name, is_w2 in [
                ("gate_proj", "w1", False),
                ("down_proj", "w2", True),
                ("up_proj", "w3", False),
            ]
            for in_features, out_features in [cls.get_weight_shape(hf_config, is_w2)]
        }

    @classmethod
    def create_weight_spec(
        cls,
        hf_config: PretrainedConfig,
        mesh_device_or_shape: ttnn.MeshDevice | tuple[int, int],
        context: WeightSpecContext,
    ) -> ModuleWeightSpec:
        base_spec = super().create_weight_spec(hf_config, mesh_device_or_shape, context)
        block_size = hf_config.quantization_config["weight_block_size"]

        def wrap_preprocessor(weight_name: str, weight_spec: WeightSpec) -> WeightSpec:
            scale_inv = context.get_reference_tensor(f"{weight_name}_scale_inv")

            def preprocessor(t: torch.Tensor) -> torch.Tensor:
                dequantized = dequantize(t, scale_inv, block_size)
                return weight_spec.preprocessor(dequantized)

            return replace(weight_spec, preprocessor=preprocessor)

        return {
            "gate_proj.weight": wrap_preprocessor("gate_proj.weight", base_spec["gate_proj.weight"]),
            "down_proj.weight": wrap_preprocessor("down_proj.weight", base_spec["down_proj.weight"]),
            "up_proj.weight": wrap_preprocessor("up_proj.weight", base_spec["up_proj.weight"]),
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
    ) -> WeightSpec:
        """
        Convert the quantized weight tensor to a format suitable for TTNN.

        Args:
            quantized_weight_tensor: The quantized weight tensor.
            scale_inv_tensor: The scale inverse tensor.
            mesh_device: The mesh device to use for the conversion.
            is_w2: Whether this is the w2 (down_proj) weight, which has different sharding.
            metaweight_block_size: Block size for dequantization.

        Returns:
            WeightSpec describing how to convert the weight.
        """
        # Dequantize the weight tensor
        dequantized_tensor = dequantize(quantized_weight_tensor, scale_inv_tensor, metaweight_block_size)

        # Use the parent class's convert_metaweight method
        return cls.convert_metaweight(
            torch_metaweight_tensor=dequantized_tensor,
            mesh_device=mesh_device,
            is_w2=is_w2,
        )
