# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.mlp.mlp import MLP
from models.demos.deepseek_v3.utils.config_helpers import get_state_dicts
from models.demos.deepseek_v3.utils.run_config import WeightConfig


class MLPDequant(MLP):
    """
    Base class for MLP modules in DeepSeek V3.
    Loads already-dequantized weights (bfloat16) directly from state dict.
    """

    WEIGHT_TORCH_DTYPE = torch.bfloat16

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        return {
            models_name: {
                "input_tensor_b": cls.convert_metaweight(
                    output_path / f"{models_name}.input_tensor_b",
                    get_state_dicts(
                        state_dict,
                        f"{hf_name}.weight",
                        shape=(out_features, in_features),
                        dtype=cls.WEIGHT_TORCH_DTYPE,
                    ),
                    mesh_device,
                    is_w2,
                ),
            }
            for hf_name, models_name, is_w2 in [
                ("gate_proj", "w1", False),
                ("down_proj", "w2", True),
                ("up_proj", "w3", False),
            ]
            for in_features, out_features in [cls.get_weight_shape(hf_config, is_w2)]
        }
