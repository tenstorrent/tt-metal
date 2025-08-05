# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_base import DecoderBlockBase
from models.demos.deepseek_v3.tt.mlp.shared_expert import SharedExpert
from models.demos.deepseek_v3.tt.moe import MoE
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class MoEDecoderBlock(DecoderBlockBase):
    @classmethod
    @abstractmethod
    def convert_mlp_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        assert mesh_device.shape[0] == len(
            state_dicts
        ), "Number of state dicts must match the number of mesh device rows"
        return {
            "shared_expert": SharedExpert.convert_weights(
                hf_config, state_dicts, output_path / "shared_expert", mesh_device
            ),
            "moe": [
                (
                    MoE.convert_weights(hf_config, state_dict, output_path / "moe", mesh_device)
                    if state_dict is not None
                    else None
                )
                for state_dict in state_dicts
            ],
        }

    @classmethod
    @abstractmethod
    def prefill_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelPrefillConfig:
        return MoE.prefill_model_config(hf_config, mesh_device)

    @classmethod
    @abstractmethod
    def decode_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelDecodeConfig:
        return MoE.decode_model_config(hf_config, mesh_device)

    @classmethod
    @abstractmethod
    def create_mlp_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL1D,
    ) -> ModelState:
        return MoE.create_state(hf_config, mesh_device, ccl)

    @classmethod
    @abstractmethod
    def forward_mlp_prefill(cls, x: ttnn.Tensor, row_idx: int, cfg: RunPrefillConfig) -> ttnn.Tensor:
        mlp_out = MoE.forward_prefill(x, cfg["moe"][row_idx])
        mlp_out += SharedExpert.forward_prefill(x, cfg["shared_expert"])
        return mlp_out

    @classmethod
    @abstractmethod
    def forward_mlp_decode(cls, x: ttnn.Tensor, row_idx: int, cfg: RunDecodeConfig) -> ttnn.Tensor:
        mlp_out = MoE.forward_decode(x, cfg["moe"][row_idx])
        mlp_out += SharedExpert.forward_decode(x, cfg["shared_expert"])
        return mlp_out
