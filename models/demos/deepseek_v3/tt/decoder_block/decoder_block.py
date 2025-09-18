# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_base import DecoderBlockBase
from models.demos.deepseek_v3.tt.mlp.non_expert import NonExpert
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class DecoderBlock(DecoderBlockBase):
    @classmethod
    def convert_mlp_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        return NonExpert.convert_weights(hf_config, state_dicts, output_path, mesh_device)

    @classmethod
    def prefill_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        is_padding_layer: tuple[bool, ...] | None = None,
    ) -> ModelPrefillConfig:
        return NonExpert.prefill_model_config(hf_config, mesh_device)

    @classmethod
    def decode_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        is_padding_layer: tuple[bool, ...] | None = None,
    ) -> ModelDecodeConfig:
        return NonExpert.decode_model_config(hf_config, mesh_device)

    @classmethod
    def create_mlp_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL1D,
        is_padding_layer: tuple[bool, ...] | None = None,
    ) -> ModelState:
        return NonExpert.create_state(hf_config, mesh_device, ccl)

    @classmethod
    def create_mlp_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        is_padding_layer: tuple[bool, ...] | None = None,
    ) -> ModelState:
        return {}

    @classmethod
    def forward_mlp_prefill(cls, x: ttnn.Tensor, row_idx: int, cfg: RunPrefillConfig) -> ttnn.Tensor:
        return NonExpert.forward_prefill(x, cfg)

    @classmethod
    def forward_mlp_decode(cls, x: ttnn.Tensor, row_idx: int, cfg: RunDecodeConfig) -> ttnn.Tensor:
        return NonExpert.forward_decode(x, cfg)
