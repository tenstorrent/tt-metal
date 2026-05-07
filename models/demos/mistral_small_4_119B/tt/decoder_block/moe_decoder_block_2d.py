# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoE decoder block implementations.

- ``MoEDecoderBlock2D``: TT-style classmethod orchestration (DeepSeek parity).
- ``Mistral4MoEDecoderBlock2D``: eager HF-compatible decoder block used by parity tests.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

import ttnn
from models.demos.mistral_small_4_119B.tt.decoder_block.decoder_block_2d_base import (
    DecoderBlock2DBase,
    Mistral4DecoderBlock2DBase,
)
from models.demos.mistral_small_4_119B.tt.moe.moe import TtMistral4MoE
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class MoEDecoderBlock2D(DecoderBlock2DBase):
    """TT MoE decoder block mirroring DeepSeek `MoEDecoderBlock2D` hooks."""

    @classmethod
    def convert_mlp_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        return TtMistral4MoE.convert_weights(hf_config, (state_dict,), output_path, mesh_device)

    @classmethod
    def prefill_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        fabric_config: ttnn.FabricConfig,
    ) -> ModelPrefillConfig:
        return TtMistral4MoE.prefill_model_config(hf_config, mesh_device, fabric_config)

    @classmethod
    def decode_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        fabric_config: ttnn.FabricConfig,
        batch_size_per_row: int,
    ) -> ModelDecodeConfig:
        return TtMistral4MoE.decode_model_config(
            hf_config,
            mesh_device,
            fabric_config,
            batch_size_per_row=batch_size_per_row,
        )

    @classmethod
    def create_mlp_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
    ) -> ModelState:
        return TtMistral4MoE.create_state(hf_config, mesh_device, ccl)

    @classmethod
    def create_mlp_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelState:
        return TtMistral4MoE.create_shared_state(hf_config, mesh_device)

    @classmethod
    def forward_mlp_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        return TtMistral4MoE.forward_prefill(x, cfg)

    @classmethod
    def forward_mlp_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        return TtMistral4MoE.forward_decode(x, cfg)


class Mistral4MoEDecoderBlock2D(Mistral4DecoderBlock2DBase):
    """Decoder layer whose FFN is HF ``Mistral4MoE`` (routed experts + shared expert MLP)."""

    def __init__(self, config: Mistral4Config, layer_idx: int) -> None:
        super().__init__(config, layer_idx, mlp=Mistral4MoE(config))


__all__ = ["MoEDecoderBlock2D", "Mistral4MoEDecoderBlock2D"]
