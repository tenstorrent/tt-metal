# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dense decoder block implementations.

- ``DecoderBlock2D``: TT-style classmethod orchestration (DeepSeek parity).
- ``Mistral4DenseDecoderBlock2D``: eager HF-compatible decoder block used by parity tests.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4MLP

import ttnn
from models.demos.mistral_small_4_119B.tt.decoder_block.decoder_block_2d_base import (
    DecoderBlock2DBase,
    Mistral4DecoderBlock2DBase,
)
from models.demos.mistral_small_4_119B.tt.mlp.non_expert import NonExpert
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class DecoderBlock2D(DecoderBlock2DBase):
    """TT dense decoder block (`NonExpert` MLP), mirroring DeepSeek `DecoderBlock2D`."""

    @classmethod
    def convert_mlp_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        return NonExpert.convert_weights(hf_config, (state_dict,) * mesh_device.shape[0], output_path, mesh_device)

    @classmethod
    def prefill_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        fabric_config: ttnn.FabricConfig,
    ) -> ModelPrefillConfig:
        return NonExpert.prefill_model_config(hf_config, mesh_device, fabric_config)

    @classmethod
    def decode_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        fabric_config: ttnn.FabricConfig,
        batch_size_per_row: int,
    ) -> ModelDecodeConfig:
        return NonExpert.decode_model_config(
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
        return NonExpert.create_state(hf_config, mesh_device, ccl)

    @classmethod
    def create_mlp_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelState:
        del hf_config, mesh_device
        return {}

    @classmethod
    def forward_mlp_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        if x.shape[1] == 1:
            return NonExpert.forward_prefill(x, cfg)

        batch_size = x.shape[1]
        seq_len = x.shape[2]
        x = ttnn.reshape(x, (x.shape[0], 1, batch_size * seq_len, x.shape[3]))
        output = NonExpert.forward_prefill(x, cfg)
        return ttnn.reshape(output, (output.shape[0], batch_size, seq_len, output.shape[3]))

    @classmethod
    def forward_mlp_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        return NonExpert.forward_decode(x, cfg)


class Mistral4DenseDecoderBlock2D(Mistral4DecoderBlock2DBase):
    """Eager decoder layer whose FFN is a standard ``Mistral4MLP`` (no routed MoE)."""

    def __init__(self, config: Mistral4Config, layer_idx: int) -> None:
        super().__init__(config, layer_idx, mlp=Mistral4MLP(config))


__all__ = ["DecoderBlock2D", "Mistral4DenseDecoderBlock2D"]
