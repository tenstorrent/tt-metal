# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GraniteTTMModelConfig:
    context_length: int
    forecast_length: int
    patch_length: int
    d_model: int
    num_layers: int
    num_channels: int
    expansion_factor: int = 2
    gated_attn: bool = True
    adaptive_patching_levels: int = 3
    decoder_d_model: int = 128
    decoder_num_layers: int = 2
    norm_eps: float = 1e-5

    @classmethod
    def from_hf_config(cls, hf_config: Any, num_channels: int = 1) -> "GraniteTTMModelConfig":
        return cls(
            context_length=getattr(hf_config, "context_length", 512),
            forecast_length=getattr(hf_config, "prediction_length", 96),
            patch_length=getattr(hf_config, "patch_length", 64),
            d_model=getattr(hf_config, "d_model", 192),
            num_layers=getattr(hf_config, "num_layers", 2),
            num_channels=num_channels,
            expansion_factor=getattr(hf_config, "expansion_factor", 2),
            gated_attn=getattr(hf_config, "gated_attn", True),
            adaptive_patching_levels=getattr(hf_config, "adaptive_patching_levels", 3),
            decoder_d_model=getattr(hf_config, "decoder_d_model", 128),
            decoder_num_layers=getattr(hf_config, "decoder_num_layers", 2),
            norm_eps=getattr(hf_config, "norm_eps", 1e-5),
        )

    @property
    def num_patches(self) -> int:
        return self.context_length // self.patch_length
