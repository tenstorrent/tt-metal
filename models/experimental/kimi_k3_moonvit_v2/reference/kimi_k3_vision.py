# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The vision path of upstream `KimiK3ForConditionalGeneration`, without the language model.

`__init__` is upstream's minus `language_model`, `_extract_image_features` is upstream's, and
`forward` is the image branch of upstream `forward`: tower (patch embed, encoder, patch merger),
then `mm_projector`. Child names are upstream's, so the checkpoint's `vision_tower.*` and
`mm_projector.*` keys load unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from .configuration_kimi_k3 import KimiK3VisionConfig
from .modeling_kimi_k3 import (
    MLP,
    IdentityMap,
    MoonViT3dPretrainedModel,
    PatchMergerMLP,
    PatchMergerMLPV2,
    ProjectorConfig,
    VisionTowerConfig,
)

CONFIG_JSON = Path(__file__).parent / "config.json"


def load_vision_config(source: str | Path | None = None, *, num_hidden_layers: int | None = None) -> KimiK3VisionConfig:
    """`KimiK3VisionConfig` from a checkpoint folder or `config.json` (default: the copy in this folder).

    Attention is upstream's `eager` path; the `flash_attention_2` path needs flash-attn and a GPU.
    """
    path = Path(source) if source is not None else CONFIG_JSON
    if path.is_dir():
        path = path / "config.json"
    vision_config = dict(json.loads(path.read_text())["vision_config"])
    if num_hidden_layers is not None:
        vision_config["vt_num_hidden_layers"] = num_hidden_layers
    vision_config["_attn_implementation"] = "eager"
    return KimiK3VisionConfig(**vision_config)


class KimiK3VisionModel(nn.Module):
    def __init__(self, config: KimiK3VisionConfig) -> None:
        super().__init__()

        vt_config = VisionTowerConfig(config)
        self.vision_tower = MoonViT3dPretrainedModel(vt_config)

        proj_config = ProjectorConfig(config)
        if proj_config.mm_projector_type == "identity":
            self.mm_projector = IdentityMap()
        elif proj_config.mm_projector_type == "mlp":
            self.mm_projector = MLP(proj_config)
        elif proj_config.mm_projector_type == "patchmerger":
            self.mm_projector = PatchMergerMLP(proj_config)
        elif proj_config.mm_projector_type == "patchmergerv2":
            self.mm_projector = PatchMergerMLPV2(proj_config)
        else:
            raise ValueError(f"Unsupported mm_projector_type: {proj_config.mm_projector_type}")

    def _extract_image_features(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> list[torch.Tensor]:
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(target_dtype)

        image_features = self.vision_tower(pixel_values, grid_thws)
        return image_features

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """(L, 3, 14, 14) patches, (N, 3) grids -> per image (h*w/4, text_hidden_size)."""
        image_features = self._extract_image_features(pixel_values, grid_thws)
        if self.mm_projector:
            image_features = self.mm_projector(image_features)
        return image_features
