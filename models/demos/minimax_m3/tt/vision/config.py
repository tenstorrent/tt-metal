# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
The MiniMax-M3 vision config: the dimensions of the vision tower and projector, read from the checkpoint's config.json
or a transformers config (from_hf_config) without importing transformers or ttnn.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MiniMaxM3VisionConfig:
    """Vision tower and projector dimensions. The defaults are the MiniMax-M3 checkpoint's."""

    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    projector_hidden_size: int = 6144
    text_hidden_size: int = 6144

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads  # 80

    @property
    def patch_dim(self) -> int:
        """Values per flattened patch: channels x frames x patch_size x patch_size (1176)."""
        return self.num_channels * self.temporal_patch_size * self.patch_size**2

    @property
    def merged_hidden_size(self) -> int:
        """Projector width after folding each block of spatial_merge_size**2 patches into one row (24576)."""
        return self.text_hidden_size * self.spatial_merge_size**2

    @classmethod
    def from_hf_config(cls, config: dict, **overrides) -> "MiniMaxM3VisionConfig":
        """Build from a config dict: the checkpoint's config.json (transformers 4.52.4 layout, with the merge and
        temporal sizes under ``img_token_compression_config``) or ``MiniMaxM3VLConfig.to_dict()`` (transformers 5.12).

        Raises ValueError for a vision ``hidden_act`` other than ``gelu``: Hugging Face's vision MLP reads it, and the
        TT MLP runs the exact (erf) GELU only.
        """
        vc = config["vision_config"]
        activation = vc.get("hidden_act", "gelu")
        if activation != "gelu":
            raise ValueError(f"vision_config hidden_act must be 'gelu', the exact (erf) GELU, got {activation!r}")
        compression = vc.get("img_token_compression_config") or config.get("img_token_compression_config") or {}
        rope = vc.get("rope_parameters") or {}
        values = dict(
            hidden_size=vc["hidden_size"],
            intermediate_size=vc["intermediate_size"],
            num_hidden_layers=vc["num_hidden_layers"],
            num_attention_heads=vc["num_attention_heads"],
            num_channels=vc.get("num_channels", 3),
            patch_size=vc["patch_size"],
            temporal_patch_size=vc.get("temporal_patch_size", compression.get("temporal_patch_size", 2)),
            spatial_merge_size=vc.get("spatial_merge_size", compression.get("spatial_merge_size", 2)),
            layer_norm_eps=vc.get("layer_norm_eps", 1e-5),
            rope_theta=rope.get("rope_theta", vc.get("rope_theta", 10000.0)),
            projector_hidden_size=config["projector_hidden_size"],
            text_hidden_size=config["text_config"]["hidden_size"],
        )
        values.update(overrides)
        return cls(**values)
