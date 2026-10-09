# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vision-tower arguments read from the snapshot ``config.json`` ``vision_config``. No weights are loaded here.

Patch-count buckets: the app's pixel budget (65536..262144 px, ``processor_config.json``) gives
256..1024 ViT patches per image. The tower pads the patch sequence to the next bucket of
256 / 512 / 768 / 1024 (multiples of the 128-row SDPA chunk, and of 4 * 32 so the 2x2 merger output
stays tile aligned) and masks the padded keys, so any patch count in 4..1024 is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

VISION_BUCKETS = (256, 512, 768, 1024)
TILE = 32


def vision_bucket_for(num_patches: int) -> int:
    """Smallest bucket holding ``num_patches`` ViT patches."""
    if num_patches < 4 or num_patches % 4:
        raise ValueError(f"{num_patches} patches is not a whole number of 2x2 merge blocks")
    for bucket in VISION_BUCKETS:
        if num_patches <= bucket:
            return bucket
    raise ValueError(f"{num_patches} patches exceed the largest vision bucket {VISION_BUCKETS[-1]}")


def round_up(value: int, multiple: int = TILE) -> int:
    return -(-value // multiple) * multiple


@dataclass(frozen=True)
class PplxVisionArgs:
    hidden_size: int  # 1152
    depth: int  # 27 blocks
    num_heads: int  # 16
    intermediate_size: int  # 4304
    hidden_act: str  # gelu_pytorch_tanh
    in_channels: int  # 3
    patch_size: int  # 16
    temporal_patch_size: int  # 2
    spatial_merge_size: int  # 2
    out_hidden_size: int  # 5120
    num_position_embeddings: int  # 2304 = 48 x 48
    rope_theta: float  # 1e4
    layer_norm_eps: float = 1e-6  # nn.LayerNorm(..., eps=1e-6) in every HF vision module

    @classmethod
    def from_hf_config(cls, vision_config) -> "PplxVisionArgs":
        rope = getattr(vision_config, "rope_parameters", None) or {}
        args = cls(
            hidden_size=vision_config.hidden_size,
            depth=vision_config.depth,
            num_heads=vision_config.num_heads,
            intermediate_size=vision_config.intermediate_size,
            hidden_act=vision_config.hidden_act,
            in_channels=vision_config.in_channels,
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            spatial_merge_size=vision_config.spatial_merge_size,
            out_hidden_size=vision_config.out_hidden_size,
            num_position_embeddings=vision_config.num_position_embeddings,
            rope_theta=float(rope.get("rope_theta", 10000.0)),
        )
        args.validate()
        return args

    def validate(self) -> None:
        # The TT modules are written for this exact vision config; fail rather than guess.
        expected = dict(
            hidden_size=1152,
            depth=27,
            num_heads=16,
            intermediate_size=4304,
            hidden_act="gelu_pytorch_tanh",
            in_channels=3,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=5120,
            num_position_embeddings=2304,
            rope_theta=10000.0,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"Unsupported vision {name}={getattr(self, name)}; expected {value}")

    @property
    def head_dim(self) -> int:  # 72
        return self.hidden_size // self.num_heads

    @property
    def padded_head_dim(self) -> int:  # 96: three tiles
        return round_up(self.head_dim)

    @property
    def rotary_dim(self) -> int:
        """HF ``Qwen3_5VisionRotaryEmbedding(head_dim // 2)``: 18 inv_freqs, each used for row and col, x2."""
        return self.head_dim

    @property
    def padded_intermediate_size(self) -> int:  # 4304 -> 4320 (134.5 -> 135 tiles)
        return round_up(self.intermediate_size)

    @property
    def patch_dim(self) -> int:  # 3 * 2 * 16 * 16 = 1536
        return self.in_channels * self.temporal_patch_size * self.patch_size**2

    @property
    def merge_unit(self) -> int:  # 4 patches per merged token
        return self.spatial_merge_size**2

    @property
    def merger_hidden_size(self) -> int:  # 4608
        return self.hidden_size * self.merge_unit

    @property
    def num_grid_per_side(self) -> int:  # 48
        return int(self.num_position_embeddings**0.5)
