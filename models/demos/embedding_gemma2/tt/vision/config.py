# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Config for the EmbeddingGemma 2 vision path (HF `Gemma4VisionModel` + `embed_vision`).

Built from `config.json["vision_config"]` (a dict or the HF `Gemma4VisionConfig`) plus the text
hidden size (the `embed_vision` output width), so the TT modules never import transformers.

google/embeddinggemma-2: hidden 768, 16 layers, 12 heads x head_dim 64 (MHA), intermediate 3072,
gelu_pytorch_tanh, patch 16 (768 = 16*16*3 values per patch), 3x3 pooling, position table
[2, 10240, 768], axial rope theta 100, rms eps 1e-6, standardize off, no clipped linears.

Physical patch buckets
    The processor emits N = max_soft_tokens * k^2 patches per image (2520 for images, 1260 per
    video frame), padded with position (-1, -1). The device runs a fixed physical length S per
    bucket: 1280 (N <= 1280, video frames) or 2560 (N <= 2560, images). Both are multiples of the
    SDPA chunk (256 / 128). Rows N..S-1 are extra padding, masked like processor padding.
    Pooled outputs per image: S_out = round_up(S // k^2, 32) rows (160 / 288), of which
    n_valid_patches / k^2 are real soft tokens (the rest are stripped on the host).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.demos.embedding_gemma2.tt.config import MLPConfig, RMSNormConfig, _get

PATCH_BUCKETS = (1280, 2560)


# E5b: vision activations live in L1 (interleaved over every bank) when one tensor needs at most
# this many bytes per bank and the forward has at most this many patches (batch x bucket); the E5a
# text rule. One image (2560 patches): the [*, 768]-wide activations, heads and the residual stream
# go to L1 (-2.5 ms per forward); the 3072-wide MLP intermediates stay in DRAM unless named below.
# Whole-batch tensors of a batch > 1 forward stay in DRAM; only the per-image MLP (VisionMLP) puts
# one image's slice and MLP tensors in L1. 128 KB / 10240 patches (batch 4 in L1) failed: op
# circular buffers clash with L1 buffers.
VISION_L1_BYTES_PER_BANK = 64 * 1024
VISION_L1_MAX_TOKENS = 4096
# Wide intermediates named by the caller ("mlp": the up / gate projection outputs, "qkv": the fused
# QKV output) also go to L1 up to this many patches, beyond the per-bank cap (121 KB / 91 KB per
# bank at 2560 patches). The 128 KB cap for every tensor clashed with an op's circular buffers
# (CB region end 1160576 B vs an L1 buffer at 1142784 B); naming the tensors keeps the rest in DRAM.
# Batches above the "mlp" cap run the MLP one image at a time to reuse this path (VisionMLP).
VISION_L1_WIDE_MAX_TOKENS = {"mlp": 2560, "qkv": 2560}
# Per-grid additions to the named wide caps (E10.1, doc/prefill_towers/; "h": the MLP gelu(gate) * up
# product), batch 1 only, gated like the tuned matmul table (tt/vision/linear.py, EG2_VISION_MM_TUNED).
# VISION_L1_WIDE_OVERRIDES (in-process, empty by default) takes precedence on every grid (config search).
VISION_L1_WIDE_TUNED: dict[tuple[int, int], dict[str, int]] = {(12, 10): {"h": 2560}}  # with the 2D down config
VISION_L1_WIDE_OVERRIDES: dict[str, int] = {}


def _wide_cap(grid, wide: str, batch: int) -> int:
    from models.demos.embedding_gemma2.tt.vision import linear as vlinear

    if batch == 1 and wide in VISION_L1_WIDE_OVERRIDES:
        return VISION_L1_WIDE_OVERRIDES[wide]
    cap = VISION_L1_WIDE_MAX_TOKENS.get(wide, 0)
    if batch == 1 and vlinear.VISION_TUNED_ON:
        cap = VISION_L1_WIDE_TUNED.get((grid.x, grid.y), {}).get(wide, cap)
    return cap


def vision_act_memory_config(device, batch: int, seq_len: int, width: int, wide: str | None = None):
    """L1 interleaved for a bf16 [batch, 1, seq_len, width] TILE tensor when it fits the rule, else DRAM."""
    import ttnn

    grid = device.compute_with_storage_grid_size()
    if wide is not None and batch * seq_len <= _wide_cap(grid, wide, batch):
        return ttnn.L1_MEMORY_CONFIG
    per_bank, max_tokens = VISION_L1_BYTES_PER_BANK, VISION_L1_MAX_TOKENS
    n = 2 * int(batch) * int(seq_len) * int(width)
    if batch * seq_len <= max_tokens and n <= per_bank * grid.x * grid.y:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def patch_bucket_for(num_patches: int) -> int:
    for b in PATCH_BUCKETS:
        if num_patches <= b:
            return b
    raise ValueError(f"{num_patches} patches exceed the largest bucket {PATCH_BUCKETS[-1]}")


@dataclass(frozen=True)
class VisionAttentionConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    scaling: float = 1.0  # HF Gemma4VisionAttention.scaling

    @property
    def width(self) -> int:
        return self.num_heads * self.head_dim


@dataclass(frozen=True)
class VisionLayerConfig:
    layer_idx: int
    hidden_size: int
    rms_norm_eps: float
    attention: VisionAttentionConfig
    mlp: MLPConfig


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    patch_size: int
    pooling_kernel_size: int
    position_embedding_size: int
    rope_theta: float
    rms_norm_eps: float
    text_hidden_size: int  # embed_vision output width
    default_output_length: int  # soft tokens per image (280)

    @property
    def patch_dim(self) -> int:
        return 3 * self.patch_size**2

    @property
    def k_squared(self) -> int:
        return self.pooling_kernel_size**2

    @property
    def root_hidden_size(self) -> float:
        return self.hidden_size**0.5

    @property
    def max_image_patches(self) -> int:
        return self.default_output_length * self.k_squared

    def pooled_rows(self, seq_len: int) -> int:
        """Physical pooled-output rows for physical patch length `seq_len` (tile aligned)."""
        return round_up(seq_len // self.k_squared, 32)

    def layer(self, i: int) -> VisionLayerConfig:
        attn = VisionAttentionConfig(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
        )
        return VisionLayerConfig(
            layer_idx=i,
            hidden_size=self.hidden_size,
            rms_norm_eps=self.rms_norm_eps,
            attention=attn,
            mlp=MLPConfig(hidden_size=self.hidden_size, intermediate_size=self.intermediate_size),
        )

    def norm(self, with_scale: bool = True, dim: int | None = None) -> RMSNormConfig:
        return RMSNormConfig(dim=dim or self.hidden_size, eps=self.rms_norm_eps, with_scale=with_scale)

    @classmethod
    def from_hf(cls, vision_cfg: Any, text_hidden_size: int) -> "VisionConfig":
        cfg = vision_cfg
        if _get(cfg, "model_type", "gemma4_vision") != "gemma4_vision":
            raise ValueError(f"unsupported vision model_type {_get(cfg, 'model_type')}")
        if _get(cfg, "standardize", False):
            raise ValueError("standardize=True is not supported (embeddinggemma-2 has it off)")
        if _get(cfg, "use_clipped_linears", False):
            raise ValueError("use_clipped_linears=True is not supported (embeddinggemma-2 has it off)")
        if _get(cfg, "attention_bias", False):
            raise ValueError("attention_bias=True is not supported")
        if _get(cfg, "hidden_activation") != "gelu_pytorch_tanh":
            raise ValueError(f"unsupported hidden_activation {_get(cfg, 'hidden_activation')}")
        rope = _get(cfg, "rope_parameters")
        if _get(rope, "rope_type") != "axial":
            raise ValueError(f"unsupported vision rope_type {_get(rope, 'rope_type')}")
        heads, kv_heads = _get(cfg, "num_attention_heads"), _get(cfg, "num_key_value_heads")
        if heads != kv_heads:
            raise ValueError(f"vision attention is MHA here; got {heads} q heads, {kv_heads} kv heads")
        head_dim = _get(cfg, "head_dim")
        if head_dim % 4 != 0:
            raise ValueError(f"axial rope needs head_dim % 4 == 0, got {head_dim}")
        return cls(
            hidden_size=_get(cfg, "hidden_size"),
            num_layers=_get(cfg, "num_hidden_layers"),
            num_heads=heads,
            head_dim=head_dim,
            intermediate_size=_get(cfg, "intermediate_size"),
            patch_size=_get(cfg, "patch_size"),
            pooling_kernel_size=_get(cfg, "pooling_kernel_size"),
            position_embedding_size=_get(cfg, "position_embedding_size"),
            rope_theta=float(_get(rope, "rope_theta")),
            rms_norm_eps=_get(cfg, "rms_norm_eps"),
            text_hidden_size=text_hidden_size,
            default_output_length=_get(cfg, "default_output_length", 280),
        )
