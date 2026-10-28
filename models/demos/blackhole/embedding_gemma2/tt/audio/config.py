# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Config for the EmbeddingGemma 2 audio path (HF `Gemma4AudioModel` + `embed_audio`).

Built from `config.json["audio_config"]` (a dict or the HF `Gemma4AudioConfig`) plus the text hidden
size (the `embed_audio` output width), so the TT modules never import transformers.

google/embeddinggemma-2: hidden 1024, 12 conformer layers, 8 heads x head_dim 128, silu, chunk 12,
context_left 13 (12 past keys + self), context_right 0, logit cap 50, invalid logit -1e9, conv
kernel 5, residual weight 0.5, subsampling channels [128, 32] on 128 mel bins, output_proj 1536
(+bias), rms eps 1e-6, clipped linears on (per-tensor bounds in the checkpoint), gradient clipping
1e10.

Physical buckets
    The processor emits T mel frames (100 per second, at most 2999 = 30 s) and the subsampling
    stack (two stride-2 convs) turns them into N = ceil(ceil(T / 2) / 2) tokens (at most 750).
    The device runs a fixed token bucket S (a multiple of the 32-row attention block) and a frame
    bucket 4 * S. Frames T..4S-1 and tokens N..S-1 are padding, masked like processor padding.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from models.demos.blackhole.embedding_gemma2.tt.config import RMSNormConfig, _get

TOKEN_BUCKETS = (32, 64, 128, 256, 512, 768)

# E5c: audio intermediates live in L1 (interleaved over every core's bank) when one tensor needs at
# most this many bytes per bank, else DRAM (the E5a text rule). The cap bounds the L1 the live
# intermediates take so matmul circular buffers (<= 900 KB per core, audio/linear.py) never clash
# with them. 0 puts everything in DRAM (A/B switch).
AUDIO_L1_BYTES_PER_BANK = int(os.environ.get("EG2_AUDIO_L1_BYTES_PER_BANK", str(64 * 1024)))
# A named wide intermediate (the fp32 fused QKV output, 72 KB per bank at 1 x 768) gets a wider
# cap, as the vision tower's (E5b).
WIDE_L1_BYTES_PER_BANK = 96 * 1024


def audio_act_memory_config(device, shape, dtype=None, bytes_per_bank: int | None = None) -> "ttnn.MemoryConfig":
    """L1 interleaved for a tile tensor of `shape` / `dtype` (bf16 when None) if it fits the per-bank
    cap (`bytes_per_bank`, default AUDIO_L1_BYTES_PER_BANK), else DRAM interleaved."""
    import ttnn

    cap = AUDIO_L1_BYTES_PER_BANK if bytes_per_bank is None or AUDIO_L1_BYTES_PER_BANK == 0 else bytes_per_bank
    n = math.prod(int(s) for s in shape) * (4 if dtype == ttnn.float32 else 2)
    grid = device.compute_with_storage_grid_size()
    if n <= cap * grid.x * grid.y:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


SUBSAMPLE = 4  # two stride-2 convs
ATTN_BLOCK = 32  # physical query block of the band attention (>= chunk_size, tile aligned)


def token_bucket_for(num_tokens: int) -> int:
    for b in TOKEN_BUCKETS:
        if num_tokens <= b:
            return b
    raise ValueError(f"{num_tokens} audio tokens exceed the largest bucket {TOKEN_BUCKETS[-1]}")


def tokens_for_frames(num_frames: int) -> int:
    """Token count after the two k3 s2 p1 convs (HF output mask = mask[::2][::2])."""
    return math.ceil(math.ceil(num_frames / 2) / 2)


@dataclass(frozen=True)
class ClipBounds:
    """Per-tensor `Gemma4ClippableLinear` bounds (python floats folded into ttnn.clamp)."""

    input_min: float
    input_max: float
    output_min: float
    output_max: float


@dataclass(frozen=True)
class AudioAttentionConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    chunk_size: int  # 12
    max_past_horizon: int  # attention_context_left - 1 = 12
    logit_cap: float  # 50
    invalid_logit: float  # -1e9

    @property
    def q_scale(self) -> float:
        """HF `Gemma4AudioAttention.q_scale` (before softplus(per_dim_scale))."""
        return (self.head_dim**-0.5) / math.log(2)

    @property
    def k_scale(self) -> float:
        return math.log(1 + math.e) / math.log(2)

    @property
    def num_rel_positions(self) -> int:
        """Rows of the sinusoidal relative table: distances max_past_horizon .. 0 (13)."""
        return self.max_past_horizon + 1


@dataclass(frozen=True)
class AudioConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    chunk_size: int
    context_left: int
    context_right: int
    logit_cap: float
    invalid_logit: float
    conv_kernel_size: int
    residual_weight: float
    subsampling_channels: tuple[int, int]
    num_mel_bins: int
    output_proj_dims: int
    rms_norm_eps: float
    gradient_clipping: float
    text_hidden_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def proj_input_dim(self) -> int:
        """Flattened subsampling output: (mel bins / 4) * channels[1] = 32 * 32."""
        return (self.num_mel_bins // SUBSAMPLE) * self.subsampling_channels[1]

    @property
    def max_frames(self) -> int:
        return SUBSAMPLE * TOKEN_BUCKETS[-1]

    @property
    def attention(self) -> AudioAttentionConfig:
        return AudioAttentionConfig(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            chunk_size=self.chunk_size,
            max_past_horizon=self.context_left - 1,
            logit_cap=self.logit_cap,
            invalid_logit=self.invalid_logit,
        )

    def norm(self, with_scale: bool = True, dim: int | None = None) -> RMSNormConfig:
        return RMSNormConfig(dim=dim or self.hidden_size, eps=self.rms_norm_eps, with_scale=with_scale)

    @classmethod
    def from_hf(cls, audio_cfg: Any, text_hidden_size: int, num_mel_bins: int = 128) -> "AudioConfig":
        cfg = audio_cfg
        if _get(cfg, "model_type", "gemma4_audio") != "gemma4_audio":
            raise ValueError(f"unsupported audio model_type {_get(cfg, 'model_type')}")
        if _get(cfg, "hidden_act") != "silu":
            raise ValueError(f"unsupported hidden_act {_get(cfg, 'hidden_act')}")
        if not _get(cfg, "use_clipped_linears", False):
            raise ValueError("use_clipped_linears=False is not supported (embeddinggemma-2 has it on)")
        if _get(cfg, "attention_context_right") != 0:
            raise ValueError("attention_context_right != 0 (look-ahead) is not supported")
        chunk, left = _get(cfg, "attention_chunk_size"), _get(cfg, "attention_context_left")
        # The band (0 <= q - kv < left - 1) must fit one 32-row block of past keys.
        if left - 1 > ATTN_BLOCK or chunk > ATTN_BLOCK:
            raise ValueError(f"chunk {chunk} / context_left {left} exceed the {ATTN_BLOCK}-row attention block")
        hidden, heads = _get(cfg, "hidden_size"), _get(cfg, "num_attention_heads")
        if hidden % heads != 0:
            raise ValueError(f"hidden {hidden} not divisible by {heads} heads")
        channels = tuple(_get(cfg, "subsampling_conv_channels"))
        if len(channels) != 2:
            raise ValueError(f"expected 2 subsampling conv layers, got {channels}")
        if num_mel_bins % SUBSAMPLE != 0:
            raise ValueError(f"mel bins {num_mel_bins} must be divisible by {SUBSAMPLE}")
        if (num_mel_bins // SUBSAMPLE) * channels[1] != (channels[0] // 4) * channels[1]:
            raise ValueError("HF sizes input_proj by channels[0] // 4; that must equal the flattened width")
        return cls(
            hidden_size=hidden,
            num_layers=_get(cfg, "num_hidden_layers"),
            num_heads=heads,
            intermediate_size=4 * hidden,
            chunk_size=chunk,
            context_left=left,
            context_right=0,
            logit_cap=float(_get(cfg, "attention_logit_cap")),
            invalid_logit=float(_get(cfg, "attention_invalid_logits_value")),
            conv_kernel_size=_get(cfg, "conv_kernel_size"),
            residual_weight=float(_get(cfg, "residual_weight")),
            subsampling_channels=channels,
            num_mel_bins=num_mel_bins,
            output_proj_dims=_get(cfg, "output_proj_dims"),
            rms_norm_eps=float(_get(cfg, "rms_norm_eps")),
            gradient_clipping=float(_get(cfg, "gradient_clipping")),
            text_hidden_size=text_hidden_size,
        )
