# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AudioConformerLayer (HF `Gemma4AudioLayer`, modeling_gemma4.py:531-579), Macaron order:

    x = FFN1(x)                                              (half-step, residual inside)
    x = x + RMSNorm_post_attn( attn( RMSNorm_pre_attn(x) ) )
    x = LightConv1d(x)                                       (residual inside)
    x = FFN2(x)
    x = RMSNorm_out(x)

The three `clamp(+-gradient_clipping)` calls of HF (+-1e10) are not emitted: activations here are
bounded by the clippable-linear clamps and the norms, many orders of magnitude below 1e10 (the
tests log the largest |activation| HF produces).

Contract
    input   x [B, 1, S, 1024] act_dtype TILE (DRAM or L1), mask [B, 1, S, 64] fp32 TILE DRAM
            (attention.attention_mask)
    output  [B, 1, S, 1024] act_dtype TILE (act_dtype bf16 by default, see AudioModel), L1 interleaved
            when it fits (config.audio_act_memory_config, E5c) else DRAM
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.audio.attention import AudioAttention
from models.demos.embedding_gemma2.tt.audio.config import AudioConfig, audio_act_memory_config
from models.demos.embedding_gemma2.tt.audio.feed_forward import AudioFeedForward
from models.demos.embedding_gemma2.tt.audio.light_conv import AudioLightConv1d
from models.demos.embedding_gemma2.tt.audio.norm import AudioRMSNorm
from models.demos.embedding_gemma2.tt.precision import AudioPrecision, resolve_precision
from models.demos.embedding_gemma2.tt.weights import substate


class AudioConformerLayer(LightweightModule):
    def __init__(
        self,
        layer_idx: int,
        ffn1: AudioFeedForward,
        norm_pre_attn: AudioRMSNorm,
        attn: AudioAttention,
        norm_post_attn: AudioRMSNorm,
        lconv: AudioLightConv1d,
        ffn2: AudioFeedForward,
        norm_out: AudioRMSNorm,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ffn1 = ffn1
        self.norm_pre_attn = norm_pre_attn
        self.attn = attn
        self.norm_post_attn = norm_post_attn
        self.lconv = lconv
        self.ffn2 = ffn2
        self.norm_out = norm_out

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: AudioConfig,
        layer_idx: int,
        device,
        *,
        cache_dir: Path | None = None,
        act_dtype=None,
        shifts: dict | None = None,
        precision: AudioPrecision | None = None,
    ) -> "AudioConformerLayer":
        """`state_dict` holds the HF `audio_tower.layers.<i>.` sub-state (`.linear.` folded). `shifts`:
        the lconv shift matrices shared by all layers (light_conv.load_causal_shifts). `precision`:
        the audio policy (None: the default precision file); `act_dtype` (not None) overrides the
        output dtype of the activation-stream linears, see AudioModel."""
        precision = precision or resolve_precision(None).audio
        name = f"audio.layers.{layer_idx}"
        kw = dict(cache_dir=cache_dir)
        akw = dict(act_dtype=act_dtype, layer_idx=layer_idx, precision=precision, **kw)

        def norm(key):
            return AudioRMSNorm.from_state_dict(
                substate(state_dict, key), config.norm(), device, precision.norms, name=f"{name}.{key}", **kw
            )

        return cls(
            layer_idx,
            AudioFeedForward.from_state_dict(
                substate(state_dict, "feed_forward1"), config, device, name=f"{name}.ffn1", **akw
            ),
            norm("norm_pre_attn"),
            AudioAttention.from_state_dict(
                substate(state_dict, "self_attn"), config.attention, device, name=f"{name}.self_attn", **akw
            ),
            norm("norm_post_attn"),
            AudioLightConv1d.from_state_dict(
                substate(state_dict, "lconv1d"), config, device, name=f"{name}.lconv1d", shifts=shifts, **akw
            ),
            AudioFeedForward.from_state_dict(
                substate(state_dict, "feed_forward2"), config, device, name=f"{name}.ffn2", **akw
            ),
            norm("norm_out"),
        )

    def forward(self, x: ttnn.Tensor, mask: ttnn.Tensor, shift_idx: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """`shift_idx`: the attention's rel-shift gather index, built once per forward (AudioModel)."""
        # E5c: the residual stream in L1 when it fits (config.audio_act_memory_config)
        mc = audio_act_memory_config(x.device(), x.shape, x.dtype)
        h1 = self.ffn1(x)
        a_in = self.norm_pre_attn(h1, memory_config=mc)
        a = self.attn(a_in, mask, shift_idx)
        ttnn.deallocate(a_in)
        a_n = self.norm_post_attn(a, memory_config=mc)
        ttnn.deallocate(a)
        h2 = ttnn.add(h1, a_n, memory_config=mc)
        ttnn.deallocate(h1)
        ttnn.deallocate(a_n)
        h3 = self.lconv(h2)
        ttnn.deallocate(h2)
        h4 = self.ffn2(h3)
        ttnn.deallocate(h3)
        out = self.norm_out(h4, memory_config=mc)
        ttnn.deallocate(h4)
        return out
