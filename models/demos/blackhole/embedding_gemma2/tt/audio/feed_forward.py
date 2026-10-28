# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AudioFeedForward (HF `Gemma4AudioFeedForward`, modeling_gemma4.py:421-453), the Macaron half-step FFN.

    h   = RMSNorm_pre(x)                                   (clamp +-1e10 before it: a no-op, see README)
    h   = ClippableLinear_2( silu( ClippableLinear_1(h) ) )     1024 -> 4096 -> 1024, no biases
    out = x + residual_weight * RMSNorm_post(h)            residual_weight 0.5

Contract
    input   x [B, 1, S, 1024] bf16 TILE (DRAM or L1)
    output  [B, 1, S, 1024] bf16 TILE, L1 when it fits (config.audio_act_memory_config) else DRAM
            (row-wise: padded rows stay independent of valid rows)
E5c: silu and ffw2's input clamp run as one unary_chain; the 0.5 residual scale is the residual
add's pre-activation (bit-identical to the separate ops).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.audio.clippable_linear import ClippableLinear
from models.demos.blackhole.embedding_gemma2.tt.audio.config import AudioConfig, audio_act_memory_config
from models.demos.blackhole.embedding_gemma2.tt.audio.norm import AudioRMSNorm
from models.demos.blackhole.embedding_gemma2.tt.precision import AudioPrecision, resolve_precision
from models.demos.blackhole.embedding_gemma2.tt.weights import substate


class AudioFeedForward(LightweightModule):
    def __init__(
        self,
        config: AudioConfig,
        pre_norm: AudioRMSNorm,
        ffw1: ClippableLinear,
        ffw2: ClippableLinear,
        post_norm: AudioRMSNorm,
    ):
        super().__init__()
        self.config = config
        self.pre_norm = pre_norm
        self.ffw1 = ffw1
        self.ffw2 = ffw2
        self.post_norm = post_norm

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: AudioConfig,
        device,
        *,
        cache_dir: Path | None = None,
        name: str = "audio.feed_forward",
        act_dtype=None,
        layer_idx: int = 0,
        precision: AudioPrecision | None = None,
    ) -> "AudioFeedForward":
        """`state_dict` holds the HF `layers.<i>.feed_forward{1,2}.` sub-state. `precision`: the audio
        policy (None: the default precision file) as layer `layer_idx` (`ffn_up` / `ffn_down`, norms).
        `act_dtype` (not None) overrides the output dtype of the two linears (and so of the residual
        stream), see AudioModel."""
        precision = precision or resolve_precision(None).audio
        H, I = config.hidden_size, config.intermediate_size
        kw = dict(cache_dir=cache_dir)
        pre = AudioRMSNorm.from_state_dict(
            substate(state_dict, "pre_layer_norm"), config.norm(), device, precision.norms, name=f"{name}.pre", **kw
        )
        post = AudioRMSNorm.from_state_dict(
            substate(state_dict, "post_layer_norm"), config.norm(), device, precision.norms, name=f"{name}.post", **kw
        )
        ffw1 = ClippableLinear.from_state_dict(
            substate(state_dict, "ffw_layer_1"),
            H,
            I,
            device,
            name=f"{name}.ffw1",
            out_dtype=act_dtype,
            policy=precision.matmul("ffn_up", layer_idx),
            **kw,
        )
        ffw2 = ClippableLinear.from_state_dict(
            substate(state_dict, "ffw_layer_2"),
            I,
            H,
            device,
            name=f"{name}.ffw2",
            out_dtype=act_dtype,
            policy=precision.matmul("ffn_down", layer_idx),
            **kw,
        )
        return cls(config, pre, ffw1, ffw2, post)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # E5c: intermediates in L1 when they fit (config.audio_act_memory_config)
        mc = audio_act_memory_config(x.device(), x.shape, x.dtype)
        h = self.pre_norm(x, memory_config=mc)
        a = self.ffw1(h)
        ttnn.deallocate(h)
        # silu and ffw2's input clamp in one SFPU pass (same ops, same order)
        s = ttnn.unary_chain(
            a,
            [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), self.ffw2.input_clamp()],
            memory_config=audio_act_memory_config(a.device(), a.shape, a.dtype),
        )
        ttnn.deallocate(a)
        d = self.ffw2.project(s)
        ttnn.deallocate(s)
        n = self.post_norm(d, memory_config=mc)
        ttnn.deallocate(d)
        # x + residual_weight * n with the scale as the add's SFPU pre-activation on n
        scale = ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.config.residual_weight)
        out = ttnn.add(x, n, input_tensor_b_activations=[scale], memory_config=mc)
        ttnn.deallocate(n)
        return out
