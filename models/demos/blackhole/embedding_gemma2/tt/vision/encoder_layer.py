# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VisionEncoderLayer (HF `Gemma4VisionEncoderLayer`, modeling_gemma4.py:972-1013): sandwich norms.

    h = x + post_attn_norm( attn( input_norm(x), mask, cos, sin ) )
    h = h + post_ff_norm( mlp( pre_ff_norm(h) ) )                  mlp: down(gelu_tanh(gate h) * up h)

No per-layer input and no layer_scalar (unlike the text layer). The MLP is the E1 `MLP` module
(same math: bias-free gated gelu-tanh), loaded with the vision widths 768 -> 3072 -> 768.

Contract
    input   x [B, 1, S, 768] bf16 TILE DRAM; mask [B, 1, S, S]; cos/sin [B, 1, S, 64] (see attention.py)
    output  [B, 1, S, 768] bf16 TILE DRAM
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import RMSNormConfig
from models.demos.blackhole.embedding_gemma2.tt.mlp import MLP
from models.demos.blackhole.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.blackhole.embedding_gemma2.tt.vision.attention import VisionAttention
from models.demos.blackhole.embedding_gemma2.tt.vision.config import VisionLayerConfig
from models.demos.blackhole.embedding_gemma2.tt.vision.rope import AxialRope
from models.demos.blackhole.embedding_gemma2.tt.weights import substate

_NORMS = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")


class VisionEncoderLayer(LightweightModule):
    def __init__(self, config: VisionLayerConfig, norms: dict[str, RMSNorm], attention: VisionAttention, mlp: MLP):
        super().__init__()
        self.config = config
        self.input_layernorm = norms["input_layernorm"]
        self.post_attention_layernorm = norms["post_attention_layernorm"]
        self.pre_feedforward_layernorm = norms["pre_feedforward_layernorm"]
        self.post_feedforward_layernorm = norms["post_feedforward_layernorm"]
        self.self_attn = attention
        self.mlp = mlp

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: VisionLayerConfig,
        device,
        rope: AxialRope,
        *,
        cache_dir: Path | None = None,
    ) -> "VisionEncoderLayer":
        """`state_dict` holds the HF `vision_tower.encoder.layers.<i>.` sub-state."""
        prefix = f"vision.layers.{config.layer_idx}"
        norm_cfg = RMSNormConfig(dim=config.hidden_size, eps=config.rms_norm_eps)
        norms = {
            n: RMSNorm.from_state_dict(
                substate(state_dict, n), norm_cfg, device, cache_dir=cache_dir, name=f"{prefix}.{n}"
            )
            for n in _NORMS
        }
        attention = VisionAttention.from_state_dict(
            substate(state_dict, "self_attn"),
            config.attention,
            device,
            rope,
            cache_dir=cache_dir,
            name=f"{prefix}.self_attn",
        )
        mlp = MLP.from_state_dict(
            substate(state_dict, "mlp"), config.mlp, device, cache_dir=cache_dir, name=f"{prefix}.mlp"
        )
        return cls(config, norms, attention, mlp)

    def forward(self, x: ttnn.Tensor, mask: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        h = self.input_layernorm(x)
        a = self.self_attn(h, mask, cos, sin)
        ttnn.deallocate(h)
        a_n = self.post_attention_layernorm(a)
        ttnn.deallocate(a)
        h1 = ttnn.add(x, a_n, memory_config=dram)
        ttnn.deallocate(a_n)

        f = self.pre_feedforward_layernorm(h1)
        m = self.mlp(f)
        ttnn.deallocate(f)
        m_n = self.post_feedforward_layernorm(m)
        ttnn.deallocate(m)
        out = ttnn.add(h1, m_n, memory_config=dram)
        ttnn.deallocate(h1)
        ttnn.deallocate(m_n)
        return out
