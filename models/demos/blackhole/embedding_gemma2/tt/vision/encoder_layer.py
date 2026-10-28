# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VisionEncoderLayer (HF `Gemma4VisionEncoderLayer`, modeling_gemma4.py:972-1013): sandwich norms.

    h = x + post_attn_norm( attn( input_norm(x), mask, cos, sin ) )
    h = h + post_ff_norm( mlp( pre_ff_norm(h) ) )                  mlp: down(gelu_tanh(gate h) * up h)

No per-layer input and no layer_scalar (unlike the text layer). The MLP is `VisionMLP` (same math
as the text `MLP`, vision policy: see mlp.py), widths 768 -> 3072 -> 768.

E5b: the norm outputs, the attention / MLP outputs and the residual stream (h, out) are placed by
`vision_act_memory_config` (L1 for one image, DRAM for batched inputs).

Contract
    input   x [B, 1, S, 768] bf16 TILE (DRAM or L1); mask [B, 1, S, S]; cos/sin (see attention.py)
    output  [B, 1, S, 768] bf16 TILE, L1 when it fits the vision L1 rule, else DRAM
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import RMSNormConfig
from models.demos.blackhole.embedding_gemma2.tt.precision import VisionPrecision, resolve_precision
from models.demos.blackhole.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.blackhole.embedding_gemma2.tt.vision.attention import VisionAttention
from models.demos.blackhole.embedding_gemma2.tt.vision.config import VisionLayerConfig, vision_act_memory_config
from models.demos.blackhole.embedding_gemma2.tt.vision.mlp import VisionMLP
from models.demos.blackhole.embedding_gemma2.tt.vision.rope import AxialRope
from models.demos.blackhole.embedding_gemma2.tt.weights import substate

_NORMS = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")


class VisionEncoderLayer(LightweightModule):
    def __init__(
        self, config: VisionLayerConfig, norms: dict[str, RMSNorm], attention: VisionAttention, mlp: VisionMLP
    ):
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
        precision: VisionPrecision | None = None,
    ) -> "VisionEncoderLayer":
        """`state_dict` holds the HF `vision_tower.encoder.layers.<i>.` sub-state; `precision` the
        vision policy (None: the default precision file)."""
        precision = precision or resolve_precision(None).vision
        i = config.layer_idx
        prefix = f"vision.layers.{i}"
        norm_cfg = RMSNormConfig(dim=config.hidden_size, eps=config.rms_norm_eps)
        ckc_norm = precision.norms.compute_kernel_config
        norms = {
            n: RMSNorm.from_state_dict(
                substate(state_dict, n),
                norm_cfg,
                device,
                cache_dir=cache_dir,
                name=f"{prefix}.{n}",
                compute_kernel_config=ckc_norm,
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
            layer_idx=i,
            precision=precision,
        )
        mlp = VisionMLP.from_state_dict(
            substate(state_dict, "mlp"),
            config.mlp,
            device,
            cache_dir=cache_dir,
            name=f"{prefix}.mlp",
            layer_idx=i,
            precision=precision,
        )
        return cls(config, norms, attention, mlp)

    def forward(self, x: ttnn.Tensor, mask: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        B, S, D = x.shape[0], x.shape[2], x.shape[3]
        # [B, 1, S, 768] activations and the residual stream in L1 when they fit (vision_act_memory_config)
        mem = vision_act_memory_config(x.device(), B, S, D)
        h = self.input_layernorm(x, memory_config=mem)
        a = self.self_attn(h, mask, cos, sin, memory_config=mem)
        ttnn.deallocate(h)
        a_n = self.post_attention_layernorm(a, memory_config=mem)
        ttnn.deallocate(a)
        h1 = ttnn.add(x, a_n, memory_config=mem)
        ttnn.deallocate(a_n)

        f = self.pre_feedforward_layernorm(h1, memory_config=mem)
        m = self.mlp(f, memory_config=mem)
        ttnn.deallocate(f)
        m_n = self.post_feedforward_layernorm(m, memory_config=mem)
        ttnn.deallocate(m)
        out = ttnn.add(h1, m_n, memory_config=mem)
        ttnn.deallocate(h1)
        ttnn.deallocate(m_n)
        return out
