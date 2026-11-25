# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One GLM-5.3-Flash decoder layer on a TP stage (modeling_glm5_next.py:1278-1348).

    streams (bf16 RM [1, 1, T, hc * D], HF's [B, S, hc, D] memory order)
      -> attn_hc.pre -> input_layernorm -> KDA | MLA -> attn_hc.post  -> bf16 RM streams
      -> ffn_hc.pre  -> post_attention_layernorm -> MLP | MoE -> ffn_hc.post -> bf16 RM streams

The residual streams are rounded to bf16 at each sublayer boundary exactly where HF rounds them (it carries them in
the model dtype); the mHC projection / Sinkhorn / collapse / mix run in fp32 TILE inside the sublayer.
"""

from __future__ import annotations

import ttnn
from models.experimental.glm53_flash.tt.hyperconnection import HyperConnection

from .common import ChunkInfo, TPContext, WeightLoader, to_rm, to_tile
from .ffn import MoEEP, SwiGLUTP
from .kda import KDAPrefill
from .mla import MLAPrefill, MLAState


class DecoderLayer:
    def __init__(self, ctx: TPContext, cfg, layer_idx: int, src, cache_dir=None):
        self.ctx = ctx
        self.idx = layer_idx
        self.eps = cfg.rms_norm_eps
        self.is_kda = cfg.layer_types[layer_idx] == "linear_attention"
        self.is_moe = cfg.mlp_layer_types[layer_idx] == "sparse"
        load = WeightLoader(ctx, cache_dir, f"L{layer_idx}")

        def hc(name):
            fn_t = load(f"hc_{name}.fn_t", lambda: src[f"hc_{name}_fn"].t())
            return HyperConnection(
                ctx.device,
                fn_t,
                src[f"hc_{name}_base"].float(),
                src[f"hc_{name}_scale"].float(),
                hc_mult=cfg.hc_mult,
                hidden_size=cfg.hidden_size,
                sinkhorn_iters=cfg.hc_sinkhorn_iters,
                hc_eps=cfg.hc_eps,
                norm_eps=cfg.rms_norm_eps,
            )

        self.attn_hc = hc("attn")
        self.ffn_hc = hc("ffn")
        gamma = lambda k: (lambda: src[k].float().reshape(1, 1, 1, -1))
        self.attn_norm = load("input_layernorm", gamma("input_layernorm.weight"))
        self.ffn_norm = load("post_attention_layernorm", gamma("post_attention_layernorm.weight"))
        self.attn = KDAPrefill(ctx, cfg, src, load) if self.is_kda else MLAPrefill(ctx, cfg, src, load)
        if self.is_moe:
            self.mlp = MoEEP(ctx, cfg, src, load)
        else:
            self.mlp = SwiGLUTP(ctx, src, "mlp", load, cfg.swiglu_limit, "mlp")

    def new_state(self):
        return self.attn.new_state() if self.is_kda else MLAState()

    def _sublayer(self, streams_rm, hcon, norm, body):
        x = to_tile(streams_rm, ttnn.float32)
        h, post, comb = hcon.pre(x)
        h = ttnn.typecast(h, ttnn.bfloat16)
        h = ttnn.rms_norm(h, weight=norm, epsilon=self.eps, compute_kernel_config=self.ctx.hifi4)
        y = body(h)
        out = hcon.post(y, x, post, comb)
        ttnn.deallocate(x)
        return to_rm(out, ttnn.bfloat16)

    def forward(self, streams_rm: ttnn.Tensor, state, info: ChunkInfo):
        """streams_rm bf16 RM [1, 1, T, hc * D] -> (bf16 RM [1, 1, T, hc * D], new state)."""
        new_state = [None]

        def attn(h):
            if self.is_kda:
                y, new_state[0] = self.attn.forward(h, state, info.n_valid, info.valid())
            else:
                y, new_state[0] = self.attn.forward(h, state, info)
            return y

        s = self._sublayer(streams_rm, self.attn_hc, self.attn_norm, attn)
        s = self._sublayer(s, self.ffn_hc, self.ffn_norm, lambda h: self.mlp.forward(h, info.valid()))
        return s, new_state[0]
