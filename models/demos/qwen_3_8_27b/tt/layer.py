# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One Qwen3.8 decoder layer, composed from the big blocks (outline pattern: gpt_oss_d_p/tt/layer.py).

    x = x + mixer(input_norm(x))          mixer = TtAttention (every 4th layer) | TtGatedDeltaNet
    x = x + mlp(post_attention_norm(x))

The residual stream is ``[1, 1, s_local, hidden]``: SP-sharded on the sequence (mesh rows),
replicated across TP (mesh columns). Every block closes with a TP all-reduce, so the residual adds
are local.
"""

from __future__ import annotations

import ttnn
from models.demos.qwen_3_8_27b.config import PrefillSpec, Qwen38Config, ttnn_dtype
from models.demos.qwen_3_8_27b.tt.attention import TtAttention
from models.demos.qwen_3_8_27b.tt.common import residual_dtype
from models.demos.qwen_3_8_27b.tt.gdn import TtGatedDeltaNet
from models.demos.qwen_3_8_27b.tt.mlp import TtMLP
from models.demos.qwen_3_8_27b.tt.rms_norm import TtRMSNorm


def sub(sd, prefix):
    if sd is None:
        return None
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


class TtDecoderLayer:
    def __init__(self, mesh_config, ccl, cfg: Qwen38Config, sd, layer_idx: int, spec: PrefillSpec, *, cache=None):
        """sd: HF tensors relative to ``layers.{i}.`` (or None when every weight is in the tensor cache)."""
        self.layer_idx = layer_idx
        self.is_full = cfg.is_full_attention(layer_idx)
        p = f"layers.{layer_idx}."
        eps = cfg.rms_norm_eps
        self.input_norm = TtRMSNorm(
            mesh_config, None if sd is None else sd["input_layernorm.weight"], eps, cache=cache, name=p + "input_norm"
        )
        self.post_norm = TtRMSNorm(
            mesh_config,
            None if sd is None else sd["post_attention_layernorm.weight"],
            eps,
            cache=cache,
            name=p + "post_norm",
        )
        if self.is_full:
            self.attn_ordinal = cfg.full_attention_layers.index(layer_idx)
            self.mixer = TtAttention(
                mesh_config,
                ccl,
                cfg,
                sub(sd, "self_attn."),
                attn_ordinal=self.attn_ordinal,
                weight_dtype=ttnn_dtype(spec.weight_dtype_attention),
                cache=cache,
                prefix=p + "attn.",
            )
        else:
            self.mixer = TtGatedDeltaNet(
                mesh_config,
                cfg,
                sub(sd, "linear_attn."),
                weight_dtype=ttnn_dtype(spec.weight_dtype_attention),
                cache=cache,
                prefix=p + "gdn.",
            )
        self.mlp = TtMLP(
            mesh_config,
            sub(sd, "mlp."),
            dtypes={w: ttnn_dtype(spec.weight_dtype_mlp(w)) for w in ("gate", "up", "down")},
            cache=cache,
            prefix=p + "mlp.",
        )

    def __call__(self, x, ctx, rope):
        if x.dtype != residual_dtype():
            x = ttnn.typecast(x, residual_dtype())
        h = self.input_norm(x)
        if self.is_full:
            m = self.mixer(h, ctx, rope)
        else:
            m = self.mixer(h, ctx.caches.gdn_state(ctx.user_id, self.layer_idx), valid_len=ctx.valid_len)
        ttnn.deallocate(h)
        x2 = ttnn.add(x, m)
        ttnn.deallocate(m)
        h = self.post_norm(x2)
        m = self.mlp(h)
        ttnn.deallocate(h)
        out = ttnn.add(x2, m)
        ttnn.deallocate(m)
        ttnn.deallocate(x2)
        return out
