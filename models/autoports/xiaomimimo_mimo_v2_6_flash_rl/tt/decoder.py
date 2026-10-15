# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo decoder layer: RMSNorm -> attention -> residual -> RMSNorm -> (dense MLP | MoE) -> residual.

The residual stream is replicated on every chip of the row (each sub-block ends in an all-reduce).
"""

import ttnn

from .attention import TtMiMoAttention
from .common import TILE, as_weight, cache_name, compute_config
from .moe import TtMiMoMoE


class TtRMSNorm:
    def __init__(self, plan, weight, eps, cache_file):
        w = None
        if weight is not None:
            w = weight.float().reshape(1, 1, -1, TILE).bfloat16()
        self.weight = as_weight(w, plan, plan.replicate(), ttnn.bfloat16, cache_file, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.eps = eps
        self.config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False)

    def __call__(self, x):
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.config,
        )


class TtMiMoDenseMLP:
    """SwiGLU MLP (layer 0): gate/up column-parallel, down row-parallel, all-reduce."""

    def __init__(self, plan, layer_idx, weights=None, cache_dir=None, weight_dtype=ttnn.bfloat8_b):
        self.plan = plan
        prefix = f"layer{layer_idx}_mlp_tp{plan.tp}"
        g = u = d = None
        if weights is not None:
            g = weights["gate_proj"].T.contiguous()
            u = weights["up_proj"].T.contiguous()
            d = weights["down_proj"].T.contiguous()
        self.w_gate = as_weight(g, plan, plan.shard(-1), weight_dtype, cache_name(cache_dir, prefix, "w_gate"))
        self.w_up = as_weight(u, plan, plan.shard(-1), weight_dtype, cache_name(cache_dir, prefix, "w_up"))
        self.w_down = as_weight(d, plan, plan.shard(-2), weight_dtype, cache_name(cache_dir, prefix, "w_down"))
        self.config = compute_config(ttnn.MathFidelity.HiFi2)

    def forward(self, x, token_mask=None):
        kw = dict(dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.config)
        gate = ttnn.linear(x, self.w_gate, **kw)
        up = ttnn.linear(x, self.w_up, **kw)
        act = ttnn.mul(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(act, self.w_down, **kw)
        act.deallocate(True)
        return self.plan.all_reduce(out)


class TtMiMoDecoderLayer:
    def __init__(
        self,
        plan,
        cfg,
        layer_idx,
        weights,
        rope_setups,
        cache_dir=None,
        attn_dtype=ttnn.bfloat8_b,
        mlp_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat8_b,
    ):
        """``weights``: dict from reference.model.load_layer_weights, or None to load from a warm cache.
        ``rope_setups``: {"swa": MiMoRotarySetup, "full": MiMoRotarySetup}."""
        self.layer_idx = layer_idx
        self.is_swa = cfg.is_swa(layer_idx)
        self.is_moe = cfg.is_moe(layer_idx)
        w = weights or {}
        prefix = f"layer{layer_idx}"
        eps = cfg.layernorm_epsilon
        self.input_norm = TtRMSNorm(plan, w.get("input_layernorm"), eps, cache_name(cache_dir, prefix, "input_norm"))
        self.post_norm = TtRMSNorm(
            plan, w.get("post_attention_layernorm"), eps, cache_name(cache_dir, prefix, "post_attn_norm")
        )
        self.rope = rope_setups["swa" if self.is_swa else "full"]
        self.attn = TtMiMoAttention(
            plan, cfg, layer_idx, w.get("attn"), self.rope, cache_dir=cache_dir, weight_dtype=attn_dtype
        )
        if self.is_moe:
            self.mlp = TtMiMoMoE(
                plan, cfg, layer_idx, w.get("router"), w.get("experts"), cache_dir=cache_dir, expert_dtype=expert_dtype
            )
        else:
            self.mlp = TtMiMoDenseMLP(plan, layer_idx, w.get("mlp"), cache_dir=cache_dir, weight_dtype=mlp_dtype)

    def _ffn(self, x, token_mask):
        h = self.post_norm(x)
        out = self.mlp.forward(h, token_mask)
        h.deallocate(True)
        y = ttnn.add(x, out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        return y

    def forward_prefill(self, x, user_id, page_table, kv_cache, token_mask=None):
        h = self.input_norm(x)
        attn = self.attn.forward_prefill(h, user_id, page_table, kv_cache)
        h.deallocate(True)
        x2 = ttnn.add(x, attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        y = self._ffn(x2, token_mask)
        x2.deallocate(True)
        return y

    def forward_decode(self, x, rot_mats, cur_pos, page_table, kv_cache, token_mask=None):
        h = self.input_norm(x)
        attn = self.attn.forward_decode(h, rot_mats[self.is_swa], cur_pos, page_table, kv_cache)
        h.deallocate(True)
        x2 = ttnn.add(x, attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        y = self._ffn(x2, token_mask)
        x2.deallocate(True)
        return y
