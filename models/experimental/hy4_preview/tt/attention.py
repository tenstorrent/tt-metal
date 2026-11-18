# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview attention sublayer on the SC1 layout (tenstorrent/tt-metal#65885): the six MLA
projections as 2D tensor parallelism, the sigmoid output gate, the per-head sink, and a dense
attention core.

Mirrors the pure-torch ground truth in ../reference/attention.py method for method.

Layout (tt/common.py MeshPlan; 4x8: hidden axis = 4 rows, head axis = 8 columns). Every
projection is sharded once, 1/32 of its weight per chip, with no replica on either axis:

    weight    in -> out              in split over     out split over    then
    qkv_a     6144 -> 2048 + 576     hidden axis       head axis         reduce(hidden), gather(head)
    q_b       2048 -> 64 x 256       hidden axis       head axis (8 heads)   \
    gate      6144 -> 64 x 256       hidden axis       head axis (8 heads)    } one reduce(hidden)
    kv_b       512 -> 64 x 448       hidden axis       head axis (8 heads)   /
    o_proj    64 x 256 -> 6144       head axis         hidden axis       reduce(head)

* qkv_a is q_a_proj and kv_a_proj_with_mqa stacked on their output dim, so one matmul and one
  reduce + gather leave q_a's output and the 576-wide latent whole on every chip. That is where
  the two norms and the cache want them: both norms are chip-local, and the latent a token adds to
  the cache is complete on whichever chip owns its block (#65886).
* q_b and kv_b take their low-rank input split over the hidden axis. It is whole on every chip
  after qkv_a, so each chip keeps its own slice with mesh_partition, which does not use the fabric.
* The q_b, gate and kv_b partials are concatenated and summed over the hidden axis in ONE
  all-reduce.
* o_proj contracts over the heads, so its partials sum over the head axis and the output lands
  split over the hidden axis: the layout the residual streams have.

So a sublayer costs four collectives besides whatever the attention core needs: reduce(hidden),
gather(head), reduce(hidden), reduce(head).

The attention core here is dense and head-parallel: each chip expands keys and values for its own
8 heads from the latent of every position and attends over all of them. It exists to validate the
projections, gate and sink against the reference; prefill sparse attention (#65887) and
context-parallel sparse decode (#65889) replace it, on the same weights. Which positions a query
may attend to arrives as an additive mask.

Tensors are [1, 1, T, width] with the T tokens down the tile rows, or [1, heads, T, width] inside
the core. The caller applies the layer's pre-norm before and iHC's hc_post after.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.hy4_preview.tt.common import MeshPlan, compute_config


def _cols(x, start, end):
    """Columns [start, end) of the last dim."""
    shape = list(x.shape)
    return ttnn.slice(x, [0, 0, 0, start], [shape[0], shape[1], shape[2], end])


def _rope(x, cos, sin):
    """Half-split RoPE on the last dim: x * cos + rotate_half(x) * sin, rotate_half(x) = (-x2 | x1).

    x: [1, heads, T, d]; cos / sin: [1, 1, T, d], broadcast over the heads.
    """
    d = x.shape[-1]
    rotated = ttnn.concat([ttnn.neg(_cols(x, d // 2, d)), _cols(x, 0, d // 2)], dim=-1)
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))


class TtHy4Attention(LightweightModule):
    """One attention sublayer. ``weights`` holds the nine tensors by checkpoint name, as
    reference.attention.Hy4Attention.weights() returns them (linear weights are [out, in]).

    ``weight_dtype`` is the dtype of the six projections; activations are ``dtype``.
    """

    def __init__(self, plan: MeshPlan, cfg, weights: dict, weight_dtype=ttnn.bfloat8_b, dtype=ttnn.bfloat16):
        self.plan = plan
        self.cfg = cfg
        self.dtype = dtype
        c, htp, ctp = cfg, plan.hidden_tp, plan.head_tp
        self.heads = c.num_heads // ctp  # heads per chip

        self.qkv_a_dim = c.q_lora_rank + c.latent_dim
        for what, size, tp in (
            ("hidden_size", c.hidden_size, htp),
            ("q_lora_rank", c.q_lora_rank, htp),
            ("kv_lora_rank", c.kv_lora_rank, htp),
            ("num_heads", c.num_heads, ctp),
            ("q_lora_rank + kv_lora_rank + qk_rope_head_dim", self.qkv_a_dim, ctp),
        ):
            assert size % tp == 0, f"{what} = {size} does not split {tp} ways"
        for what, size in (("q_lora_rank", c.q_lora_rank), ("kv_lora_rank", c.kv_lora_rank)):
            assert htp == 1 or (size // htp) % ttnn.TILE_SIZE == 0, (
                f"{what} = {size} over {htp} hidden slices is {size // htp} wide; "
                f"mesh_partition needs a multiple of {ttnn.TILE_SIZE}"
            )

        # BFP8 keeps 7 mantissa bits, which HiFi2 covers; BF16 weights take HiFi4.
        fidelity = ttnn.MathFidelity.HiFi2 if weight_dtype == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi4
        self.ckc = compute_config(fidelity, fp32_acc=True, packer_l1_acc=True)
        self.core_ckc = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=True)

        def linear(w, in_axis):
            """An [out, in] weight as [1, 1, in, out], its input split over ``in_axis`` and its
            output over the other axis."""
            hidden_dim, head_dim = (2, 3) if in_axis == "hidden" else (3, 2)
            return ttnn.from_torch(
                w.float().t().contiguous().reshape(1, 1, w.shape[1], w.shape[0]),
                layout=ttnn.TILE_LAYOUT,
                device=plan.mesh_device,
                dtype=weight_dtype,
                mesh_mapper=plan.shard(hidden_dim=hidden_dim, head_dim=head_dim),
            )

        def replicated(t, shape):
            return ttnn.from_torch(
                t.float().contiguous().reshape(shape), layout=ttnn.TILE_LAYOUT, device=plan.mesh_device, dtype=dtype
            )

        self.w_qkv_a = linear(torch.cat([weights["q_a_proj"], weights["kv_a_proj_with_mqa"]], dim=0), "hidden")
        self.w_q_b = linear(weights["q_b_proj"], "hidden")
        self.w_gate = linear(weights["linear_gate"], "hidden")
        self.w_kv_b = linear(weights["kv_b_proj"], "hidden")
        self.w_o = linear(weights["o_proj"], "head")
        self.q_a_norm = replicated(weights["q_a_layernorm"], (1, 1, 1, c.q_lora_rank))
        self.kv_a_norm = replicated(weights["kv_a_layernorm"], (1, 1, 1, c.kv_lora_rank))
        # One sink logit per head, each chip holding its own heads': [1, heads, 1, 1].
        self.sinks = ttnn.from_torch(
            weights["learnable_sink_param"].float().reshape(1, c.num_heads, 1, 1),
            layout=ttnn.TILE_LAYOUT,
            device=plan.mesh_device,
            dtype=dtype,
            mesh_mapper=plan.shard(head_dim=1),
        )

    def _matmul(self, x, w):
        return ttnn.matmul(x, w, compute_kernel_config=self.ckc, dtype=self.dtype)

    def _to_heads(self, x, width):
        """[1, 1, T, heads * width] head-major -> [1, heads, T, width]."""
        T = x.shape[-2]
        return ttnn.permute(ttnn.reshape(x, [1, T, self.heads, width]), (0, 2, 1, 3))

    def latent(self, x, cos, sin):
        """x: [1,1,T,hidden/htp] -> (q_resid [1,1,T,q_lora], k_pass [1,1,T,kv_lora], k_rope [1,1,T,rope]),
        each whole and identical on every chip.

        (k_pass | k_rope) is the per-token cache entry; q_resid also feeds the indexer.
        """
        c, plan = self.cfg, self.plan
        a = self._matmul(x, self.w_qkv_a)  # this chip's hidden slice x this chip's output slice
        a = plan.all_reduce(a, plan.hidden_axis)
        a = plan.all_gather(a, dim=3, axis=plan.head_axis)  # [1,1,T, q_lora + kv_lora + rope]
        q_a = _cols(a, 0, c.q_lora_rank)
        kv_pass = _cols(a, c.q_lora_rank, c.q_lora_rank + c.kv_lora_rank)
        k_rope = _cols(a, c.q_lora_rank + c.kv_lora_rank, self.qkv_a_dim)
        q_resid = ttnn.rms_norm(q_a, epsilon=c.norm_eps, weight=self.q_a_norm)
        k_pass = ttnn.rms_norm(kv_pass, epsilon=c.norm_eps, weight=self.kv_a_norm)
        return q_resid, k_pass, _rope(k_rope, cos, sin)

    def head_projections(self, x, q_resid, k_pass):
        """The three projections whose output is per head, for this chip's heads:
        (q [1,1,T,heads*qk], gate logits [1,1,T,heads*v], kv [1,1,T,heads*(nope+v)]), each head-major.

        One all-reduce over the hidden axis carries all three.
        """
        c, plan, h = self.cfg, self.plan, self.heads
        q_p = self._matmul(plan.partition(q_resid, 3, plan.hidden_axis), self.w_q_b)
        g_p = self._matmul(x, self.w_gate)
        kv_p = self._matmul(plan.partition(k_pass, 3, plan.hidden_axis), self.w_kv_b)
        q_w, g_w, kv_w = h * c.qk_head_dim, h * c.v_head_dim, h * c.kv_b_head_dim
        if plan.hidden_tp == 1:
            return q_p, g_p, kv_p
        both = plan.all_reduce(ttnn.concat([q_p, g_p, kv_p], dim=-1), plan.hidden_axis)
        return _cols(both, 0, q_w), _cols(both, q_w, q_w + g_w), _cols(both, q_w + g_w, q_w + g_w + kv_w)

    def queries(self, q, cos, sin):
        """[1,1,T,heads*qk] -> [1,heads,T,qk], RoPE applied to the trailing rope dims."""
        c = self.cfg
        q = self._to_heads(q, c.qk_head_dim)
        q_rope = _rope(_cols(q, c.qk_nope_head_dim, c.qk_head_dim), cos, sin)
        return ttnn.concat([_cols(q, 0, c.qk_nope_head_dim), q_rope], dim=-1)

    def keys_values(self, kv, k_rope):
        """kv [1,1,T,heads*(nope+v)], k_rope [1,1,T,rope] -> (keys [1,heads,T,qk], values [1,heads,T,v]).

        k_rope is shared by every head.
        """
        c = self.cfg
        kv = self._to_heads(kv, c.kv_b_head_dim)
        keys = ttnn.concat([_cols(kv, 0, c.qk_nope_head_dim), ttnn.repeat(k_rope, [1, self.heads, 1, 1])], dim=-1)
        return keys, _cols(kv, c.qk_nope_head_dim, c.kv_b_head_dim)

    def core(self, q, keys, values, mask):
        """Dense attention with a sink: [1,heads,S,qk] x [1,heads,T,qk] -> [1,heads,S,v].

        ``mask`` is additive, [1,1,S,T]. The sink is one more logit per head that takes probability
        mass and contributes no value, so it is appended as a column and dropped after the softmax.
        """
        S, T = q.shape[-2], keys.shape[-2]
        logits = ttnn.matmul(q, ttnn.transpose(keys, -2, -1), compute_kernel_config=self.core_ckc)
        logits = ttnn.add(ttnn.multiply(logits, self.cfg.scaling), mask)
        combined = ttnn.concat([logits, ttnn.repeat(self.sinks, [1, 1, S, 1])], dim=-1)  # [1,heads,S,T+1]
        probs = ttnn.softmax(combined, dim=-1)
        probs = ttnn.slice(probs, [0, 0, 0, 0], [1, self.heads, S, T])
        return ttnn.matmul(probs, values, compute_kernel_config=self.core_ckc)

    def output(self, attn, gate):
        """attn [1,heads,T,v], gate logits [1,1,T,heads*v] -> [1,1,T,hidden/htp].

        (attn * sigmoid(gate)) @ o_proj: this chip's heads into this chip's hidden slice, summed
        over the head axis.
        """
        T = attn.shape[-2]
        attn = ttnn.reshape(ttnn.permute(attn, (0, 2, 1, 3)), [1, 1, T, self.heads * self.cfg.v_head_dim])
        gated = ttnn.multiply(attn, ttnn.sigmoid(gate))
        return self.plan.all_reduce(self._matmul(gated, self.w_o), self.plan.head_axis)

    def forward(self, x, cos, sin, mask):
        """x: [1,1,T,hidden/htp]; cos / sin: [1,1,T,rope]; mask: additive [1,1,T,T].

        Returns [1,1,T,hidden/htp], split over the hidden axis like x.
        """
        q_resid, k_pass, k_rope = self.latent(x, cos, sin)
        q, gate, kv = self.head_projections(x, q_resid, k_pass)
        q = self.queries(q, cos, sin)
        keys, values = self.keys_values(kv, k_rope)
        return self.output(self.core(q, keys, values, mask), gate)
