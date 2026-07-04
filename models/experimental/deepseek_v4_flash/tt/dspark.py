# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn port of the DeepSeek-V4-Flash **DSpark** speculative-decoding module.

DSpark bolts an ``n_mtp``-stage speculative head onto the frozen V4-Flash stack
(the checkpoint's ``inference/model.py``). This module ports the *draft* forward
(``forward_spec``) to ttnn, gated for correctness against
``tests/dspark_reference.py``:

* :class:`DSparkAttention` — sliding-window MLA over the ``block_size`` draft
  query tokens. Unlike the main model's single-token fused SDPA-decode, all draft
  queries attend to the *same* key set (the seeded sliding cache + the draft
  block), so it is expressed as a small explicit gathered-softmax-with-sink here.
* :class:`DSparkBlock` — one MTP stage: reuses :class:`DeepSeekV4HyperConnection`
  (mHC) and :class:`DeepSeekV4SparseMoeBlock` (the 256-expert score-routed MoE) as
  the main decoder layer, with :class:`DSparkAttention` in place of the compressor
  attention. Stage 0 owns ``main_proj`` / ``main_norm``; the last stage owns the
  heads.
* :class:`DSparkModel` — ``forward_embed`` (build + embed the draft block from the
  main hidden and the accepted token), run the stages, then ``forward_head``
  (mHC-head collapse + norm + shared ``lm_head``, the Markov logit-bias sampling
  loop, and the confidence head).

Scope matches the agreed first deliverable: the draft module + PCC harness, greedy
sampling, no accept/verify loop and no traced fast path yet.
"""

from __future__ import annotations

from typing import Optional

import torch
import ttnn

from .common import DeepSeekV4Module, _HIFI4, _MASK_NEG
from .hyperconnection import DeepSeekV4HyperConnection, DeepSeekV4HyperHead
from .layers import DeepSeekV4RMSNorm, Linear, _rms_norm_unweighted, to_ttnn_device
from .moe import DeepSeekV4PreloadedExperts, DeepSeekV4SparseMoeBlock
from .quant import dequantize_weight
from .weight_cache import _as_cache, _load_weight, _materialize


def _rotate_half_matrix(rd: int) -> torch.Tensor:
    """``[Rd, Rd]`` matrix ``R`` with ``x @ R == interleaved rotate_half(x)``.

    Interleaved rotate_half maps ``(x_{2p}, x_{2p+1}) -> (-x_{2p+1}, x_{2p})``.
    """
    r = torch.zeros(rd, rd, dtype=torch.float32)
    for p in range(rd // 2):
        r[2 * p, 2 * p + 1] = 1.0
        r[2 * p + 1, 2 * p] = -1.0
    return r


def build_rope_tables(positions: torch.Tensor, rd: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved ``(cos, sin)`` rope tables ``[len(positions), Rd]`` for ``positions``.

    Matches the reference ``precompute_freqs_cis`` + ``apply_rotary_emb`` (base
    ``theta``, no YaRN — DSpark attention has ``compress_ratio == 0``). Each
    interleaved pair shares one angle, so the half-width table is
    ``repeat_interleave``'d to full ``Rd``.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    ang = torch.outer(positions.float(), inv_freq)  # [P, Rd/2]
    cos = torch.cos(ang).repeat_interleave(2, dim=-1)
    sin = torch.sin(ang).repeat_interleave(2, dim=-1)
    return cos, sin


class DSparkConfig:
    """Attribute shim so the reused MoE / hyper-connection modules see the config
    fields they expect (``num_local_experts`` etc.), populated from the DSpark args."""

    def __init__(self, a):
        self.hidden_size = a.dim
        self.num_attention_heads = a.n_heads
        self.head_dim = a.head_dim
        self.qk_rope_head_dim = a.rope_head_dim
        self.o_groups = a.o_groups
        self.o_lora_rank = a.o_lora_rank
        self.rms_norm_eps = a.norm_eps
        self.sliding_window = a.window_size
        self.num_local_experts = a.n_routed_experts
        self.num_experts_per_tok = a.n_activated_experts
        self.moe_intermediate_size = a.moe_inter_dim
        self.routed_scaling_factor = a.route_scale
        self.swiglu_limit = a.swiglu_limit
        self.hc_mult = a.hc_mult
        self.hc_sinkhorn_iters = a.hc_sinkhorn_iters
        self.hc_eps = a.hc_eps


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class DSparkAttention(DeepSeekV4Module):
    """Sliding-window MLA over the ``block_size`` draft query tokens.

    Reuses the V4 low-rank Q (``wq_a`` -> ``q_norm`` -> ``wq_b``), shared-KV MQA
    (``wkv`` -> ``kv_norm``), grouped O projection (``wo_a`` block-diagonal +
    ``wo_b``) and per-head sink. ``main_x`` (the single main-token hidden) seeds a
    sliding KV cache; the ``block_size`` draft tokens attend to the valid cache
    slots + their own block via an explicit gathered softmax with the sink folded
    into the denominator (the dense re-expression of the checkpoint's
    ``sparse_attn`` kernel).
    """

    def __init__(self, args, weights: dict, device, cache=None, weight_dtype: ttnn.DataType = ttnn.bfloat16):
        self.device = device
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rd = args.rope_head_dim
        self.o_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window = args.window_size
        self.eps = args.norm_eps
        self.rope_theta = args.rope_theta
        self.scaling = self.head_dim**-0.5
        cache = _as_cache(cache)

        self.wq_a = Linear(weights["attn.wq_a.weight"], device, cache.file("wq_a"), dtype=weight_dtype)
        self.q_norm = DeepSeekV4RMSNorm(weights["attn.q_norm.weight"], self.eps, device, cache.file("q_norm"))
        self.wq_b = Linear(weights["attn.wq_b.weight"], device, cache.file("wq_b"), dtype=weight_dtype)
        self.wkv = Linear(weights["attn.wkv.weight"], device, cache.file("wkv"), dtype=weight_dtype)
        self.kv_norm = DeepSeekV4RMSNorm(weights["attn.kv_norm.weight"], self.eps, device, cache.file("kv_norm"))
        self.wo_b = Linear(weights["attn.wo_b.weight"], device, cache.file("wo_b"), dtype=weight_dtype)

        oa = _materialize(weights["attn.wo_a.weight"], cache.file("wo_a"), weight_dtype)
        in_per_group = (self.n_heads * self.head_dim) // self.o_groups
        if oa is not None:
            oa = oa.reshape(self.o_groups, self.o_lora_rank, in_per_group).transpose(1, 2).contiguous()
        self.o_a_weight = _load_weight(oa, device, cache_file_name=cache.file("wo_a"), dtype=weight_dtype)

        sinks = weights["attn.attn_sink"]
        sinks = sinks() if callable(sinks) else sinks
        # per-head sink [1, H, 1, 1] for the [1, H, block, Skv] score layout.
        self.sink_tt = ttnn.from_torch(
            sinks.reshape(1, self.n_heads, 1, 1).float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.rot = _load_weight(_rotate_half_matrix(self.rd), device, cache_file_name=cache.file("rot"))

        self.kv_cache: Optional[ttnn.Tensor] = None  # [1, 1, window, Dh]

    def _apply_rope(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """Interleaved RoPE on the trailing ``rd`` dims of ``x`` (``[.., Dh]``).

        ``cos`` / ``sin`` broadcast to ``x``'s leading dims and cover the trailing
        ``rd``. ``nope`` head dims pass through. Uses an explicit ``x*cos +
        (x@rot)*sin`` (rather than the sharded fused rope op) so arbitrary
        ``[.., rows, Dh]`` block layouts work.
        """
        nope = self.head_dim - self.rd
        x_nope = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], nope])
        x_rope = ttnn.slice(x, [0, 0, 0, nope], [x.shape[0], x.shape[1], x.shape[2], self.head_dim])
        rotated = ttnn.matmul(x_rope, self.rot, compute_kernel_config=_HIFI4)
        out = ttnn.add(ttnn.multiply(x_rope, cos), ttnn.multiply(rotated, sin))
        return ttnn.concat([x_nope, out], dim=-1)

    def seed_cache(self, main_x: ttnn.Tensor, main_cos: ttnn.Tensor, main_sin: ttnn.Tensor, start_pos: int) -> None:
        """Write the main token's rotated K=V into the sliding ring at ``start_pos``."""
        main_kv = self.kv_norm(self.wkv(main_x))  # [1,1,1,Dh]
        main_kv = ttnn.reshape(main_kv, [1, 1, 1, self.head_dim])
        main_kv = self._apply_rope(main_kv, main_cos, main_sin)
        if self.kv_cache is None:
            self.kv_cache = ttnn.from_torch(
                torch.zeros(1, 1, self.window, self.head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        slot = start_pos % self.window
        head = ttnn.slice(self.kv_cache, [0, 0, 0, 0], [1, 1, slot, self.head_dim]) if slot > 0 else None
        tail = (
            ttnn.slice(self.kv_cache, [0, 0, slot + 1, 0], [1, 1, self.window, self.head_dim])
            if slot + 1 < self.window
            else None
        )
        parts = [p for p in (head, main_kv, tail) if p is not None]
        self.kv_cache = ttnn.concat(parts, dim=2) if len(parts) > 1 else main_kv

    def forward(
        self,
        x: ttnn.Tensor,  # [1, block, 1, D] collapsed draft hidden
        main_x: ttnn.Tensor,  # [1, 1, 1, D] main-token hidden
        q_cos: ttnn.Tensor,
        q_sin: ttnn.Tensor,
        q_neg_sin: ttnn.Tensor,
        main_cos: ttnn.Tensor,
        main_sin: ttnn.Tensor,
        start_pos: int,
        n_valid_window: int,
    ) -> ttnn.Tensor:
        block = x.shape[1]
        h, dh = self.n_heads, self.head_dim
        # -- Q projection + per-head RMSNorm + RoPE --------------------------- #
        q = self.wq_b(self.q_norm(self.wq_a(x)))  # [1, block, 1, H*Dh]
        q = ttnn.reshape(q, [1, block, h, dh])
        q = _rms_norm_unweighted(q, self.eps)
        q = self._apply_rope(q, q_cos, q_sin)  # [1, block, H, Dh]

        # -- draft KV + seed the sliding cache from the main token ------------ #
        kv = self.kv_norm(self.wkv(x))  # [1, block, 1, Dh]
        kv = ttnn.reshape(kv, [1, block, 1, dh])
        kv = self._apply_rope(kv, q_cos, q_sin)  # rope per draft position
        kv = ttnn.reshape(kv, [1, 1, block, dh])
        self.seed_cache(main_x, main_cos, main_sin, start_pos)
        kv_full = ttnn.concat([self.kv_cache, kv], dim=2)  # [1, 1, window + block, Dh]
        skv = self.window + block

        # -- gathered softmax attention with per-head sink ------------------- #
        q_h = ttnn.permute(q, [0, 2, 1, 3])  # [1, H, block, Dh]
        kvT = ttnn.permute(kv_full, [0, 1, 3, 2])  # [1, 1, Dh, Skv]
        scores = ttnn.matmul(q_h, kvT, compute_kernel_config=_HIFI4)  # [1, H, block, Skv]
        scores = ttnn.multiply(scores, self.scaling)

        # additive mask: keep window slots [0, n_valid_window) and the block slots
        # [window, window+block); drop the rest. Same for every head / query.
        keep = torch.zeros(1, 1, 1, skv, dtype=torch.float32)
        invalid = list(range(n_valid_window, self.window))
        for j in invalid:
            keep[0, 0, 0, j] = _MASK_NEG
        mask = ttnn.from_torch(keep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        scores = ttnn.add(scores, mask)

        m = ttnn.max(scores, dim=-1, keepdim=True)  # [1, H, block, 1]
        e = ttnn.exp(ttnn.subtract(scores, m))
        denom = ttnn.add(ttnn.sum(e, dim=-1, keepdim=True), ttnn.exp(ttnn.subtract(self.sink_tt, m)))
        w = ttnn.div(e, denom)  # [1, H, block, Skv]

        o = ttnn.matmul(w, kv_full, compute_kernel_config=_HIFI4)  # [1, H, block, Dh]
        o = ttnn.permute(o, [0, 2, 1, 3])  # [1, block, H, Dh]
        o = self._apply_rope(o, q_cos, q_neg_sin)  # inverse rope (neg sin)

        # -- grouped output projection --------------------------------------- #
        in_per_group = (h * dh) // self.o_groups
        y = ttnn.reshape(o, [block, self.o_groups, in_per_group])
        y = ttnn.permute(y, [1, 0, 2])  # [g, block, in_per_group]
        y = ttnn.matmul(y, self.o_a_weight, compute_kernel_config=_HIFI4)  # [g, block, o_lora_rank]
        y = ttnn.permute(y, [1, 0, 2])
        y = ttnn.reshape(y, [1, block, 1, self.o_groups * self.o_lora_rank])
        return self.wo_b(y)  # [1, block, 1, D]


# --------------------------------------------------------------------------- #
# Block
# --------------------------------------------------------------------------- #
class DSparkBlock(DeepSeekV4Module):
    """One DSpark MTP stage: mHC + :class:`DSparkAttention` + MoE."""

    def __init__(self, args, stage_id: int, weights: dict, device, experts, cache=None, weight_dtype=ttnn.bfloat16):
        self.args = args
        self.stage_id = stage_id
        self.device = device
        self.hc_mult = args.hc_mult
        cfg = DSparkConfig(args)
        cache = _as_cache(cache)

        self.attn = DSparkAttention(args, weights, device, cache=cache.sub("attn"), weight_dtype=weight_dtype)
        self.attn_norm = DeepSeekV4RMSNorm(weights["attn_norm.weight"], args.norm_eps, device, cache.file("attn_norm"))
        self.ffn_norm = DeepSeekV4RMSNorm(weights["ffn_norm.weight"], args.norm_eps, device, cache.file("ffn_norm"))
        self.attn_hc = DeepSeekV4HyperConnection(
            cfg,
            {"fn": weights["hc_attn_fn"], "base": weights["hc_attn_base"], "scale": weights["hc_attn_scale"]},
            device,
            cache=cache.sub("attn_hc"),
        )
        self.ffn_hc = DeepSeekV4HyperConnection(
            cfg,
            {"fn": weights["hc_ffn_fn"], "base": weights["hc_ffn_base"], "scale": weights["hc_ffn_scale"]},
            device,
            cache=cache.sub("ffn_hc"),
        )
        moe_weights = {
            "gate.weight": weights["ffn.gate.weight"],
            "gate.e_score_correction_bias": weights["ffn.gate.bias"],
            "shared_experts.gate_proj.weight": weights["ffn.shared_experts.w1.weight"],
            "shared_experts.down_proj.weight": weights["ffn.shared_experts.w2.weight"],
            "shared_experts.up_proj.weight": weights["ffn.shared_experts.w3.weight"],
        }
        self.mlp = DeepSeekV4SparseMoeBlock(
            cfg, moe_weights, device, experts=experts, cache=cache.sub("mlp"), weight_dtype=weight_dtype
        )

        if stage_id == 0:
            self.main_proj = Linear(weights["main_proj.weight"], device, cache.file("main_proj"), dtype=weight_dtype)
            self.main_norm = DeepSeekV4RMSNorm(
                weights["main_norm.weight"], args.norm_eps, device, cache.file("main_norm")
            )

    def _hc_block(self, hc_module, streams):
        """Run the (decode-only, T==1) fused hyper-connection over each of the
        ``block`` draft tokens independently and stack the results.

        ``streams`` ``[1, block, hc, D]`` -> ``(post [1,block,hc,1],
        comb [1,block,hc,hc], collapsed [1,block,1,D])``. The mHC is per-token, so
        looping is exact; only the fused op's T==1 restriction forces the split.
        """
        block, hc, d = streams.shape[1], streams.shape[2], streams.shape[3]
        posts, combs, cols = [], [], []
        for i in range(block):
            tok = ttnn.slice(streams, [0, i, 0, 0], [1, i + 1, hc, d])  # [1,1,hc,D]
            post, comb, collapsed = hc_module(tok)
            posts.append(post)
            combs.append(comb)
            cols.append(collapsed)
        post = ttnn.concat(posts, dim=1) if block > 1 else posts[0]
        comb = ttnn.concat(combs, dim=1) if block > 1 else combs[0]
        collapsed = ttnn.concat(cols, dim=1) if block > 1 else cols[0]
        return post, comb, collapsed

    def _mix(self, post, comb, sublayer_out, streams):
        b, s, hc, d = streams.shape
        t = b * s
        out = ttnn.repeat(ttnn.reshape(sublayer_out, [1, t, 1, d]), ttnn.Shape([1, 1, hc, 1]))
        placement = ttnn.multiply(out, ttnn.reshape(post, [1, t, hc, 1]))
        comb_t = ttnn.transpose(ttnn.reshape(comb, [1, t, hc, hc]), -2, -1)
        mixed = ttnn.matmul(comb_t, ttnn.reshape(streams, [1, t, hc, d]), compute_kernel_config=_HIFI4)
        return ttnn.reshape(ttnn.add(placement, mixed), [b, s, hc, d])

    def forward(self, streams, rope, start_pos, n_valid_window, main_x=None):
        """``streams`` ``[1, block, hc, D]``; ``main_x`` ``[1,1,1,D]`` (stage-0 only,
        else the already-projected/normed main hidden passed through)."""
        block = streams.shape[1]
        post, comb, collapsed = self._hc_block(self.attn_hc, streams)
        normed = self.attn_norm(collapsed)  # [1, block, 1, D]
        attn_out = self.attn(
            normed,
            main_x,
            rope["q_cos"],
            rope["q_sin"],
            rope["q_neg_sin"],
            rope["m_cos"],
            rope["m_sin"],
            start_pos,
            n_valid_window,
        )
        streams = self._mix(post, comb, attn_out, streams)
        post, comb, collapsed = self._hc_block(self.ffn_hc, streams)
        normed = self.ffn_norm(collapsed)  # [1, block, 1, D]
        # The preloaded-experts MoE is natively single-token; run each draft token
        # through it and concat back (prefill-by-decode, as the main model does).
        d = normed.shape[-1]
        per_tok = [
            self.mlp(ttnn.reshape(ttnn.slice(normed, [0, i, 0, 0], [1, i + 1, 1, d]), [1, 1, 1, d]))
            for i in range(block)
        ]
        mlp_out = ttnn.concat(per_tok, dim=1) if block > 1 else per_tok[0]  # [1, block, 1, D]
        mlp_out = ttnn.reshape(mlp_out, [1, block, 1, d])
        return self._mix(post, comb, mlp_out, streams)


# --------------------------------------------------------------------------- #
# Full DSpark speculative stack
# --------------------------------------------------------------------------- #
class DSparkModel(DeepSeekV4Module):
    def __init__(self, args, loader, device, cache=None, weight_dtype: ttnn.DataType = ttnn.bfloat4_b):
        self.args = args
        self.device = device
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        cache = _as_cache(cache)

        def w(name):
            return lambda: dequantize_weight(
                loader.get_tensor(name, translate=False), loader.get_scale(name, translate=False)
            )

        # shared main embedding + lm_head
        embed = loader.get_tensor("embed.weight", translate=False).float()
        self.embed_weight = to_ttnn_device(
            embed, device, layout=ttnn.ROW_MAJOR_LAYOUT, cache_file_name=cache.file("embed")
        )
        self.lm_head = Linear(w("head.weight"), device, cache.file("head"), dtype=weight_dtype)

        self.blocks: list[DSparkBlock] = []
        for s in range(args.n_mtp_layers):
            p = f"mtp.{s}"
            weights = {
                "attn.wq_a.weight": w(f"{p}.attn.wq_a.weight"),
                "attn.q_norm.weight": w(f"{p}.attn.q_norm.weight"),
                "attn.wq_b.weight": w(f"{p}.attn.wq_b.weight"),
                "attn.wkv.weight": w(f"{p}.attn.wkv.weight"),
                "attn.kv_norm.weight": w(f"{p}.attn.kv_norm.weight"),
                "attn.wo_a.weight": w(f"{p}.attn.wo_a.weight"),
                "attn.wo_b.weight": w(f"{p}.attn.wo_b.weight"),
                "attn.attn_sink": w(f"{p}.attn.attn_sink"),
                "attn_norm.weight": w(f"{p}.attn_norm.weight"),
                "ffn_norm.weight": w(f"{p}.ffn_norm.weight"),
                "hc_attn_fn": w(f"{p}.hc_attn_fn"),
                "hc_attn_base": w(f"{p}.hc_attn_base"),
                "hc_attn_scale": w(f"{p}.hc_attn_scale"),
                "hc_ffn_fn": w(f"{p}.hc_ffn_fn"),
                "hc_ffn_base": w(f"{p}.hc_ffn_base"),
                "hc_ffn_scale": w(f"{p}.hc_ffn_scale"),
                "ffn.gate.weight": w(f"{p}.ffn.gate.weight"),
                "ffn.gate.bias": w(f"{p}.ffn.gate.bias"),
                "ffn.shared_experts.w1.weight": w(f"{p}.ffn.shared_experts.w1.weight"),
                "ffn.shared_experts.w2.weight": w(f"{p}.ffn.shared_experts.w2.weight"),
                "ffn.shared_experts.w3.weight": w(f"{p}.ffn.shared_experts.w3.weight"),
            }
            if s == 0:
                weights["main_proj.weight"] = w(f"{p}.main_proj.weight")
                weights["main_norm.weight"] = w(f"{p}.main_norm.weight")
            experts = DeepSeekV4PreloadedExperts(
                DSparkConfig(args),
                self._expert_provider(loader, p),
                device,
                dtype=weight_dtype,
                cache=cache.sub(f"{p}.mlp"),
            )
            self.blocks.append(
                DSparkBlock(args, s, weights, device, experts, cache=cache.sub(p), weight_dtype=weight_dtype)
            )

        # heads on the last stage
        last = f"mtp.{args.n_mtp_layers - 1}"
        self.norm = DeepSeekV4RMSNorm(w(f"{last}.norm.weight"), args.norm_eps, device, cache.file("norm"))
        self.hc_head = DeepSeekV4HyperHead(
            DSparkConfig(args),
            {
                "hc_fn": w(f"{last}.hc_head_fn"),
                "hc_base": w(f"{last}.hc_head_base"),
                "hc_scale": w(f"{last}.hc_head_scale"),
            },
            device,
            cache=cache.sub("hc_head"),
        )
        mw1 = loader.get_tensor(f"{last}.markov_head.markov_w1.weight", translate=False).float()
        self.markov_w1 = to_ttnn_device(
            mw1, device, layout=ttnn.ROW_MAJOR_LAYOUT, cache_file_name=cache.file("markov_w1")
        )
        self.markov_w2 = Linear(
            w(f"{last}.markov_head.markov_w2.weight"), device, cache.file("markov_w2"), dtype=weight_dtype
        )
        self.confidence_proj = Linear(
            w(f"{last}.confidence_head.proj.weight"), device, cache.file("confidence"), dtype=weight_dtype
        )

    def _expert_provider(self, loader, prefix):
        def provider(e):
            base = f"{prefix}.ffn.experts.{e}"
            g = dequantize_weight(
                loader.get_tensor(f"{base}.w1.weight", translate=False),
                loader.get_scale(f"{base}.w1.weight", translate=False),
            )
            u = dequantize_weight(
                loader.get_tensor(f"{base}.w3.weight", translate=False),
                loader.get_scale(f"{base}.w3.weight", translate=False),
            )
            d = dequantize_weight(
                loader.get_tensor(f"{base}.w2.weight", translate=False),
                loader.get_scale(f"{base}.w2.weight", translate=False),
            )
            return torch.cat([g, u], dim=0).float(), d.float()

        return provider

    def _main_x(self, main_hidden: ttnn.Tensor, rows: int) -> ttnn.Tensor:
        """Stage-0 ``main_norm(main_proj(main_hidden))`` -> ``[1, 1, rows, D]``."""
        stage0 = self.blocks[0]
        mh = ttnn.reshape(main_hidden, [1, 1, rows, main_hidden.shape[-1]])
        return stage0.main_norm(stage0.main_proj(mh))

    def prefill(self, main_hidden_seq: ttnn.Tensor) -> None:
        """Seed every stage's sliding KV cache from a length-``L`` main-hidden
        sequence (positions ``0..L-1``), mirroring the reference prefill. ``L`` must
        be ``<= window`` (the empty-cache fast path)."""
        L = main_hidden_seq.shape[-2]
        d, rd, theta = self.args.dim, self.args.rope_head_dim, self.args.rope_theta
        main_x = self._main_x(main_hidden_seq, L)  # [1,1,L,D]
        for p in range(L):
            mx = ttnn.slice(main_x, [0, 0, p, 0], [1, 1, p + 1, d])  # [1,1,1,D]
            cos, sin = build_rope_tables(torch.tensor([p]), rd, theta)
            cos_tt = ttnn.from_torch(
                cos.reshape(1, 1, 1, rd), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            sin_tt = ttnn.from_torch(
                sin.reshape(1, 1, 1, rd), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            for blk in self.blocks:
                blk.attn.seed_cache(mx, cos_tt, sin_tt, p)

    def _rope_bundle(self, start_pos: int) -> dict:
        rd, theta = self.args.rope_head_dim, self.args.rope_theta
        q_pos = start_pos + 1 + torch.arange(self.block_size)  # draft query positions
        m_pos = torch.tensor([start_pos])
        q_cos, q_sin = build_rope_tables(q_pos, rd, theta)
        m_cos, m_sin = build_rope_tables(m_pos, rd, theta)

        def tt(t, rows):
            return ttnn.from_torch(
                t.reshape(1, rows, 1, rd), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )

        # q tables broadcast over the head axis: shape [1, block, 1, rd].
        return {
            "q_cos": tt(q_cos, self.block_size),
            "q_sin": tt(q_sin, self.block_size),
            "q_neg_sin": tt(-q_sin, self.block_size),
            "m_cos": ttnn.from_torch(
                m_cos.reshape(1, 1, 1, rd), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            ),
            "m_sin": ttnn.from_torch(
                m_sin.reshape(1, 1, 1, rd), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            ),
        }

    def _embed(self, ids: torch.Tensor) -> ttnn.Tensor:
        ids_tt = ttnn.from_torch(
            ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        return ttnn.embedding(ids_tt, self.embed_weight, layout=ttnn.TILE_LAYOUT)

    def forward_spec(self, token_id: int, main_hidden: ttnn.Tensor, start_pos: int):
        """One draft step. ``main_hidden`` ``[1, 1, len(target)*D]``; returns
        ``(output_ids [block+1], logits [block, vocab], confidence [block])``."""
        hc = self.args.hc_mult
        # stage-0 main projection + draft block embed
        stage0 = self.blocks[0]
        main_x = stage0.main_norm(stage0.main_proj(ttnn.reshape(main_hidden, [1, 1, 1, main_hidden.shape[-1]])))
        draft_ids = torch.full((1, self.block_size), self.noise_token_id, dtype=torch.long)
        draft_ids[0, 0] = token_id
        x = self._embed(draft_ids)  # [1, block, D]
        streams = ttnn.repeat(ttnn.reshape(x, [1, self.block_size, 1, self.args.dim]), ttnn.Shape([1, 1, hc, 1]))

        rope = self._rope_bundle(start_pos)
        n_valid = min(self.args.window_size, start_pos + 1)
        for blk in self.blocks:
            streams = blk(streams, rope, start_pos, n_valid, main_x=main_x)

        # heads (hc-head collapse is per-token; the fused width-sharded head config
        # is built for a single row, so run each draft token through it and stack).
        d = self.args.dim
        x = ttnn.concat(
            [self.hc_head(ttnn.slice(streams, [0, i, 0, 0], [1, i + 1, hc, d])) for i in range(self.block_size)],
            dim=1,
        )  # [1, block, 1, D]
        x_normed = self.norm(x)
        logits = self.lm_head(x_normed)  # [1, block, 1, vocab]
        logits = ttnn.to_torch(logits).reshape(self.block_size, -1).float()

        # markov logit-bias autoregressive greedy loop (host, block_size steps)
        markov_w1 = ttnn.to_torch(self.markov_w1).float()
        x_host = ttnn.to_torch(x).reshape(self.block_size, -1).float()
        output_ids = torch.empty(self.block_size + 1, dtype=torch.long)
        output_ids[0] = token_id
        markov_embeds = []
        for i in range(self.block_size):
            m_embed = markov_w1[output_ids[i]]  # [markov_rank]
            m_embed_tt = ttnn.from_torch(
                m_embed.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            bias = ttnn.to_torch(self.markov_w2(m_embed_tt)).reshape(-1).float()
            logits[i] = logits[i] + bias
            markov_embeds.append(m_embed)
            output_ids[i + 1] = int(logits[i].argmax().item())
        markov_embed = torch.stack(markov_embeds, dim=0)  # [block, markov_rank]

        conf_in = torch.cat([x_host, markov_embed], dim=-1)  # [block, D + markov_rank]
        conf_in_tt = ttnn.from_torch(
            conf_in.reshape(1, 1, self.block_size, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        confidence = ttnn.to_torch(self.confidence_proj(conf_in_tt)).reshape(-1).float()
        return output_ids, logits, confidence


__all__ = ["DSparkModel", "DSparkBlock", "DSparkAttention", "DSparkConfig", "build_rope_tables"]
