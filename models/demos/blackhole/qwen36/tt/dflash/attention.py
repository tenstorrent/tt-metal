# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M6: DFlash dual-source attention.

The one genuinely unusual module in the drafter. Queries come from the **block only** (16
rows), while keys and values come from ``concat(context, block)``. So this is neither self-
nor cross-attention but both at once, and three things follow that no stock attention module
does:

1. **Q and K/V have different sequence lengths**, making the score matrix rectangular
   (16 x ctx+16) rather than square.
2. **Q and K take different slices of the RoPE table** -- q the last 16 rows, k all of them
   (see ``rope.py``).
3. **The mask differs per layer kind**: sliding layers are causal *and* windowed, while the
   single ``full_attention`` layer gets no mask at all and attends bidirectionally over
   everything. That layer is where the "block diffusion" behaviour lives.

``k_norm`` is applied to the **concatenated** K, i.e. over all ``ctx + 16`` positions, and
``v`` receives neither a norm nor RoPE -- both straight from the reference.

Context K/V are computed from ``target_hidden`` alone and are therefore independent of the
block, which is why the real serving loop caches them and only projects the newly-committed
tokens. ``ctx_k``/``ctx_v`` are accepted pre-computed here so that path is a drop-in later;
Milestone 1 simply passes the whole context every call.

TP: q/k/v column-parallel (at TP=8 that is 4 query heads and exactly 1 KV head per chip, so
GQA needs no KV replication), ``o_proj`` row-parallel followed by one all-reduce.
"""

from __future__ import annotations

import ttnn
from models.demos.blackhole.qwen36.tt.dflash.ccl import all_reduce_replicated
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig
from models.demos.blackhole.qwen36.tt.dflash.rms_norm import rms_norm

_MC = ttnn.DRAM_MEMORY_CONFIG


def _split_heads(x, n_heads_local: int, head_dim: int):
    """``[1, 1, S, n*d]`` -> ``[1, n, S, d]``, matching the reference's view+transpose."""
    seq = x.shape[-2]
    x = ttnn.reshape(x, (1, seq, n_heads_local, head_dim))
    return ttnn.permute(x, (0, 2, 1, 3))


def _merge_heads(x, head_dim: int):
    """``[1, n, S, d]`` -> ``[1, 1, S, n*d]``, the inverse of :func:`_split_heads`."""
    n, seq = x.shape[-3], x.shape[-2]
    x = ttnn.permute(x, (0, 2, 1, 3))
    return ttnn.reshape(x, (1, 1, seq, n * head_dim))


class DFlashAttention:
    """One layer's dual-source attention."""

    def __init__(
        self,
        mesh_device,
        cfg: DFlashDrafterConfig,
        weights,
        layer_idx: int,
        tt_ccl,
        topology=None,
        compute_kernel_config=None,
    ):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.w = weights
        self.tt_ccl = tt_ccl
        self.topology = topology
        tp = mesh_device.get_num_devices()
        self.n_heads_local = cfg.num_attention_heads // tp
        self.n_kv_local = cfg.num_key_value_heads // tp
        self.scale = cfg.head_dim**-0.5
        self.compute_kernel_config = compute_kernel_config or ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def project_context(self, target_hidden):
        """Context K/V from ``target_hidden`` -- block-independent, hence cacheable.

        Returns ``(ctx_k, ctx_v)`` as ``[1, 1, ctx, kv_local*head_dim]``, pre-head-split, so
        a cache can hold them in the same layout the block path concatenates against.
        """
        ckc = self.compute_kernel_config
        k = ttnn.linear(target_hidden, self.w.k_proj, compute_kernel_config=ckc, memory_config=_MC)
        v = ttnn.linear(target_hidden, self.w.v_proj, compute_kernel_config=ckc, memory_config=_MC)
        return k, v

    def forward(self, x, ctx_k, ctx_v, rope, cos_q, sin_q, cos_k, sin_k, attn_mask):
        """``x`` is the normed block ``[1, 1, block, 5120]`` replicated.

        ``ctx_k``/``ctx_v`` are this layer's context projections from
        :meth:`project_context`. ``attn_mask`` is ``None`` for the full-attention layer.
        """
        cfg = self.cfg
        ckc = self.compute_kernel_config

        # ---- Q: block only ------------------------------------------------------------
        q = ttnn.linear(x, self.w.q_proj, compute_kernel_config=ckc, memory_config=_MC)
        q = _split_heads(q, self.n_heads_local, cfg.head_dim)
        q = rms_norm(q, self.w.q_norm, cfg.rms_norm_eps, memory_config=_MC)
        q = rope.apply(q, cos_q, sin_q)  # q takes the TAIL of the table

        # ---- K/V: concat(context, block) ----------------------------------------------
        blk_k = ttnn.linear(x, self.w.k_proj, compute_kernel_config=ckc, memory_config=_MC)
        blk_v = ttnn.linear(x, self.w.v_proj, compute_kernel_config=ckc, memory_config=_MC)
        k = ttnn.concat([ctx_k, blk_k], dim=-2, memory_config=_MC)
        v = ttnn.concat([ctx_v, blk_v], dim=-2, memory_config=_MC)
        ttnn.deallocate(blk_k)
        ttnn.deallocate(blk_v)

        k = _split_heads(k, self.n_kv_local, cfg.head_dim)
        v = _split_heads(v, self.n_kv_local, cfg.head_dim)
        # k_norm covers the WHOLE concatenation, not just the block. v gets no norm.
        k = rms_norm(k, self.w.k_norm, cfg.rms_norm_eps, memory_config=_MC)
        k = rope.apply(k, cos_k, sin_k)  # k takes the FULL table

        # ---- attention ----------------------------------------------------------------
        # The op handles GQA natively (nkv may differ from nqh) and accepts a rectangular
        # mask, so no repeat_interleave and no hand-rolled softmax is needed.
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,  # causality, where wanted, is in the mask
            scale=self.scale,
            compute_kernel_config=ckc,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        merged = _merge_heads(attn, cfg.head_dim)
        ttnn.deallocate(attn)
        partial = ttnn.linear(merged, self.w.o_proj, compute_kernel_config=ckc, memory_config=_MC)
        ttnn.deallocate(merged)
        out = all_reduce_replicated(partial, self.mesh_device, self.tt_ccl, self.topology, dim=3)
        ttnn.deallocate(partial)
        return out
