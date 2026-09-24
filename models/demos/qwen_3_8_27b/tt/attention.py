# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gated full attention (every 4th layer) on the SP x TP mesh.

Structure from minimax_m3/tt/attention (fused per-column wqkv, column-parallel projection,
nlp_create_qkv_heads, the KV write through the canonical cache, ring-joint SDPA over SP, nlp_concat_heads,
row-parallel o_proj + TP all-reduce). Qwen-specific math (qwen36 attention/tp.py, math only):

  q_proj emits [q | gate] per head (2 x head_dim); gate -> sigmoid, multiplied into the SDPA output
  q_norm / k_norm: zero-centred RMSNorm over head_dim (the +1 folded at load)
  partial RoPE on the first 64 of 256 channels (tt/rope.py)

Per TP column c: q heads [6c, 6c+6), kv head c (TP=4 == num_kv_heads, one per chip, no replication).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.qwen_3_8_27b.config import Qwen38Config
from models.demos.qwen_3_8_27b.tt.common import hifi4_fp32, residual_dtype, upload
from models.demos.qwen_3_8_27b.tt.kv_cache import write_kv_chunk
from models.demos.qwen_3_8_27b.tt.rms_norm import gain_tensor, rms_norm_fp32
from models.demos.qwen_3_8_27b.tt.sdpa import (
    cache_attn_mode,
    masked_sdpa_cache,
    plain_sdpa_configs,
    ring_sdpa_cache,
    ring_sdpa_nocache,
    sdpa_configs,
)


class TtAttention:
    def __init__(
        self, mesh_config, ccl, cfg: Qwen38Config, sd, *, attn_ordinal: int, weight_dtype, cache=None, prefix=""
    ):
        """sd: HF tensors relative to ``self_attn.`` or None on a cache hit."""
        self.mc = mesh_config
        self.ccl = ccl
        self.mesh = mesh_config.mesh_device
        self.cfg = cfg
        self.attn_ordinal = attn_ordinal
        tp = mesh_config.tp
        self.hd = cfg.head_dim
        self.nq, self.nkv = cfg.num_attention_heads // tp, cfg.num_key_value_heads // tp
        assert self.nkv == 1, "one kv head per TP column"
        self.qkv_w = (self.nq + 2 * self.nkv) * self.hd  # 2048
        self.g_w = self.nq * self.hd  # 1536
        self.scale = self.hd**-0.5
        self.ckc = hifi4_fp32()
        self.sdpa_cfg_live = sdpa_configs(self.mesh, fp32_acc=True)  # live K/V ring
        self.sdpa_cfg_cache = sdpa_configs(self.mesh, fp32_acc=False)  # cache-read ring (op requires False)
        self.sdpa_cfg_masked = plain_sdpa_configs(self.mesh)  # composed cache read, fp32 accumulation

        host = dict.fromkeys(["wqkvg", "wo", "q_norm", "k_norm"])
        if sd is not None:
            hd = self.hd
            wq = sd["q_proj.weight"].float().reshape(cfg.num_attention_heads, 2, hd, -1)  # [heads, q|gate, hd, H]
            wk = sd["k_proj.weight"].float().reshape(cfg.num_key_value_heads, hd, -1)
            wv = sd["v_proj.weight"].float().reshape(cfg.num_key_value_heads, hd, -1)
            wo = sd["o_proj.weight"].float()  # [H, nh*hd]
            blocks, oblocks = [], []
            for c in range(tp):
                hs = slice(c * self.nq, (c + 1) * self.nq)
                q = wq[hs, 0].reshape(-1, wq.shape[-1])
                g = wq[hs, 1].reshape(-1, wq.shape[-1])
                blocks.append(torch.cat([q, wk[c], wv[c], g], 0).T)  # [H, qkv_w + g_w]
                oblocks.append(wo[:, c * self.g_w : (c + 1) * self.g_w].T)  # [g_w, H]
            host["wqkvg"] = torch.cat(blocks, 1)[None, None]
            host["wo"] = torch.cat(oblocks, 0)[None, None]
        up = lambda n, **kw: upload(host[n], self.mesh, cache=cache, name=f"{prefix}{n}", **kw)  # noqa: E731
        self.wqkvg = up("wqkvg", dtype=weight_dtype, mapper=mesh_config.shard(None, 3))
        self.wo = up("wo", dtype=weight_dtype, mapper=mesh_config.shard(None, 2))
        # q/k-norm gains fp32 (zero-centred: the folded 1+w loses most of w in bf16; see tt/rms_norm.py)
        g = lambda k: None if sd is None else sd[k]  # noqa: E731
        self.q_norm = gain_tensor(
            mesh_config, g("q_norm.weight"), unit_offset=True, cache=cache, name=f"{prefix}q_norm"
        )
        self.k_norm = gain_tensor(
            mesh_config, g("k_norm.weight"), unit_offset=True, cache=cache, name=f"{prefix}k_norm"
        )

    def __call__(self, x, ctx, rope):
        """x [1,1,S,H] TP-replicated, SP-sharded. ctx: PrefillCtx (caches, user_id, start, valid_end, cos, sin)."""
        S = x.shape[2]
        mcfg = dict(compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xqkvg = ttnn.linear(x, self.wqkvg, **mcfg)
        qkv = ttnn.slice(xqkvg, [0, 0, 0, 0], [1, 1, S, self.qkv_w])
        gate = ttnn.slice(xqkvg, [0, 0, 0, self.qkv_w], [1, 1, S, self.qkv_w + self.g_w])
        ttnn.deallocate(xqkvg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.nq,
            num_kv_heads=self.nkv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        eps = self.cfg.rms_norm_eps
        qn = rms_norm_fp32(q, self.q_norm, eps)
        kn = rms_norm_fp32(k, self.k_norm, eps)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        q = rope.apply(qn, ctx.cos, ctx.sin)
        k = rope.apply(kn, ctx.cos, ctx.sin)
        ttnn.deallocate(qn)
        ttnn.deallocate(kn)

        caches = ctx.caches
        write_kv_chunk(caches, k, v, slot_idx=ctx.user_id, layer_idx=self.attn_ordinal, kv_actual=ctx.start)
        common = dict(mesh_device=self.mesh, ccl=self.ccl, n_kv=self.cfg.num_key_value_heads, scale=self.scale)
        if ctx.start == 0:
            attn = ring_sdpa_nocache(
                q, k, v, head_dim=self.hd, logical_n=ctx.valid_end, configs=self.sdpa_cfg_live, **common
            )
        elif cache_attn_mode() == "masked":
            prog, kcfg = self.sdpa_cfg_masked
            attn = masked_sdpa_cache(
                q,
                caches,
                mesh_config=self.mc,
                rows_local=(ctx.start + ctx.tokens) // self.mc.sp,
                slot=caches.slot(ctx.user_id, self.attn_ordinal),
                mask=ctx.cache_mask,
                scale=self.scale,
                program_config=prog,
                compute_kernel_config=kcfg,
            )
        else:
            attn = ring_sdpa_cache(
                q,
                caches,
                kv_actual=ctx.start,
                logical_n=ctx.valid_end,
                slot_idx=ctx.user_id,
                layer_idx=self.attn_ordinal,
                configs=self.sdpa_cfg_cache,
                **common,
            )
        for t in (q, k, v):
            ttnn.deallocate(t)
        o = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,S,g_w]
        ttnn.deallocate(attn)
        o2 = ttnn.multiply(o, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(o)
        ttnn.deallocate(gate)
        out = ttnn.linear(o2, self.wo, dtype=residual_dtype(), **mcfg)
        ttnn.deallocate(o2)
        return self.mc.all_reduce_tp(out)
