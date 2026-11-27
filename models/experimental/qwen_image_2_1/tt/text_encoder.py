# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL-8B text stack (the T2I text encoder of Qwen-Image-2.1) in TTNN, prefill only, batch 1.

Returns the LAST decoder layer's output *before* the final RMSNorm (what the DiT was trained on).
GQA 32/8 heads x 128, SwiGLU 12288, RMSNorm eps 1e-6, RoPE theta 5e6. The q/k projection rows are
permuted per head so the adjacent-pair RoPE kernel reproduces llama-style rotate_half (see common/rope.py).
Token embeddings are gathered on host (1.2 GB table stays off-device).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch

import ttnn

from ..common import rope as rope_mod
from ..common.config import TE, TextEncoderConfig
from ..common.weights import (
    LazyCheckpoint,
    interleave_pairs_permutation,
    linear_to_mm,
    permute_heads_rows,
    swiglu_interleave,
)

TILE = 32
PFX = "model.language_model."


@dataclass
class TEPrecision:
    weight_dtype: ttnn.DataType = ttnn.bfloat8_b  # weights: bfp8 keeps the 17.5 GB model at ~9 GB
    mm_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2


class TELayer:
    def __init__(
        self, dev, ckpt: LazyCheckpoint, idx: int, prec: TEPrecision, perm: torch.Tensor, cfg: TextEncoderConfig = TE
    ):
        p = f"{PFX}layers.{idx}."
        g = lambda k: ckpt.get(p + k, torch.bfloat16)
        wd = prec.weight_dtype
        mem = ttnn.DRAM_MEMORY_CONFIG
        wq = permute_heads_rows(g("self_attn.q_proj.weight"), cfg.heads, cfg.head_dim, perm)
        wk = permute_heads_rows(g("self_attn.k_proj.weight"), cfg.kv_heads, cfg.head_dim, perm)
        wv = g("self_attn.v_proj.weight")
        self.wqkv = ttnn.from_torch(
            torch.cat([linear_to_mm(wq), linear_to_mm(wk), linear_to_mm(wv)], dim=1),
            dtype=wd,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        f_norm = lambda t: ttnn.from_torch(
            t.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem
        )
        self.q_norm = f_norm(g("self_attn.q_norm.weight")[perm])
        self.k_norm = f_norm(g("self_attn.k_norm.weight")[perm])
        self.wo = ttnn.from_torch(
            linear_to_mm(g("self_attn.o_proj.weight")), dtype=wd, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem
        )
        self.w_gateup = ttnn.from_torch(
            swiglu_interleave(g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")),
            dtype=wd,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        self.w_down = ttnn.from_torch(
            linear_to_mm(g("mlp.down_proj.weight")), dtype=wd, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem
        )
        self.ln1 = f_norm(g("input_layernorm.weight"))
        self.ln2 = f_norm(g("post_attention_layernorm.weight"))


class Qwen3VLTextEncoder:
    def __init__(
        self,
        dev,
        ckpt: LazyCheckpoint,
        prec: Optional[TEPrecision] = None,
        cfg: TextEncoderConfig = TE,
        layers: Optional[int] = None,
    ):
        self.dev = dev
        self.cfg = cfg
        self.ckpt = ckpt
        self.prec = prec or TEPrecision()
        self.perm = interleave_pairs_permutation(cfg.head_dim)
        n = cfg.num_layers if layers is None else layers  # `layers or ...` made 0 mean "all 36"
        self.layers: List[TELayer] = [TELayer(dev, ckpt, i, self.prec, self.perm, cfg) for i in range(n)]
        self.grid = dev.compute_with_storage_grid_size()
        arch = dev.arch()
        self.ck_mm = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=self.prec.mm_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.ck_norm = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.trans_mat = ttnn.from_torch(
            rope_mod.rot_transformation_mat(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Flash-attention tiling and SDPA precision. The text-only prompt is one 32-row chunk, so
        # these defaults never mattered there; a 1k-token image prompt is 34 chunks and the rescaling
        # accumulation shows up (see text_encoder_vl.py, which raises both).
        self.sdpa_chunk = TILE
        self.ck_sdpa = self.ck_mm
        self._embed_key = PFX + "embed_tokens.weight"

    # ------------------------------------------------------------------ host side
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[L] or [1, L] token ids -> [L, 4096] bf16 (host gather of the embedding rows)."""
        ids = input_ids.reshape(-1)
        return self.ckpt.get_rows(self._embed_key, ids).to(torch.bfloat16)

    # ------------------------------------------------------------------ device side
    def _rope_tables(self, S: int):
        cos, sin = rope_mod.te_cos_sin(S)
        f = lambda t: ttnn.from_torch(
            t.reshape(1, 1, S, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return f(cos), f(sin)

    def _linear(self, x, w, dtype=ttnn.bfloat16):
        return ttnn.linear(x, w, compute_kernel_config=self.ck_mm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dtype)

    def _rms(self, x, w):
        return ttnn.rms_norm(
            x,
            epsilon=self.cfg.rms_eps,
            weight=w,
            compute_kernel_config=self.ck_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_device(
        self,
        emb: torch.Tensor,
        taps: Optional[List[int]] = None,
        cos_sin=None,
        post_add: Optional[dict] = None,
    ):
        """emb [L, 4096] bf16 host -> (hidden [1,1,Sp,4096] device bf16 after the last layer, L, taps dict).

        `cos_sin` overrides the 1-D RoPE tables, which is how the image-conditioned path (see
        `text_encoder_vl.py`) supplies its 3-D mRoPE tables; `post_add` maps a layer index to a
        [1, 1, Sp, 4096] tensor added to the residual stream after that layer, which is how the same
        path injects the vision tower's deepstack features. Taps are taken BEFORE the add, matching
        what transformers' `output_hidden_states` hooks record.
        """
        c = self.cfg
        L = emb.shape[0]
        Sp = (L + TILE - 1) // TILE * TILE
        x_host = torch.zeros(1, 1, Sp, c.hidden, dtype=torch.bfloat16)
        x_host[0, 0, :L] = emb
        h = ttnn.from_torch(
            x_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        cos, sin = self._rope_tables(Sp) if cos_sin is None else cos_sin
        chunk = min(self.sdpa_chunk, Sp)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid, q_chunk_size=chunk, k_chunk_size=chunk, exp_approx_mode=True
        )
        out_taps = {}
        for li, layer in enumerate(self.layers):
            x = self._rms(h, layer.ln1)
            qkv = self._linear(x, layer.wqkv)
            ttnn.deallocate(x)
            if len(qkv.shape) != 4:
                qkv = ttnn.reshape(qkv, [1, 1, Sp, -1])
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=c.heads,
                num_kv_heads=c.kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv)
            qn = self._rms(q, layer.q_norm)
            ttnn.deallocate(q)
            kn = self._rms(k, layer.k_norm)
            ttnn.deallocate(k)
            qr = ttnn.experimental.rotary_embedding_llama(
                qn, cos, sin, self.trans_mat, is_decode_mode=False, compute_kernel_config=self.ck_norm
            )
            ttnn.deallocate(qn)
            kr = ttnn.experimental.rotary_embedding_llama(
                kn, cos, sin, self.trans_mat, is_decode_mode=False, compute_kernel_config=self.ck_norm
            )
            ttnn.deallocate(kn)
            attn = ttnn.transformer.scaled_dot_product_attention(
                qr,
                kr,
                v,
                is_causal=True,
                scale=1.0 / math.sqrt(c.head_dim),
                program_config=sdpa_cfg,
                compute_kernel_config=self.ck_sdpa,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qr)
            ttnn.deallocate(kr)
            ttnn.deallocate(v)
            a = ttnn.transformer.concatenate_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn)
            o = self._linear(a, layer.wo)
            ttnn.deallocate(a)
            h2 = ttnn.add(h, o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(o)
            ttnn.deallocate(h)
            x2 = self._rms(h2, layer.ln2)
            if self.prec.weight_dtype != ttnn.bfloat16:
                # minimal_matmul needs matching dtypes; the fused SwiGLU is worth the extra cast
                x2c = ttnn.typecast(x2, self.prec.weight_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(x2)
                x2 = x2c
            m = ttnn.experimental.minimal_matmul(
                x2,
                layer.w_gateup,
                compute_kernel_config=self.ck_mm,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                fuse_swiglu=True,
            )
            ttnn.deallocate(x2)
            d = self._linear(m, layer.w_down)
            ttnn.deallocate(m)
            h = ttnn.add(h2, d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(d)
            ttnn.deallocate(h2)
            if taps is not None and li in taps:
                out_taps[li] = ttnn.to_torch(h)[0, 0, :L]
            if post_add is not None and li in post_add:
                hs = ttnn.add(h, post_add[li], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(h)
                h = hs
        return h, L, out_taps

    def encode(self, input_ids: torch.Tensor, taps: Optional[List[int]] = None):
        """input_ids [1, L] -> hidden states [L, 4096] bf16 (host) of the last layer, pre final-norm."""
        emb = self.embed(input_ids)
        h, L, out_taps = self.forward_device(emb, taps)
        out = ttnn.to_torch(h)[0, 0, :L]
        ttnn.deallocate(h)
        if taps is not None:
            return out, out_taps
        return out
