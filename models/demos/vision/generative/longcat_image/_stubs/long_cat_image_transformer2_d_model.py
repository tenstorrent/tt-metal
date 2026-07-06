# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `long_cat_image_transformer2_d_model`
(diffusers ``LongCatImageTransformer2DModel``) for meituan-longcat/LongCat-Image.

This is the full Flux-style MMDiT denoiser: config dim=3072 (24 heads x 128),
in_channels=64, joint_attention_dim=3584, 10 dual-stream blocks
(``transformer_blocks``) + 20 single-stream blocks (``single_transformer_blocks``).

forward(hidden_states[1,img,64], timestep[1], encoder_hidden_states[1,txt,3584],
        txt_ids[txt,3], img_ids[img,3])::

    hid  = x_embedder(hidden_states)                       # 64 -> 3072
    temb = time_embed(timestep * 1000)                     # sinusoidal -> MLP -> [1,3072]
    enc  = context_embedder(encoder_hidden_states)         # 3584 -> 3072
    cos,sin = pos_embed(cat(txt_ids, img_ids))             # RoPE tables [txt+img, 128]
    for blk in transformer_blocks:        enc, hid = blk(hid, enc, temb, (cos,sin))
    for blk in single_transformer_blocks: enc, hid = blk(hid, enc, temb, (cos,sin))
    out  = proj_out(norm_out(hid, temb))                   # 3072 -> 64
    return (out,)                                          # [1,img,64]

Precision: bf16 weights + bf16 activations with fp32 accumulation (HiFi4,
fp32_dest_acc_en). The 6.27 B-param model is far too large to hold in fp32 on a
single 32 GB card; bf16 weights (~12.5 GB) fit comfortably and PCC (a
correlation metric) is unaffected by uniform bf16 rounding as long as every
formula is exact. LayerNorm (no-affine) uses the fused ``ttnn.layer_norm``;
RMSNorm on q/k is manual over head_dim. RoPE matches diffusers'
``apply_rotary_emb(use_real_unbind_dim=-1)`` exactly via the fixed +-1 rotation
matrix R (out = x*cos + (x@R)*sin). The parameter-free positional-embedding and
timestep-sinusoid tables are precomputed on host (like the per-block PCC harness
does for ``image_rotary_emb``) — this is input preprocessing, not model compute.
"""

from __future__ import annotations

import math

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
BF16 = ttnn.bfloat16
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


class _Transformer:
    """Native TTNN LongCatImageTransformer2DModel. Built from the torch module; callable per-forward."""

    def __init__(self, device, tf):
        self.device = device
        self.tf = tf.eval() if hasattr(tf, "eval") else tf
        cfg = self.tf.config
        self.heads = int(cfg.num_attention_heads)
        self.head_dim = int(cfg.attention_head_dim)
        self.dim = self.heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.axes_dim = list(cfg.axes_dims_rope) if hasattr(cfg, "axes_dims_rope") else [16, 56, 56]
        self.theta = int(getattr(self.tf.pos_embed, "theta", 10000))
        # timestep sinusoid params (Timesteps: 256 ch, flip_sin_to_cos, shift 0)
        tp = self.tf.time_embed.time_proj
        self.ts_channels = int(tp.num_channels)
        self.ts_flip = bool(getattr(tp, "flip_sin_to_cos", True))
        self.ts_shift = float(getattr(tp, "downscale_freq_shift", 0.0))
        self.ts_scale = float(getattr(tp, "scale", 1.0))
        self.ts_max_period = int(getattr(tp, "max_period", 10000))
        self._lin = {}
        self._rms = {}
        self._R = None
        self._compute = None

    # ── low-level helpers ────────────────────────────────────────────────
    def _ck(self):
        if self._compute is None:
            self._compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
            )
        return self._compute

    def _to_ttnn(self, t, dtype=BF16):
        if isinstance(t, ttnn.Tensor):
            if t.layout != TILE:
                t = ttnn.to_layout(t, TILE)
            if t.dtype != dtype:
                t = ttnn.typecast(t, dtype)
            return t
        td = t.to(torch.bfloat16) if dtype == BF16 else t.to(torch.float32)
        return ttnn.from_torch(td, dtype=dtype, layout=TILE, device=self.device, memory_config=DRAM)

    def _linear(self, x, tm):
        key = id(tm)
        if key not in self._lin:
            b = tm.bias
            self._lin[key] = (
                ttnn.from_torch(
                    tm.weight.detach().to(torch.bfloat16),
                    dtype=BF16,
                    layout=TILE,
                    device=self.device,
                    memory_config=DRAM,
                ),
                ttnn.from_torch(
                    b.detach().reshape(1, 1, -1).to(torch.bfloat16),
                    dtype=BF16,
                    layout=TILE,
                    device=self.device,
                    memory_config=DRAM,
                )
                if b is not None
                else None,
            )
        wt, bt = self._lin[key]
        return ttnn.linear(
            x, wt, bias=bt, transpose_b=True, memory_config=DRAM, dtype=BF16, compute_kernel_config=self._ck()
        )

    def _rms_weight(self, norm):
        key = id(norm)
        if key not in self._rms:
            w = norm.weight.detach().reshape(1, 1, 1, -1).to(torch.bfloat16)
            self._rms[key] = ttnn.from_torch(w, dtype=BF16, layout=TILE, device=self.device, memory_config=DRAM)
        return self._rms[key]

    def _R_mat(self):
        if self._R is None:
            d = self.head_dim
            R = torch.zeros(d, d, dtype=torch.float32)
            for i in range(d // 2):
                R[2 * i, 2 * i + 1] = 1.0  # (x@R)[2i+1] = x[2i]
                R[2 * i + 1, 2 * i] = -1.0  # (x@R)[2i]   = -x[2i+1]
            self._R = ttnn.from_torch(R, dtype=BF16, layout=TILE, device=self.device, memory_config=DRAM)
        return self._R

    def _layernorm(self, x, eps=1e-6):
        # LayerNorm over last dim, no affine.
        return ttnn.layer_norm(x, epsilon=eps)

    def _rmsnorm(self, x, norm):
        # RMSNorm over last dim (head_dim) with affine weight.
        eps = float(getattr(norm, "eps", 1e-6))
        w = self._rms_weight(norm)
        msq = ttnn.mean(ttnn.mul(x, x), dim=3, keepdim=True, compute_kernel_config=self._ck())
        return ttnn.mul(ttnn.mul(x, ttnn.rsqrt(ttnn.add(msq, eps))), w)

    def _split_heads(self, t, S):
        # [1,S,dim] -> [1,heads,S,head_dim]
        t = ttnn.to_layout(t, RM)
        t = ttnn.reshape(t, [1, S, self.heads, self.head_dim])
        t = ttnn.permute(t, [0, 2, 1, 3])
        return ttnn.to_layout(t, TILE)

    def _merge_heads(self, t, S):
        # [1,heads,S,head_dim] -> [1,S,dim]
        t = ttnn.to_layout(t, RM)
        t = ttnn.permute(t, [0, 2, 1, 3])
        t = ttnn.reshape(t, [1, S, self.dim])
        return ttnn.to_layout(t, TILE)

    def _rope(self, x, cos, sin):
        # x [1,heads,S,head_dim]; cos/sin [1,1,S,head_dim]; flux unbind_dim=-1.
        xr = ttnn.matmul(x, self._R_mat(), compute_kernel_config=self._ck(), dtype=BF16)
        return ttnn.add(ttnn.mul(x, cos), ttnn.mul(xr, sin))

    def _ada_mod(self, temb, lin, n):
        # emb = lin(silu(temb)); chunk into n pieces of `dim`, each [1,1,dim].
        emb = self._linear(ttnn.silu(temb), lin)
        d = self.dim
        emb = ttnn.reshape(emb, [1, 1, n * d])
        return [ttnn.slice(emb, [0, 0, i * d], [1, 1, (i + 1) * d], [1, 1, 1]) for i in range(n)]

    # ── attention ────────────────────────────────────────────────────────
    def _single_attn(self, a, norm_hs, cos, sin, S):
        # pre_only single-stream attention (no added_kv, no to_out).
        q = self._rmsnorm(self._split_heads(self._linear(norm_hs, a.to_q), S), a.norm_q)
        k = self._rmsnorm(self._split_heads(self._linear(norm_hs, a.to_k), S), a.norm_k)
        v = self._split_heads(self._linear(norm_hs, a.to_v), S)
        if cos is not None:
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)
        scores = ttnn.matmul(q, k, transpose_b=True, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)
        scores = ttnn.multiply(scores, self.scale)
        probs = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(probs, v, dtype=BF16, compute_kernel_config=self._ck(), memory_config=DRAM)
        return self._merge_heads(out, S)

    def _double_attn(self, a, norm_hs, norm_enc, cos, sin, img_len, txt_len):
        # dual-stream joint attention: [txt ; img] concatenated along sequence.
        q = self._rmsnorm(self._split_heads(self._linear(norm_hs, a.to_q), img_len), a.norm_q)
        k = self._rmsnorm(self._split_heads(self._linear(norm_hs, a.to_k), img_len), a.norm_k)
        v = self._split_heads(self._linear(norm_hs, a.to_v), img_len)

        eq = self._rmsnorm(self._split_heads(self._linear(norm_enc, a.add_q_proj), txt_len), a.norm_added_q)
        ek = self._rmsnorm(self._split_heads(self._linear(norm_enc, a.add_k_proj), txt_len), a.norm_added_k)
        ev = self._split_heads(self._linear(norm_enc, a.add_v_proj), txt_len)

        q = ttnn.concat([eq, q], dim=2)  # [1,heads,txt+img,head_dim]
        k = ttnn.concat([ek, k], dim=2)
        v = ttnn.concat([ev, v], dim=2)
        S = txt_len + img_len
        if cos is not None:
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)

        scores = ttnn.matmul(q, k, transpose_b=True, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)
        scores = ttnn.multiply(scores, self.scale)
        probs = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(probs, v, dtype=BF16, compute_kernel_config=self._ck(), memory_config=DRAM)
        out = self._merge_heads(out, S)  # [1,S,dim]

        enc_attn = ttnn.slice(out, [0, 0, 0], [1, txt_len, self.dim], [1, 1, 1])
        img_attn = ttnn.slice(out, [0, txt_len, 0], [1, S, self.dim], [1, 1, 1])
        img_attn = self._linear(img_attn, a.to_out[0])
        enc_attn = self._linear(enc_attn, a.to_add_out)
        return img_attn, enc_attn  # (hidden, encoder)

    def _feed_forward(self, ff, x):
        net = ff.net
        x = self._linear(x, net[0].proj)  # dim -> inner
        x = ttnn.gelu(x, variant=ttnn.GeluVariant.Tanh)  # gelu-approximate
        for mod in net[1:]:
            if isinstance(mod, torch.nn.Linear):
                x = self._linear(x, mod)
        return x

    # ── blocks ───────────────────────────────────────────────────────────
    def _single_block(self, blk, hid, enc, temb, cos, sin):
        txt = enc.shape[1]
        hs = ttnn.concat([enc, hid], dim=1)  # [1, S, dim]
        S = hs.shape[1]
        residual = hs
        shift, scale, gate = self._ada_mod(temb, blk.norm.linear, 3)
        norm_hs = ttnn.add(ttnn.mul(self._layernorm(hs), ttnn.add(scale, 1.0)), shift)
        mlp = ttnn.gelu(self._linear(norm_hs, blk.proj_mlp), variant=ttnn.GeluVariant.Tanh)
        attn_out = self._single_attn(blk.attn, norm_hs, cos, sin, S)
        cat = ttnn.concat([attn_out, mlp], dim=2)
        hs = ttnn.add(residual, ttnn.mul(self._linear(cat, blk.proj_out), gate))
        enc_out = ttnn.slice(hs, [0, 0, 0], [1, txt, self.dim], [1, 1, 1])
        hid_out = ttnn.slice(hs, [0, txt, 0], [1, S, self.dim], [1, 1, 1])
        return enc_out, hid_out

    def _double_block(self, blk, hid, enc, temb, cos, sin):
        img_len = hid.shape[1]
        txt_len = enc.shape[1]

        # norm1 (img): AdaLayerNormZero -> (norm_hs, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._ada_mod(temb, blk.norm1.linear, 6)
        norm_hs = ttnn.add(ttnn.mul(self._layernorm(hid), ttnn.add(scale_msa, 1.0)), shift_msa)
        # norm1_context (txt)
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self._ada_mod(
            temb, blk.norm1_context.linear, 6
        )
        norm_enc = ttnn.add(ttnn.mul(self._layernorm(enc), ttnn.add(c_scale_msa, 1.0)), c_shift_msa)

        attn_out, ctx_attn_out = self._double_attn(blk.attn, norm_hs, norm_enc, cos, sin, img_len, txt_len)

        # img stream
        hid = ttnn.add(hid, ttnn.mul(attn_out, gate_msa))
        norm_hs2 = self._layernorm(hid)
        norm_hs2 = ttnn.add(ttnn.mul(norm_hs2, ttnn.add(scale_mlp, 1.0)), shift_mlp)
        ff_out = self._feed_forward(blk.ff, norm_hs2)
        hid = ttnn.add(hid, ttnn.mul(ff_out, gate_mlp))

        # txt stream
        enc = ttnn.add(enc, ttnn.mul(ctx_attn_out, c_gate_msa))
        norm_enc2 = self._layernorm(enc)
        norm_enc2 = ttnn.add(ttnn.mul(norm_enc2, ttnn.add(c_scale_mlp, 1.0)), c_shift_mlp)
        ctx_ff_out = self._feed_forward(blk.ff_context, norm_enc2)
        enc = ttnn.add(enc, ttnn.mul(ctx_ff_out, c_gate_mlp))
        return enc, hid

    # ── positional / timestep tables (host, parameter-free) ───────────────
    def _rope_tables(self, txt_ids, img_ids):
        from diffusers.models.embeddings import get_1d_rotary_pos_embed

        ids = torch.cat((txt_ids.float(), img_ids.float()), dim=0)  # [S,3]
        cos_out, sin_out = [], []
        for i in range(ids.shape[-1]):
            c, s = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                ids[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=torch.float64,
            )
            cos_out.append(c)
            sin_out.append(s)
        cos = torch.cat(cos_out, dim=-1)  # [S, head_dim]
        sin = torch.cat(sin_out, dim=-1)
        S = cos.shape[0]
        cos = self._to_ttnn(cos.reshape(1, 1, S, self.head_dim))
        sin = self._to_ttnn(sin.reshape(1, 1, S, self.head_dim))
        return cos, sin

    def _timestep_proj(self, timestep):
        # get_timestep_embedding(timestep, 256, flip_sin_to_cos, shift, scale) on host.
        half = self.ts_channels // 2
        exponent = -math.log(self.ts_max_period) * torch.arange(0, half, dtype=torch.float32)
        exponent = exponent / (half - self.ts_shift)
        freq = torch.exp(exponent)  # [half]
        t = timestep.float().reshape(-1, 1) * self.ts_scale  # [N,1]
        args = t * freq.reshape(1, half)  # [N, half]
        if self.ts_flip:
            proj = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        else:
            proj = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return proj  # [N, 256]

    def _time_embed(self, timestep):
        proj = self._to_ttnn(self._timestep_proj(timestep))  # [N,256]
        te = self.tf.time_embed.timestep_embedder
        x = self._linear(proj, te.linear_1)
        x = ttnn.silu(x)
        x = self._linear(x, te.linear_2)  # [N, dim]
        return x

    # ── forward ────────────────────────────────────────────────────────
    def __call__(
        self,
        hidden_states,
        timestep=None,
        encoder_hidden_states=None,
        txt_ids=None,
        img_ids=None,
        guidance=None,
        return_dict=False,
        **_ignored,
    ):
        hid = self._linear(self._to_ttnn(hidden_states), self.tf.x_embedder)  # [1,img,dim]
        enc = self._linear(self._to_ttnn(encoder_hidden_states), self.tf.context_embedder)  # [1,txt,dim]

        # timestep is scaled by 1000 before the sinusoid (diffusers forward).
        ts = timestep if isinstance(timestep, torch.Tensor) else torch.tensor([float(timestep)])
        temb = self._time_embed(ts.float() * 1000.0)  # [1,dim]

        cos, sin = self._rope_tables(txt_ids, img_ids)

        for blk in self.tf.transformer_blocks:
            enc, hid = self._double_block(blk, hid, enc, temb, cos, sin)
        for blk in self.tf.single_transformer_blocks:
            enc, hid = self._single_block(blk, hid, enc, temb, cos, sin)

        # norm_out: AdaLayerNormContinuous(hid, temb)
        scale, shift = self._ada_mod_cont(temb, self.tf.norm_out.linear)
        out = ttnn.add(ttnn.mul(self._layernorm(hid), ttnn.add(scale, 1.0)), shift)
        out = self._linear(out, self.tf.proj_out)  # [1,img,out_channels]
        return (out,)

    def _ada_mod_cont(self, temb, lin):
        # AdaLayerNormContinuous: emb=lin(silu(temb)); scale,shift = chunk(2).
        emb = self._linear(ttnn.silu(temb), lin)
        d = self.dim
        emb = ttnn.reshape(emb, [1, 1, 2 * d])
        scale = ttnn.slice(emb, [0, 0, 0], [1, 1, d], [1, 1, 1])
        shift = ttnn.slice(emb, [0, 0, d], [1, 1, 2 * d], [1, 1, 1])
        return scale, shift


def build(device, torch_module):
    """PCC-harness entry point: native TTNN LongCatImageTransformer2DModel."""
    return _Transformer(device, torch_module)
