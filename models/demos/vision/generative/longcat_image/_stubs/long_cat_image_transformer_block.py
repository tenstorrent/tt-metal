# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `long_cat_image_transformer_block`
(diffusers ``LongCatImageTransformerBlock``, a Flux-style DUAL-stream MMDiT
block) for meituan-longcat/LongCat-Image.

forward(hidden_states[B,img,dim], encoder_hidden_states[B,txt,dim], temb[B,dim],
        image_rotary_emb=(cos,sin))::

    n_hs, g_msa, s_mlp, sc_mlp, g_mlp        = norm1(hidden_states, temb)         # AdaLayerNormZero
    n_enc, cg_msa, cs_mlp, csc_mlp, cg_mlp   = norm1_context(encoder_hidden_states, temb)
    attn, ctx_attn = attn(n_hs, n_enc, rope)                                       # joint added-kv attention
    hidden = hidden + g_msa * attn
    hidden = hidden + g_mlp  * ff( norm2(hidden)        *(1+sc_mlp)  + s_mlp )
    enc    = enc    + cg_msa * ctx_attn
    enc    = enc    + cg_mlp * ff_context( norm2_context(enc)*(1+csc_mlp)+cs_mlp )
    return enc, hidden

attention (added_kv, joint)::

    img q,k,v = to_q/k/v(n_hs);           q,k = RMSNorm(head_dim)(q),(k)
    txt q,k,v = add_q/k/v_proj(n_enc);    q,k = RMSNorm_added(q),(k)
    q = cat([txt_q, img_q], seq); k,v likewise                # [txt ; img] order
    q,k = RoPE(q), RoPE(k)                                     # flux use_real_unbind_dim=-1
    out = flatten(SDPA(q,k,v, scale=head_dim**-0.5))
    enc, img = out[:, :txt], out[:, txt:]
    return to_out[0](img), to_add_out(enc)

Precision: fully fp32 (fp32 activations + matmul/softmax, HiFi4 / fp32-accumulate).
LayerNorm (no-affine) and RMSNorm are manual fp32. RoPE's pair rotation is a fixed
[head_dim,head_dim] +-1 matmul (exact in bf16): out = x*cos + (x@R)*sin, matching
torch apply_rotary_emb(use_real_unbind_dim=-1) for any (cos,sin). The gelu is the
tanh variant. The block returns (encoder, hidden) matching the reference.
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
BF16 = ttnn.bfloat16
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


class _DoubleBlock:
    def __init__(self, device, block):
        self.device = device
        self.blk = block.eval() if hasattr(block, "eval") else block
        a = self.blk.attn
        self.heads = int(a.heads)
        self.head_dim = int(a.head_dim)
        self.dim = int(getattr(a, "query_dim", self.heads * self.head_dim))
        self.scale = self.head_dim**-0.5
        gv = ttnn.GeluVariant
        approx = getattr(self.blk.ff.net[0], "approximate", "tanh")
        self.gelu_variant = gv.Tanh if approx == "tanh" else gv.Accurate
        self._lin = {}
        self._rms = {}
        self._R = None
        self._compute = None

    # ── helpers ──────────────────────────────────────────────────────────
    def _ck(self):
        if self._compute is None:
            self._compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
            )
        return self._compute

    def _to_ttnn(self, t):
        if isinstance(t, ttnn.Tensor):
            if t.layout != TILE:
                t = ttnn.to_layout(t, TILE)
            if t.dtype != F32:
                t = ttnn.typecast(t, F32)
            return t
        return ttnn.from_torch(t.to(torch.float32), dtype=F32, layout=TILE, device=self.device, memory_config=DRAM)

    def _linear(self, x, tm):
        key = id(tm)
        if key not in self._lin:
            b = tm.bias
            self._lin[key] = (
                ttnn.from_torch(
                    tm.weight.detach().to(torch.float32), dtype=F32, layout=TILE, device=self.device, memory_config=DRAM
                ),
                ttnn.from_torch(
                    b.detach().reshape(1, 1, -1).to(torch.float32),
                    dtype=F32,
                    layout=TILE,
                    device=self.device,
                    memory_config=DRAM,
                )
                if b is not None
                else None,
            )
        wt, bt = self._lin[key]
        return ttnn.linear(
            x, wt, bias=bt, transpose_b=True, memory_config=DRAM, dtype=F32, compute_kernel_config=self._ck()
        )

    def _rms_weight(self, norm):
        key = id(norm)
        if key not in self._rms:
            w = norm.weight.detach().reshape(1, 1, 1, -1).to(torch.float32)
            self._rms[key] = ttnn.from_torch(w, dtype=F32, layout=TILE, device=self.device, memory_config=DRAM)
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
        mu = ttnn.mean(x, dim=2, keepdim=True, compute_kernel_config=self._ck())
        xc = ttnn.subtract(x, mu)
        var = ttnn.mean(ttnn.mul(xc, xc), dim=2, keepdim=True, compute_kernel_config=self._ck())
        return ttnn.mul(xc, ttnn.rsqrt(ttnn.add(var, eps)))

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
        xr = ttnn.matmul(x, self._R_mat(), compute_kernel_config=self._ck(), dtype=F32)
        return ttnn.add(ttnn.mul(x, cos), ttnn.mul(xr, sin))

    def _ada_mod(self, temb, lin, n):
        # emb = lin(silu(temb)); chunk into n pieces of `dim`, each [1,1,dim].
        emb = self._linear(ttnn.silu(temb), lin)
        d = self.dim
        emb = ttnn.reshape(emb, [1, 1, n * d])
        return [ttnn.slice(emb, [0, 0, i * d], [1, 1, (i + 1) * d], [1, 1, 1]) for i in range(n)]

    # ── attention (dual-stream, added_kv) ────────────────────────────────
    def _attn(self, norm_hs, norm_enc, cos, sin, img_len, txt_len):
        a = self.blk.attn
        q = self._rmsnorm(self._split_heads(self._linear(norm_hs, a.to_q), img_len), a.norm_q)
        k = self._rmsnorm(self._split_heads(self._linear(norm_hs, a.to_k), img_len), a.norm_k)
        v = self._split_heads(self._linear(norm_hs, a.to_v), img_len)

        eq = self._rmsnorm(self._split_heads(self._linear(norm_enc, a.add_q_proj), txt_len), a.norm_added_q)
        ek = self._rmsnorm(self._split_heads(self._linear(norm_enc, a.add_k_proj), txt_len), a.norm_added_k)
        ev = self._split_heads(self._linear(norm_enc, a.add_v_proj), txt_len)

        q = ttnn.concat([eq, q], dim=2)  # [1,heads,txt+img,head_dim]  ([txt ; img])
        k = ttnn.concat([ek, k], dim=2)
        v = ttnn.concat([ev, v], dim=2)
        S = txt_len + img_len
        if cos is not None:
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)

        scores = ttnn.matmul(q, k, transpose_b=True, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)
        scores = ttnn.multiply(scores, self.scale)
        probs = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(probs, v, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)
        out = self._merge_heads(out, S)  # [1,S,dim]

        enc_attn = ttnn.slice(out, [0, 0, 0], [1, txt_len, self.dim], [1, 1, 1])
        img_attn = ttnn.slice(out, [0, txt_len, 0], [1, S, self.dim], [1, 1, 1])
        img_attn = self._linear(img_attn, a.to_out[0])
        enc_attn = self._linear(enc_attn, a.to_add_out)
        return img_attn, enc_attn  # (hidden, encoder)

    def _feed_forward(self, ff, x):
        net = ff.net
        x = self._linear(x, net[0].proj)  # dim -> inner
        x = ttnn.gelu(x, variant=self.gelu_variant)  # gelu-approximate (tanh)
        for mod in net[1:]:
            if isinstance(mod, torch.nn.Linear):
                x = self._linear(x, mod)
        return x

    def __call__(
        self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None, **_ignored
    ):
        hid = self._to_ttnn(hidden_states)  # [1, img, dim]
        enc = self._to_ttnn(encoder_hidden_states)  # [1, txt, dim]
        temb_t = self._to_ttnn(temb)  # [1, dim]
        cos = sin = None
        if image_rotary_emb is not None:
            c, s = image_rotary_emb
            c = c.reshape(1, 1, c.shape[-2], c.shape[-1]) if isinstance(c, torch.Tensor) else c
            s = s.reshape(1, 1, s.shape[-2], s.shape[-1]) if isinstance(s, torch.Tensor) else s
            cos = self._to_ttnn(c)
            sin = self._to_ttnn(s)

        img_len = hid.shape[1]
        txt_len = enc.shape[1]

        # norm1 (img): AdaLayerNormZero -> (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._ada_mod(temb_t, self.blk.norm1.linear, 6)
        norm_hs = ttnn.add(ttnn.mul(self._layernorm(hid), ttnn.add(scale_msa, 1.0)), shift_msa)
        # norm1_context (txt)
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self._ada_mod(
            temb_t, self.blk.norm1_context.linear, 6
        )
        norm_enc = ttnn.add(ttnn.mul(self._layernorm(enc), ttnn.add(c_scale_msa, 1.0)), c_shift_msa)

        attn_out, ctx_attn_out = self._attn(norm_hs, norm_enc, cos, sin, img_len, txt_len)

        # img stream
        hid = ttnn.add(hid, ttnn.mul(attn_out, gate_msa))
        norm_hs2 = self._layernorm(hid)
        norm_hs2 = ttnn.add(ttnn.mul(norm_hs2, ttnn.add(scale_mlp, 1.0)), shift_mlp)
        ff_out = self._feed_forward(self.blk.ff, norm_hs2)
        hid = ttnn.add(hid, ttnn.mul(ff_out, gate_mlp))

        # txt stream
        enc = ttnn.add(enc, ttnn.mul(ctx_attn_out, c_gate_msa))
        norm_enc2 = self._layernorm(enc)
        norm_enc2 = ttnn.add(ttnn.mul(norm_enc2, ttnn.add(c_scale_mlp, 1.0)), c_shift_mlp)
        ctx_ff_out = self._feed_forward(self.blk.ff_context, norm_enc2)
        enc = ttnn.add(enc, ttnn.mul(ctx_ff_out, c_gate_mlp))

        return enc, hid  # (encoder, hidden) — matches reference return order


def build(device, torch_module):
    """PCC-harness entry point: native TTNN LongCatImageTransformerBlock."""
    return _DoubleBlock(device, torch_module)
