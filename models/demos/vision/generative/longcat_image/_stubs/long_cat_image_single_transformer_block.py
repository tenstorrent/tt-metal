# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `long_cat_image_single_transformer_block`
(diffusers ``LongCatImageSingleTransformerBlock``, a Flux-style single-stream
MMDiT block) for meituan-longcat/LongCat-Image.

forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb)::

    hs   = cat([encoder_hidden_states, hidden_states], dim=1)      # [B, S, dim]
    res  = hs
    norm_hs, gate = AdaLayerNormZeroSingle(hs, temb)               # LN(no affine)*(1+scale)+shift
    mlp  = gelu_tanh(proj_mlp(norm_hs))                            # [B, S, mlp_dim]
    attn = attn(norm_hs, image_rotary_emb)                        # pre_only single attention
    hs   = gate * proj_out(cat([attn, mlp], dim=2)) + res
    return hs[:, :txt], hs[:, txt:]                                # (encoder_out, hidden_out)

attention (pre_only, no added-kv, no out-proj)::

    q,k,v = to_q/k/v(norm_hs) -> [B, S, heads, head_dim]
    q,k   = RMSNorm(head_dim)(q), RMSNorm(head_dim)(k)
    q,k   = RoPE(q), RoPE(k)          # flux use_real_unbind_dim=-1
    out   = flatten(SDPA(q,k,v, scale=head_dim**-0.5))

Precision: fully fp32 (fp32 activations + fp32 matmul/softmax, HiFi4 / fp32
accumulate). LayerNorm & RMSNorm are manual fp32. RoPE's pair rotation is a fixed
[head_dim,head_dim] ±1 matmul (exact in bf16): out = x*cos + (x@R)*sin, matching
torch apply_rotary_emb(use_real_unbind_dim=-1) for any (cos,sin). The gelu is the
tanh variant. I/O are (B, N, C) sequences; the block returns (encoder, hidden).
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
BF16 = ttnn.bfloat16
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


class _SingleBlock:
    def __init__(self, device, block):
        self.device = device
        self.blk = block.eval() if hasattr(block, "eval") else block
        a = self.blk.attn
        self.heads = int(a.heads)
        self.head_dim = int(a.head_dim)
        self.dim = int(getattr(a, "query_dim", self.heads * self.head_dim))
        self.scale = self.head_dim**-0.5
        self.qk_eps = float(getattr(a.norm_q, "eps", 1e-6))
        gv = ttnn.GeluVariant
        approx = getattr(self.blk.act_mlp, "approximate", "tanh")
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
                R[2 * i, 2 * i + 1] = 1.0  # x_rot[2i+1] = x[2i]
                R[2 * i + 1, 2 * i] = -1.0  # x_rot[2i]   = -x[2i+1]
            self._R = ttnn.from_torch(R, dtype=BF16, layout=TILE, device=self.device, memory_config=DRAM)
        return self._R

    def _layernorm(self, x, eps=1e-6):
        # LayerNorm over last dim, no affine (AdaLayerNormZeroSingle.norm).
        mu = ttnn.mean(x, dim=2, keepdim=True, compute_kernel_config=self._ck())
        xc = ttnn.subtract(x, mu)
        var = ttnn.mean(ttnn.mul(xc, xc), dim=2, keepdim=True, compute_kernel_config=self._ck())
        return ttnn.mul(xc, ttnn.rsqrt(ttnn.add(var, eps)))

    def _rmsnorm(self, x, norm):
        # RMSNorm over last dim (head_dim) with affine weight.
        w = self._rms_weight(norm)
        msq = ttnn.mean(ttnn.mul(x, x), dim=3, keepdim=True, compute_kernel_config=self._ck())
        return ttnn.mul(ttnn.mul(x, ttnn.rsqrt(ttnn.add(msq, self.qk_eps))), w)

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
        # x [1,heads,S,head_dim]; cos/sin [1,1,S,head_dim]; matches flux unbind_dim=-1.
        xr = ttnn.matmul(x, self._R_mat(), compute_kernel_config=self._ck(), dtype=F32)
        return ttnn.add(ttnn.mul(x, cos), ttnn.mul(xr, sin))

    def _attn(self, norm_hs, cos, sin, S):
        a = self.blk.attn
        q = self._split_heads(self._linear(norm_hs, a.to_q), S)
        k = self._split_heads(self._linear(norm_hs, a.to_k), S)
        v = self._split_heads(self._linear(norm_hs, a.to_v), S)
        q = self._rmsnorm(q, a.norm_q)
        k = self._rmsnorm(k, a.norm_k)
        if cos is not None:
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)
        scores = ttnn.matmul(q, k, transpose_b=True, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)
        scores = ttnn.multiply(scores, self.scale)
        probs = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(
            probs, v, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM
        )  # [1,heads,S,head_dim]
        return self._merge_heads(out, S)  # [1,S,dim]  (pre_only: no to_out)

    def _ada_norm(self, hs, temb):
        # AdaLayerNormZeroSingle: emb=linear(silu(temb)); shift,scale,gate=chunk(3);
        # x = LN(hs)*(1+scale)+shift.  temb [1,dim] -> emb [1,3*dim].
        emb = self._linear(ttnn.silu(temb), self.blk.norm.linear)  # [.., 3*dim]
        d = self.dim
        emb = ttnn.reshape(emb, [1, 1, 3 * d])  # force rank 3
        shift = ttnn.slice(emb, [0, 0, 0], [1, 1, d], [1, 1, 1])  # [1,1,dim]
        scale = ttnn.slice(emb, [0, 0, d], [1, 1, 2 * d], [1, 1, 1])
        gate = ttnn.slice(emb, [0, 0, 2 * d], [1, 1, 3 * d], [1, 1, 1])  # [1,1,dim]
        xn = self._layernorm(hs)
        xn = ttnn.add(ttnn.mul(xn, ttnn.add(scale, 1.0)), shift)
        return xn, gate

    def __call__(
        self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None, **_ignored
    ):
        hs = self._to_ttnn(hidden_states)  # [1, img, dim]
        ehs = self._to_ttnn(encoder_hidden_states)  # [1, txt, dim]
        temb_t = self._to_ttnn(temb)  # [1, dim]
        cos = sin = None
        if image_rotary_emb is not None:
            c, s = image_rotary_emb
            c = c.reshape(1, 1, c.shape[-2], c.shape[-1]) if isinstance(c, torch.Tensor) else c
            s = s.reshape(1, 1, s.shape[-2], s.shape[-1]) if isinstance(s, torch.Tensor) else s
            cos = self._to_ttnn(c)
            sin = self._to_ttnn(s)

        txt = ehs.shape[1]
        hs = ttnn.concat([ehs, hs], dim=1)  # [1, S, dim]
        S = hs.shape[1]
        residual = hs

        norm_hs, gate = self._ada_norm(hs, temb_t)
        mlp = ttnn.gelu(self._linear(norm_hs, self.blk.proj_mlp), variant=self.gelu_variant)
        attn_out = self._attn(norm_hs, cos, sin, S)

        cat = ttnn.concat([attn_out, mlp], dim=2)  # [1, S, dim+mlp_dim]
        proj = self._linear(cat, self.blk.proj_out)  # [1, S, dim]
        hs = ttnn.add(residual, ttnn.mul(proj, gate))  # gate [1,1,dim] broadcasts over S

        enc_out = ttnn.slice(hs, [0, 0, 0], [1, txt, self.dim], [1, 1, 1])
        hid_out = ttnn.slice(hs, [0, txt, 0], [1, S, self.dim], [1, 1, 1])
        return enc_out, hid_out


def build(device, torch_module):
    """PCC-harness entry point: native TTNN LongCatImageSingleTransformerBlock."""
    return _SingleBlock(device, torch_module)
