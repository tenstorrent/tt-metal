# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of QwenImage joint (dual-stream) attention
(`transformer_blocks.N.attn`, diffusers `QwenDoubleStreamAttnProcessor2_0`).

    img: q,k,v = to_q/k/v(hidden_states)            txt: q,k,v = add_q/k/v_proj(encoder_hidden_states)
    q,k        = RMSNorm_head(q,k)  -> RoPE(complex, interleaved pairs)
    joint      = SDPA(cat[txt, img] q/k/v)           (non-causal, optional key mask)
    img_out    = to_out[0](joint[img])               txt_out = to_add_out(joint[txt])

Tensor parallel (TP = mesh size): heads are split across chips.
  * q/k/v and add_q/k/v projections are COLUMN-parallel (output features = whole heads per chip).
  * to_out / to_add_out are ROW-parallel (input features = this chip's heads) followed by all_reduce;
    their biases are added once, after the reduce.
  * RMSNorm weights and rotary tables are replicated.

Numerics. The QK-RMSNorm weights of this checkpoint are large (up to ~65), so attention logits reach
~4e4 and softmax is nearly an argmax: rounding normalized q/k to bf16 alone drops PCC to ~0.93 and a
TF32 matmul to ~0.988. So everything from the q/k projection outputs through the logits stays in
float32 on the SFPU (elementwise), and QK^T is computed as a 3-term bf16 hi/lo split with float32
accumulation (qh.kh + qh.kl + ql.kh), which carries ~16 mantissa bits. P@V and the output
projections are insensitive and run as bf16 matmuls with float32 accumulation.

RoPE without a pair-swap: q/k projection output features are permuted per head from interleaved
(x0,x1,x2,x3,...) to half-split (x0,x2,...,x1,x3,...) order at build time (norm weights likewise).
The dot product q.k is invariant under a shared permutation, and in half-split order the complex
rotation is pure slicing + elementwise math. V is not permuted, so the attention output layout is
unchanged.

Rotary input convention (the harness stages complex tensors this way): each freqs tensor is a
float32 ttnn tensor [S, 2*(D/2)] = [cos | sin].
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _ccl, _precise


def _is_mesh(device):
    return isinstance(device, ttnn.MeshDevice) and device.get_num_devices() > 1


def _replicated(t, device, dtype=ttnn.float32):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kw)


def _sharded(t, device, dim, dtype=ttnn.bfloat16):
    if _is_mesh(device):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim),
        )
    return _replicated(t, device, dtype=dtype)


def _as_tt(t, device, dtype=ttnn.float32):
    """Inputs normally arrive on device already; stage a host tensor if one does not."""
    if t is None or isinstance(t, ttnn.Tensor):
        return t
    if t.is_complex():
        t = torch.cat([t.real, t.imag], dim=-1)
    return _replicated(t.to(torch.float32), device, dtype=dtype)


class TtQwenJointAttention:
    def __init__(self, device, torch_module):
        self.device = device
        m = torch_module
        self.n_heads = int(m.heads)
        inner = m.to_q.weight.shape[0]
        self.head_dim = inner // self.n_heads
        self.tp = device.get_num_devices() if _is_mesh(device) else 1
        assert self.n_heads % self.tp == 0, f"{self.n_heads} heads not divisible by TP={self.tp}"
        self.local_heads = self.n_heads // self.tp
        self.half = self.head_dim // 2

        hd = self.head_dim
        in_head = torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)])  # interleaved -> half-split
        perm = torch.cat([h * hd + in_head for h in range(self.n_heads)])

        def w_col(lin, permute):
            w = lin.weight.detach().to(torch.float32)
            b = lin.bias.detach().to(torch.float32) if lin.bias is not None else torch.zeros(w.shape[0])
            if permute:
                w, b = w[perm], b[perm]
            return _sharded(w.t(), device, dim=-1), _sharded(b.reshape(1, -1), device, dim=-1, dtype=ttnn.float32)

        self.img_q = w_col(m.to_q, True)
        self.img_k = w_col(m.to_k, True)
        self.img_v = w_col(m.to_v, False)
        self.txt_q = w_col(m.add_q_proj, True)
        self.txt_k = w_col(m.add_k_proj, True)
        self.txt_v = w_col(m.add_v_proj, False)

        def w_row(lin):
            w = lin.weight.detach().to(torch.float32)
            b = lin.bias.detach().to(torch.float32) if lin.bias is not None else torch.zeros(w.shape[0])
            return _sharded(w.t(), device, dim=-2), _replicated(b.reshape(1, -1), device)

        self.img_out = w_row(m.to_out[0])
        self.txt_out = w_row(m.to_add_out)

        scale = 1.0 / (hd**0.5)

        def norm_w(norm, q_scale):
            if norm is None:
                w = torch.ones(hd)
            else:
                w = norm.weight.detach().to(torch.float32) if norm.weight is not None else torch.ones(hd)
            w = w[in_head] * q_scale  # softmax scale folded into the query norm weight
            return _replicated(w.reshape(1, 1, 1, hd), device)

        self.eps = float(getattr(m.norm_q, "eps", 1e-6) or 1e-6)
        self.nw_img_q = norm_w(m.norm_q, scale)
        self.nw_img_k = norm_w(m.norm_k, 1.0)
        self.nw_txt_q = norm_w(m.norm_added_q, scale)
        self.nw_txt_k = norm_w(m.norm_added_k, 1.0)

        self.hifi = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    # ---- helpers -------------------------------------------------------------------------------
    def _proj(self, x, wb, dtype=ttnn.float32):
        w, b = wb
        if _precise.ENABLED:
            return _precise.linear(x, w, bias=b)
        return ttnn.linear(x, w, bias=b, dtype=dtype, compute_kernel_config=self.hifi)

    def _heads(self, x):  # [B, S, Hl*D] -> [B, Hl, S, D]
        B, S = x.shape[0], x.shape[1]
        x = ttnn.reshape(x, (B, S, self.local_heads, self.head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _rms(self, x, w):  # float32, elementwise on SFPU
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True, compute_kernel_config=self.hifi)
        r = ttnn.rsqrt(ttnn.add(ms, self.eps))
        return ttnn.multiply(ttnn.multiply(x, r), w)

    def _rope(self, x, freqs):  # x [B, Hl, S, D] half-split; freqs [S, D] = [cos | sin]
        if freqs is None:
            return x
        S = freqs.shape[0]
        f = ttnn.reshape(freqs, (1, 1, S, 2 * self.half))
        cos = ttnn.slice(f, (0, 0, 0, 0), (1, 1, S, self.half))
        sin = ttnn.slice(f, (0, 0, 0, self.half), (1, 1, S, 2 * self.half))
        B, H = x.shape[0], x.shape[1]
        x1 = ttnn.slice(x, (0, 0, 0, 0), (B, H, S, self.half))
        x2 = ttnn.slice(x, (0, 0, 0, self.half), (B, H, S, 2 * self.half))
        o1 = ttnn.subtract(ttnn.multiply(x1, cos), ttnn.multiply(x2, sin))
        o2 = ttnn.add(ttnn.multiply(x1, sin), ttnn.multiply(x2, cos))
        return ttnn.concat([o1, o2], dim=-1)

    @staticmethod
    def _split(x):  # float32 -> (hi, lo) bf16 with x ~= hi + lo
        hi = ttnn.typecast(x, ttnn.bfloat16)
        lo = ttnn.typecast(ttnn.subtract(x, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)
        return hi, lo

    def _qk(self, q, kh, kl):  # 3-term split QK^T, float32 out
        qh, ql = self._split(q)
        mm = lambda a, b: ttnn.matmul(a, b, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        return ttnn.add(ttnn.add(mm(qh, kh), mm(qh, kl)), mm(ql, kh))

    def _attend(self, q, kh, kl, v, mask_add):
        if _precise.ENABLED:  # kh = k (float32) in precise mode
            s = _precise.exact_matmul_bt(q, kh) if _precise.EXACT_QK else _precise.matmul_bt(q, kh)
        else:
            s = self._qk(q, kh, kl)
        if mask_add is not None:
            s = ttnn.add(s, mask_add)
        # Explicit softmax: the fused ttnn.softmax loses ~0.2 abs on these ~1e4 logits when the key
        # length is not tile-aligned; max/exp/sum/divide on float32 stays within ~2e-3.
        mx = ttnn.max(s, dim=-1, keepdim=True)
        e = ttnn.exp(ttnn.subtract(s, mx))
        p = ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=self.hifi))
        if _precise.ENABLED:
            o = _precise.matmul(p, v)  # float32 [B, Hl, Sq, D]
        else:
            o = ttnn.matmul(
                ttnn.typecast(p, ttnn.bfloat16), v, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi
            )  # [B, Hl, Sq, D]
        B, Sq = o.shape[0], o.shape[2]
        o = ttnn.permute(o, (0, 2, 1, 3))
        return ttnn.reshape(o, (B, Sq, self.local_heads * self.head_dim))

    def _out(self, x, wb):  # row-parallel + all_reduce, bias once
        w, b = wb
        if _precise.ENABLED:
            y = _precise.linear(x, w)
        else:
            y = ttnn.linear(x, w, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        if self.tp > 1:
            y = _ccl.all_reduce(y, self.device)
        return ttnn.add(y, b)

    # ---- forward -------------------------------------------------------------------------------
    def __call__(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        d = self.device
        hs = _as_tt(hidden_states, d, ttnn.float32 if _precise.ENABLED else ttnn.bfloat16)
        ehs = _as_tt(encoder_hidden_states, d, ttnn.float32 if _precise.ENABLED else ttnn.bfloat16)
        vdt = ttnn.float32 if _precise.ENABLED else ttnn.bfloat16
        img_f, txt_f = (None, None) if image_rotary_emb is None else image_rotary_emb
        img_f, txt_f = _as_tt(img_f, d), _as_tt(txt_f, d)

        qi = self._rope(self._rms(self._heads(self._proj(hs, self.img_q)), self.nw_img_q), img_f)
        ki = self._rope(self._rms(self._heads(self._proj(hs, self.img_k)), self.nw_img_k), img_f)
        vi = self._heads(self._proj(hs, self.img_v, vdt))
        qt = self._rope(self._rms(self._heads(self._proj(ehs, self.txt_q)), self.nw_txt_q), txt_f)
        kt = self._rope(self._rms(self._heads(self._proj(ehs, self.txt_k)), self.nw_txt_k), txt_f)
        vt = self._heads(self._proj(ehs, self.txt_v, vdt))

        # Joint sequence order is [text, image].
        k = ttnn.concat([kt, ki], dim=2)
        v = ttnn.concat([vt, vi], dim=2)
        kh, kl = (k, None) if _precise.ENABLED else self._split(k)

        mask_add = None
        if attention_mask is not None:
            m = _as_tt(attention_mask, d)  # 1 = attend, 0 = masked; [B, 1, 1, S_joint]
            mask_add = ttnn.multiply(ttnn.subtract(m, 1.0), 1e30)

        img = self._out(self._attend(qi, kh, kl, v, mask_add), self.img_out)
        txt = self._out(self._attend(qt, kh, kl, v, mask_add), self.txt_out)
        return img, txt


def build(device, torch_module=None):
    return TtQwenJointAttention(device, torch_module)


def attention(device, torch_module=None):
    return TtQwenJointAttention(device, torch_module)
