# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `qwen2_v_l_vision_block` (transformers
``Qwen2_5_VLVisionBlock``) — one block of the LongCat-Image text encoder's
Qwen2.5-VL vision tower, submodule ``text_encoder.model.visual.blocks.0``.

forward(hidden_states[seq,dim], cu_seqlens, position_embeddings=(cos,sin))::

    h = hidden_states + attn(norm1(hidden_states), cu_seqlens, (cos,sin))
    h = h + mlp(norm2(h))
    return h                                              # [seq, dim]

Vision attention is FULL (non-causal, no GQA — num_heads == num_kv_heads):
``qkv`` projects to 3·dim, the 3·dim axis splits as [q(dim) | k(dim) | v(dim)]
(the reshape is ``[seq,3,heads,head_dim]``), 2-D rotary is applied to q/k, and
the packed sequence attends within each ``cu_seqlens`` chunk (a single image =>
one full-attention chunk, no mask). MLP is SwiGLU with bias.

Precision: fp32 weights + fp32 activations, HiFi4 with fp32_dest_acc_en +
packer_l1_acc, and a manual fp32 softmax — the same recipe the graduated
`qwen2_v_l_for_conditional_generation` text stack uses. One block over a short
packed sequence is precision-robust, so no limb emulation is needed here.
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
BF16 = ttnn.bfloat16
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


class _VisionBlock:
    def __init__(self, device, block):
        self.device = device
        b = block.eval() if hasattr(block, "eval") else block
        self.norm1 = b.norm1
        self.norm2 = b.norm2
        a = b.attn
        self.qkv = a.qkv
        self.proj = a.proj
        self.num_heads = int(a.num_heads)
        self.head_dim = int(a.head_dim)
        self.dim = int(getattr(a, "dim", self.num_heads * self.head_dim))
        self.scaling = float(getattr(a, "scaling", self.head_dim**-0.5))
        mlp = b.mlp
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self._lin = {}
        self._rms = {}
        self._compute = None

    # ── helpers ──────────────────────────────────────────────────────────
    def _ck(self):
        if self._compute is None:
            self._compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
            )
        return self._compute

    def _to_ttnn(self, t, dtype=F32):
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
            bpar = tm.bias
            self._lin[key] = (
                ttnn.from_torch(
                    tm.weight.detach().to(torch.float32), dtype=F32, layout=TILE, device=self.device, memory_config=DRAM
                ),
                ttnn.from_torch(
                    bpar.detach().reshape(1, 1, -1).to(torch.float32),
                    dtype=F32,
                    layout=TILE,
                    device=self.device,
                    memory_config=DRAM,
                )
                if bpar is not None
                else None,
            )
        wt, bt = self._lin[key]
        return ttnn.linear(
            x, wt, bias=bt, transpose_b=True, memory_config=DRAM, dtype=F32, compute_kernel_config=self._ck()
        )

    def _rmsnorm(self, x, norm):
        # Qwen2_5_VLRMSNorm in fp32: variance + normalize in fp32, then * weight.
        eps = float(getattr(norm, "variance_epsilon", 1e-6))
        key = id(norm)
        if key not in self._rms:
            w = norm.weight.detach().reshape(1, 1, -1).to(torch.float32)
            self._rms[key] = ttnn.from_torch(w, dtype=F32, layout=TILE, device=self.device, memory_config=DRAM)
        w = self._rms[key]
        xf = x if x.dtype == F32 else ttnn.typecast(x, F32)
        msq = ttnn.mean(ttnn.mul(xf, xf), dim=-1, keepdim=True, compute_kernel_config=self._ck())
        normed = ttnn.mul(xf, ttnn.rsqrt(ttnn.add(msq, eps)))
        return ttnn.mul(normed, w)

    def _split_heads(self, t, S):
        # [1,S,dim] -> [1,heads,S,head_dim]
        t = ttnn.to_layout(t, RM)
        t = ttnn.reshape(t, [1, S, self.num_heads, self.head_dim])
        t = ttnn.permute(t, [0, 2, 1, 3])
        return ttnn.to_layout(t, TILE)

    def _merge_heads(self, t, S):
        # [1,heads,S,head_dim] -> [1,S,dim]
        t = ttnn.to_layout(t, RM)
        t = ttnn.permute(t, [0, 2, 1, 3])
        t = ttnn.reshape(t, [1, S, self.dim])
        return ttnn.to_layout(t, TILE)

    def _rotate_half(self, x):
        d = self.head_dim
        h = d // 2
        heads = x.shape[1]
        S = x.shape[2]
        x1 = ttnn.slice(x, [0, 0, 0, 0], [1, heads, S, h], [1, 1, 1, 1])
        x2 = ttnn.slice(x, [0, 0, 0, h], [1, heads, S, d], [1, 1, 1, 1])
        return ttnn.concat([ttnn.multiply(x2, -1.0), x1], dim=-1)

    def _rope(self, x, cos, sin):
        return ttnn.add(ttnn.mul(x, cos), ttnn.mul(self._rotate_half(x), sin))

    def _softmax(self, scores):
        # Manual fp32 softmax over the last dim (reference uses dtype=fp32).
        m = ttnn.max(scores, dim=-1, keepdim=True)
        e = ttnn.exp(ttnn.subtract(scores, m), fast_and_approximate_mode=False)
        s = ttnn.sum(e, dim=-1, keepdim=True)
        return ttnn.multiply(e, ttnn.reciprocal(s))

    # ── host preprocessing (parameter-free) ──────────────────────────────
    def _rope_tables(self, cos, sin, S):
        # cos/sin come as [S, head_dim]; broadcast over heads -> [1,1,S,head_dim].
        c = cos if isinstance(cos, torch.Tensor) else torch.as_tensor(cos)
        s = sin if isinstance(sin, torch.Tensor) else torch.as_tensor(sin)
        c = c.reshape(1, 1, S, self.head_dim).to(torch.float32)
        s = s.reshape(1, 1, S, self.head_dim).to(torch.float32)
        return self._to_ttnn(c), self._to_ttnn(s)

    def _chunk_mask(self, cu_seqlens, S):
        # Additive 0/-1e9 mask [1,1,S,S] that is 0 within each cu_seqlens chunk
        # and -1e9 across chunks. The reference processes each chunk separately;
        # a block-diagonal mask over the full sequence is numerically equivalent.
        # For a single image (cu_seqlens=[0,S]) this is all-zeros (full attention).
        add = torch.full((S, S), -1e9, dtype=torch.float32)
        if cu_seqlens is not None:
            cu = cu_seqlens.reshape(-1).to(torch.long).tolist()
        else:
            cu = [0, S]
        for i in range(len(cu) - 1):
            a, b = int(cu[i]), int(cu[i + 1])
            add[a:b, a:b] = 0.0
        return self._to_ttnn(add.reshape(1, 1, S, S))

    # ── core block (prebuilt cos/sin/mask) — reused by the vision tower ───
    def _apply(self, x, cos, sin, mask, S):
        """One vision block on x[1,S,dim] with prebuilt ttnn cos/sin[1,1,S,hd]
        and additive mask[1,1,S,S]. Returns [1,S,dim]. Shared with the full
        `qwen2_vision_transformer_pretrained_model` port so the 32-block stack
        drives this exact math per block (varying only the mask)."""
        # ── attention ──
        residual = x
        n = self._rmsnorm(x, self.norm1)
        qkv = self._linear(n, self.qkv)  # [1,S,3*dim]
        # 3*dim axis splits as [q(dim) | k(dim) | v(dim)] (reshape [S,3,heads,hd]).
        q = ttnn.slice(qkv, [0, 0, 0], [1, S, self.dim], [1, 1, 1])
        k = ttnn.slice(qkv, [0, 0, self.dim], [1, S, 2 * self.dim], [1, 1, 1])
        v = ttnn.slice(qkv, [0, 0, 2 * self.dim], [1, S, 3 * self.dim], [1, 1, 1])
        q = self._split_heads(q, S)  # [1,heads,S,hd]
        k = self._split_heads(k, S)
        v = self._split_heads(v, S)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)
        scores = ttnn.matmul(q, k, transpose_b=True, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)
        scores = ttnn.multiply(scores, self.scaling)
        scores = ttnn.add(scores, mask)
        probs = self._softmax(scores)
        out = ttnn.matmul(probs, v, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)
        out = self._merge_heads(out, S)  # [1,S,dim]
        x = ttnn.add(residual, self._linear(out, self.proj))

        # ── SwiGLU MLP ──
        residual = x
        n = self._rmsnorm(x, self.norm2)
        gate = ttnn.silu(self._linear(n, self.gate_proj))
        up = self._linear(n, self.up_proj)
        x = ttnn.add(residual, self._linear(ttnn.mul(gate, up), self.down_proj))
        return x

    # ── forward ──────────────────────────────────────────────────────────
    def __call__(self, hidden_states=None, cu_seqlens=None, position_embeddings=None, **_ignored):
        if hidden_states is None:
            raise ValueError("qwen2_v_l_vision_block stub requires `hidden_states`")
        if position_embeddings is None:
            raise ValueError("qwen2_v_l_vision_block stub requires `position_embeddings` (cos, sin)")
        cos_t, sin_t = position_embeddings

        x = self._to_ttnn(hidden_states, dtype=F32)  # [seq, dim] fp32
        S = int(x.shape[-2])
        x = ttnn.reshape(x, [1, S, self.dim])  # add batch dim

        cos, sin = self._rope_tables(cos_t, sin_t, S)
        mask = self._chunk_mask(cu_seqlens, S)

        x = self._apply(x, cos, sin, mask, S)
        return ttnn.reshape(x, [S, self.dim])  # [seq, dim] (matches golden)


def build(device, torch_module):
    """PCC-harness entry point: native TTNN Qwen2.5-VL vision block."""
    return _VisionBlock(device, torch_module)
