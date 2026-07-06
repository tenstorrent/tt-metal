# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `qwen2_v_l_decoder_layer`
(transformers ``Qwen2_5_VLDecoderLayer``) — one layer of the LongCat-Image
text encoder (Qwen2.5-VL 7B language model).

forward(hidden_states[1,S,3584], attention_mask[1,1,S,S] bool,
        position_embeddings=(cos,sin) each [3,1,S,128])::

    r = hidden_states
    h = RMSNorm(hidden_states)                 # input_layernorm
    h = self_attn(h, mask, (cos,sin))          # GQA + M-RoPE
    hidden_states = r + h
    r = hidden_states
    h = RMSNorm(hidden_states)                 # post_attention_layernorm
    h = mlp(h)                                 # SwiGLU: down(silu(gate(h)) * up(h))
    hidden_states = r + h
    return hidden_states

self_attn (GQA, num_heads=28, num_kv_heads=4 -> n_rep=7, head_dim=128)::

    q,k,v = q/k/v_proj(h)  (biased)  -> [1,heads,S,128]
    q,k   = M-RoPE(q,k, cos,sin)     # rotate_half style (NOT interleaved)
    k,v   = repeat_kv(k,v, 7)
    scores= q @ k^T * head_dim**-0.5 + additive_mask
    out   = softmax(scores) @ v      -> merge heads -> o_proj (no bias)

M-RoPE assembly: cos/sin arrive as 3 sections [temporal,height,width]; the
per-channel selection (``mrope_section*2`` split, pick section i%3) is a
deterministic reshape of the *input* rope tables, so it is precomputed on host
into a single [1,1,S,128] cos/sin — like the per-block RoPE tables the harness
builds. rotate_half(x) = cat(-x[64:], x[:64]). The bool causal mask is turned
into an additive 0/-1e9 mask on host. Precision: fully fp32 (HiFi4 / fp32-acc);
RMSNorm is manual fp32; this is one 7 B-model layer (~0.9 GB) so fp32 fits.
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


class _DecoderLayer:
    def __init__(self, device, layer):
        self.device = device
        self.blk = layer.eval() if hasattr(layer, "eval") else layer
        a = self.blk.self_attn
        self.num_heads = int(a.num_heads)
        self.num_kv_heads = int(a.num_key_value_heads)
        self.head_dim = int(a.head_dim)
        self.n_rep = self.num_heads // self.num_kv_heads
        self.hidden = self.num_heads * self.head_dim
        self.scaling = float(getattr(a, "scaling", self.head_dim**-0.5))
        cfg = a.config
        rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        self.mrope_section = list(rp.get("mrope_section", [16, 24, 24]))
        self.attn_eps = float(getattr(self.blk.input_layernorm, "variance_epsilon", 1e-6))
        self.mlp_eps = float(getattr(self.blk.post_attention_layernorm, "variance_epsilon", 1e-6))
        self._lin = {}
        self._rms = {}
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

    def _rmsnorm(self, x, norm, eps):
        # RMSNorm over last dim (hidden) with affine weight.
        key = id(norm)
        if key not in self._rms:
            w = norm.weight.detach().reshape(1, 1, -1).to(torch.float32)
            self._rms[key] = ttnn.from_torch(w, dtype=F32, layout=TILE, device=self.device, memory_config=DRAM)
        w = self._rms[key]
        msq = ttnn.mean(ttnn.mul(x, x), dim=2, keepdim=True, compute_kernel_config=self._ck())
        return ttnn.mul(ttnn.mul(x, ttnn.rsqrt(ttnn.add(msq, eps))), w)

    def _split_heads(self, t, S, heads):
        # [1,S,heads*head_dim] -> [1,heads,S,head_dim]
        t = ttnn.to_layout(t, RM)
        t = ttnn.reshape(t, [1, S, heads, self.head_dim])
        t = ttnn.permute(t, [0, 2, 1, 3])
        return ttnn.to_layout(t, TILE)

    def _merge_heads(self, t, S):
        # [1,heads,S,head_dim] -> [1,S,heads*head_dim]
        t = ttnn.to_layout(t, RM)
        t = ttnn.permute(t, [0, 2, 1, 3])
        t = ttnn.reshape(t, [1, S, self.hidden])
        return ttnn.to_layout(t, TILE)

    def _rotate_half(self, x):
        # x [1,heads,S,head_dim]; return cat(-x[..,d/2:], x[..,:d/2]).
        d = self.head_dim
        h = d // 2
        heads = x.shape[1]
        S = x.shape[2]
        x1 = ttnn.slice(x, [0, 0, 0, 0], [1, heads, S, h], [1, 1, 1, 1])
        x2 = ttnn.slice(x, [0, 0, 0, h], [1, heads, S, d], [1, 1, 1, 1])
        return ttnn.concat([ttnn.multiply(x2, -1.0), x1], dim=-1)

    def _rope(self, x, cos, sin):
        return ttnn.add(ttnn.mul(x, cos), ttnn.mul(self._rotate_half(x), sin))

    def _repeat_kv(self, t):
        # [1,kv_heads,S,head_dim] -> [1,kv_heads*n_rep,S,head_dim] (repeat_interleave dim=1)
        if self.n_rep == 1:
            return t
        return ttnn.repeat_interleave(t, self.n_rep, dim=1)

    # ── M-RoPE / mask host preprocessing ─────────────────────────────────
    def _mrope_tables(self, position_embeddings, S):
        cos, sin = position_embeddings
        doubled = self.mrope_section * 2

        def assemble(t):
            t = t.float()  # [3,1,S,head_dim]
            chunks = torch.split(t, doubled, dim=-1)  # list of [3,1,S,size]
            picked = [chunks[i][i % 3] for i in range(len(chunks))]  # each [1,S,size]
            return torch.cat(picked, dim=-1).unsqueeze(1)  # [1,1,S,head_dim]

        c = self._to_ttnn(assemble(cos))
        s = self._to_ttnn(assemble(sin))
        return c, s

    def _additive_mask(self, attention_mask, S):
        if attention_mask is None:
            m = torch.zeros(1, 1, S, S, dtype=torch.float32)
            causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
            m.masked_fill_(causal, -1e9)
            return self._to_ttnn(m)
        am = attention_mask
        if am.dtype == torch.bool:
            add = torch.zeros(am.shape, dtype=torch.float32)
            add.masked_fill_(~am, -1e9)
        else:
            add = am.to(torch.float32)
        return self._to_ttnn(add)

    # ── forward ──────────────────────────────────────────────────────────
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **_ignored,
    ):
        x = self._to_ttnn(hidden_states)  # [1,S,hidden]
        S = x.shape[1]

        # ---- self attention ----
        residual = x
        h = self._rmsnorm(x, self.blk.input_layernorm, self.attn_eps)
        a = self.blk.self_attn
        q = self._split_heads(self._linear(h, a.q_proj), S, self.num_heads)  # [1,28,S,128]
        k = self._split_heads(self._linear(h, a.k_proj), S, self.num_kv_heads)  # [1,4,S,128]
        v = self._split_heads(self._linear(h, a.v_proj), S, self.num_kv_heads)

        if position_embeddings is not None:
            cos, sin = self._mrope_tables(position_embeddings, S)
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)

        k = self._repeat_kv(k)  # [1,28,S,128]
        v = self._repeat_kv(v)

        scores = ttnn.matmul(q, k, transpose_b=True, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)
        scores = ttnn.multiply(scores, self.scaling)
        scores = ttnn.add(scores, self._additive_mask(attention_mask, S))  # broadcast over heads
        probs = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(probs, v, dtype=F32, compute_kernel_config=self._ck(), memory_config=DRAM)  # [1,28,S,128]
        out = self._merge_heads(out, S)  # [1,S,hidden]
        attn = self._linear(out, a.o_proj)  # o_proj (no bias)
        x = ttnn.add(residual, attn)

        # ---- mlp (SwiGLU) ----
        residual = x
        h = self._rmsnorm(x, self.blk.post_attention_layernorm, self.mlp_eps)
        mlp = self.blk.mlp
        gate = ttnn.silu(self._linear(h, mlp.gate_proj))
        up = self._linear(h, mlp.up_proj)
        h = self._linear(ttnn.mul(gate, up), mlp.down_proj)
        x = ttnn.add(residual, h)
        return x


def build(device, torch_module):
    """PCC-harness entry point: native TTNN Qwen2_5_VLDecoderLayer."""
    return _DecoderLayer(device, torch_module)
