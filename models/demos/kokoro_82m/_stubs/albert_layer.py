# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `albert_layer` (hexgrad/Kokoro-82M
`bert.encoder.albert_layer_groups.0.albert_layers.0`, a HF `AlbertLayer`).

Reference torch forward (attention_mask is None here):

    # AlbertAttention (self-attention + residual + LayerNorm):
    q,k,v = query/key/value(h).view(B, T, H, Dh).transpose(1, 2)
    scores = softmax(q @ kᵀ * scaling)              # scaling = Dh**-0.5
    ctx    = (scores @ v).reshape(B, T, H*Dh)
    attn   = attention.LayerNorm(h + dense(ctx))
    # AlbertLayer feed-forward:
    ffn    = ffn_output(gelu_new(ffn(attn)))
    out    = full_layer_layer_norm(ffn + attn)

`gelu_new` is the tanh GELU approximation. All linears are `ttnn.linear`; the
two residual LayerNorms are computed over the last (hidden) axis. Everything
runs natively in float32 (HiFi4 matmuls) for a clean PCC.
"""

from __future__ import annotations

import math

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind all layer params and return a native ttnn forward closure."""

    m = torch_module
    att = m.attention
    n_heads = int(att.num_attention_heads)
    head_dim = int(att.attention_head_size)
    scaling = float(att.scaling)
    ln_eps = float(getattr(att.LayerNorm, "eps", 1e-12))
    full_eps = float(getattr(m.full_layer_layer_norm, "eps", 1e-12))

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def _lin_w(lin):
        return ttnn.from_torch(
            lin.weight.detach().t().contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_DRAM,
        )

    def _lin_b(lin):
        if lin.bias is None:
            return None
        return ttnn.from_torch(
            lin.bias.detach().reshape(1, -1).contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_DRAM,
        )

    qw, qb = _lin_w(att.query), _lin_b(att.query)
    kw, kb = _lin_w(att.key), _lin_b(att.key)
    vw, vb = _lin_w(att.value), _lin_b(att.value)
    dw, db = _lin_w(att.dense), _lin_b(att.dense)
    fw, fb = _lin_w(m.ffn), _lin_b(m.ffn)
    ow, ob = _lin_w(m.ffn_output), _lin_b(m.ffn_output)

    def _ln_wb(ln):
        w = ttnn.from_torch(
            ln.weight.detach().reshape(1, 1, -1).contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_DRAM,
        )
        b = ttnn.from_torch(
            ln.bias.detach().reshape(1, 1, -1).contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_DRAM,
        )
        return w, b

    att_ln_w, att_ln_b = _ln_wb(att.LayerNorm)
    full_ln_w, full_ln_b = _ln_wb(m.full_layer_layer_norm)

    _gelu_c = math.sqrt(2.0 / math.pi)

    def _layernorm(x, w, b, eps):
        mean = ttnn.mean(x, dim=2, keepdim=True)
        xc = ttnn.subtract(x, mean)
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=2, keepdim=True)
        xn = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, eps)))
        return ttnn.add(ttnn.multiply(xn, w), b)

    def _gelu_new(x):
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x3 = ttnn.multiply(ttnn.multiply(x, x), x)
        inner = ttnn.multiply(ttnn.add(x, ttnn.multiply(x3, 0.044715)), _gelu_c)
        t = ttnn.tanh(inner)
        return ttnn.multiply(ttnn.multiply(x, 0.5), ttnn.add(t, 1.0))

    def _lin(x, w, b):
        return ttnn.linear(x, w, bias=b, compute_kernel_config=compute_config, memory_config=_DRAM)

    def forward(hidden_states, attention_mask=None, *args, **kwargs):
        h = hidden_states
        if not isinstance(h, ttnn.Tensor):
            h = ttnn.from_torch(
                h.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_DRAM
            )
        if h.get_dtype() != ttnn.float32:
            h = ttnn.typecast(h, ttnn.float32)

        B, T, _ = int(h.shape[0]), int(h.shape[1]), int(h.shape[2])

        q = _lin(h, qw, qb)  # [B, T, H*Dh]
        k = _lin(h, kw, kb)
        v = _lin(h, vw, vb)

        # [B, T, H*Dh] -> [B, H, T, Dh]
        def _split_heads(t):
            t = ttnn.reshape(t, (B, T, n_heads, head_dim))
            return ttnn.transpose(t, 1, 2)

        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)

        # scores = q @ kᵀ * scaling ; softmax ; @ v
        kt = ttnn.transpose(k, -2, -1)  # [B, H, Dh, T]
        scores = ttnn.matmul(q, kt, compute_kernel_config=compute_config, memory_config=_DRAM)
        scores = ttnn.multiply(scores, scaling)
        # ttnn.softmax runs at ~bf16 precision (~2.3% error), and attention scores
        # here reach |s|~30 so that error is amplified into the context vector and
        # accumulates across the 12 shared layers -> a noisy bert output whose
        # noise inflates the downstream F0Ntrain instance-norm variance (shrinking
        # F0) and drifts the NSF source phase. A manual fp32 softmax is ~30x more
        # accurate (rel ~7e-4).
        mx = ttnn.max(scores, dim=-1, keepdim=True)
        e = ttnn.exp(ttnn.subtract(scores, mx))
        probs = ttnn.multiply(e, ttnn.reciprocal(ttnn.sum(e, dim=-1, keepdim=True)))
        ctx = ttnn.matmul(probs, v, compute_kernel_config=compute_config, memory_config=_DRAM)  # [B, H, T, Dh]

        # [B, H, T, Dh] -> [B, T, H*Dh]
        ctx = ttnn.transpose(ctx, 1, 2)
        ctx = ttnn.reshape(ctx, (B, T, n_heads * head_dim))

        attn = _lin(ctx, dw, db)
        attn = _layernorm(ttnn.add(h, attn), att_ln_w, att_ln_b, ln_eps)

        ffn = _lin(attn, fw, fb)
        ffn = _gelu_new(ffn)
        ffn = _lin(ffn, ow, ob)

        out = _layernorm(ttnn.add(ffn, attn), full_ln_w, full_ln_b, full_eps)
        return out

    return forward


def albert_layer(*args, **kwargs):
    raise RuntimeError(
        "albert_layer requires build(device, torch_module) to bind trained "
        "weights; the bare callable has no parameters."
    )
