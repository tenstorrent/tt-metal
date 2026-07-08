# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hunyuan_video15_individual_token_refiner` of tencent/HunyuanVideo-1.5.

Reference submodule: `context_embedder.token_refiner`, a
`HunyuanVideo15IndividualTokenRefiner` = a stack of
`HunyuanVideo15IndividualTokenRefinerBlock`s:

    for block in refiner_blocks:
        hidden_states = block(hidden_states, temb, self_attn_mask)

Each block:
    norm_hidden   = norm1(hidden_states)                 # LayerNorm, affine
    attn_output   = attn(norm_hidden)                    # MHSA (AttnProcessor2_0)
    gate_msa, gate_mlp = norm_out(temb)                  # HunyuanVideo15AdaNorm -> (B,1,C)
    hidden_states = hidden_states + attn_output * gate_msa
    ff_output     = ff(norm2(hidden_states))             # LayerNorm + FeedForward(silu)
    hidden_states = hidden_states + ff_output * gate_mlp

Inputs at test time:
    hidden_states : (B, L, C) — PRIMARY (arrives as a ttnn tensor)
    temb          : (B, C)    — per-batch conditioning (arrives as a torch tensor)
    attention_mask: None      — the per-component test drives the no-mask path

Native ttnn strategy
--------------------
Standard multi-head self-attention done natively: q/k/v = `matmul + bias`, split
into (B, H, L, Dh), `softmax(q @ kᵀ * scale) @ v`, merge heads, `to_out[0]`.
LayerNorms are affine `ttnn.layer_norm`; the AdaNorm gate is `silu -> matmul`
split into the two halves; the FeedForward is `matmul + bias -> act -> matmul +
bias` (activation detected from net[0]). Float32 math with a HiFi4 config.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"


def _make_f32(device):
    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    return f32


def _prep_linear(linear, f32):
    w = f32(linear.weight.detach().t())
    b = f32(linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None
    return w, b


def _prep_norm(norm, f32):
    w = f32(norm.weight.detach().reshape(1, 1, -1)) if getattr(norm, "weight", None) is not None else None
    b = f32(norm.bias.detach().reshape(1, 1, -1)) if getattr(norm, "bias", None) is not None else None
    eps = float(getattr(norm, "eps", 1e-6))
    return w, b, eps


def _activation_fn(mod):
    """Map a torch/diffusers activation module to a ttnn callable."""
    name = type(mod).__name__.lower()
    if "silu" in name or "swish" in name:
        return lambda t: ttnn.silu(t)
    if "gelu" in name:
        approx = str(getattr(mod, "approximate", "none"))
        variant = ttnn.GeluVariant.Tanh if approx == "tanh" else ttnn.GeluVariant.Accurate
        return lambda t: ttnn.gelu(t, variant=variant)
    if "relu" in name:
        return lambda t: ttnn.relu(t)
    if "mish" in name:
        return lambda t: ttnn.mish(t)
    return lambda t: t


def _prep_ff(ff, f32):
    """Extract (w1, b1, act, w2, b2) from a diffusers FeedForward."""
    import torch

    net = ff.net
    act_wrapper = net[0]
    if type(act_wrapper).__name__ == "GELU":
        proj = act_wrapper.proj
        approx = str(getattr(act_wrapper, "approximate", "none"))
        variant = ttnn.GeluVariant.Tanh if approx == "tanh" else ttnn.GeluVariant.Accurate
        act = lambda t: ttnn.gelu(t, variant=variant)
    else:
        proj = getattr(act_wrapper, "proj", act_wrapper)
        act = _activation_fn(getattr(act_wrapper, "activation", act_wrapper))
    lin2 = None
    for module in reversed(list(net)):
        if isinstance(module, torch.nn.Linear):
            lin2 = module
            break
    w1, b1 = _prep_linear(proj, f32)
    w2, b2 = _prep_linear(lin2, f32)
    return w1, b1, act, w2, b2


def build(device, torch_module):
    """Extract every refiner block's weights and return a native ttnn forward."""

    f32 = _make_f32(device)
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    blocks = []
    for blk in torch_module.refiner_blocks:
        attn = blk.attn
        heads = int(attn.heads)
        inner = int(attn.to_q.out_features)
        dim_head = inner // heads
        scale = float(getattr(attn, "scale", dim_head**-0.5))

        wq, bq = _prep_linear(attn.to_q, f32)
        wk, bk = _prep_linear(attn.to_k, f32)
        wv, bv = _prep_linear(attn.to_v, f32)
        wo, bo = _prep_linear(attn.to_out[0], f32)

        # HunyuanVideo15AdaNorm norm_out: silu -> linear(C, 2C) -> chunk(2, dim=1).
        ada_lin = blk.norm_out.linear
        half = int(ada_lin.out_features) // 2
        aw = ada_lin.weight.detach()
        ab = ada_lin.bias.detach() if ada_lin.bias is not None else None
        w_msa = f32(aw[:half, :].t())
        w_mlp = f32(aw[half:, :].t())
        b_msa = f32(ab[:half].reshape(1, half)) if ab is not None else None
        b_mlp = f32(ab[half:].reshape(1, half)) if ab is not None else None

        blocks.append(
            dict(
                n1=_prep_norm(blk.norm1, f32),
                n2=_prep_norm(blk.norm2, f32),
                wq=wq,
                bq=bq,
                wk=wk,
                bk=bk,
                wv=wv,
                bv=bv,
                wo=wo,
                bo=bo,
                heads=heads,
                dim_head=dim_head,
                inner=inner,
                scale=scale,
                w_msa=w_msa,
                b_msa=b_msa,
                w_mlp=w_mlp,
                b_mlp=b_mlp,
                half=half,
                ff=_prep_ff(blk.ff, f32),
            )
        )

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def _linear(x, w, b):
        y = ttnn.matmul(x, w, compute_kernel_config=compute_config)
        if b is not None:
            y = ttnn.add(y, b)
        return y

    def _layer_norm(x, wbe):
        w, b, eps = wbe
        return ttnn.layer_norm(x, epsilon=eps, weight=w, bias=b, compute_kernel_config=compute_config)

    def _attention(h, blk):
        B = int(h.shape[0])
        L = int(h.shape[1])
        heads, dh, inner, scale = blk["heads"], blk["dim_head"], blk["inner"], blk["scale"]

        q = _linear(h, blk["wq"], blk["bq"])
        k = _linear(h, blk["wk"], blk["bk"])
        v = _linear(h, blk["wv"], blk["bv"])

        def split_heads(t):
            t = ttnn.reshape(t, (B, L, heads, dh))
            return ttnn.permute(t, (0, 2, 1, 3))  # (B, H, L, Dh)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        kt = ttnn.permute(k, (0, 1, 3, 2))  # (B, H, Dh, L)
        scores = ttnn.matmul(q, kt, compute_kernel_config=compute_config)  # (B, H, L, L)
        scores = ttnn.multiply(scores, scale)
        attn = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(attn, v, compute_kernel_config=compute_config)  # (B, H, L, Dh)

        out = ttnn.permute(out, (0, 2, 1, 3))  # (B, L, H, Dh)
        out = ttnn.reshape(out, (B, L, inner))
        return _linear(out, blk["wo"], blk["bo"])

    def _ada_gate(temb, blk):
        s = ttnn.silu(temb)
        gate_msa = _linear(s, blk["w_msa"], blk["b_msa"])
        gate_mlp = _linear(s, blk["w_mlp"], blk["b_mlp"])
        bdim = int(gate_msa.shape[0])
        half = blk["half"]
        gate_msa = ttnn.reshape(gate_msa, (bdim, 1, half))
        gate_mlp = ttnn.reshape(gate_mlp, (bdim, 1, half))
        return gate_msa, gate_mlp

    def _feed_forward(x, ff):
        w1, b1, act, w2, b2 = ff
        h = _linear(x, w1, b1)
        h = act(h)
        h = _linear(h, w2, b2)
        return h

    def forward(hidden_states, temb=None, attention_mask=None, *args, **kwargs):
        if temb is None:
            temb = kwargs.get("temb", args[0] if args else None)
        if temb is None:
            raise TypeError("hunyuan_video15_individual_token_refiner needs `temb`")

        x = _to_f32_device(hidden_states)
        t = _to_f32_device(temb)

        for blk in blocks:
            norm_h = _layer_norm(x, blk["n1"])
            attn_out = _attention(norm_h, blk)
            gate_msa, gate_mlp = _ada_gate(t, blk)
            x = ttnn.add(x, ttnn.multiply(attn_out, gate_msa))
            ff_out = _feed_forward(_layer_norm(x, blk["n2"]), blk["ff"])
            x = ttnn.add(x, ttnn.multiply(ff_out, gate_mlp))

        return x

    return forward


def hunyuan_video15_individual_token_refiner(*args, **kwargs):
    raise RuntimeError(
        "hunyuan_video15_individual_token_refiner requires build(device, torch_module) "
        "to bind the block weights; the bare callable has no parameters."
    )
