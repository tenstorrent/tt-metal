# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hunyuan_video15_token_refiner` of tencent/HunyuanVideo-1.5.

Reference submodule: `context_embedder`, a `HunyuanVideo15TokenRefiner`:

    pooled_projections = hidden_states.mean(dim=1)              # (B, in_channels)  [no-mask path]
    temb          = self.time_text_embed(timestep, pooled)     # CombinedTimestepTextProjEmbeddings -> (B, D)
    hidden_states = self.proj_in(hidden_states)                # Linear(in_channels, D)
    hidden_states = self.token_refiner(hidden_states, temb)    # stack of refiner blocks
    return hidden_states

Inputs at test time:
    hidden_states : (B, L, in_channels) — PRIMARY (arrives as a ttnn tensor)
    timestep      : (N,) 1-D            — bf16-exact (arrives as a torch tensor)
    attention_mask: None                — the per-component test drives the no-mask path

Native ttnn strategy
--------------------
Composition of already-validated pieces, all native: `ttnn.mean` pooling; the
time_text_embed = sinusoid (host-freq constant + device matmul/cos/sin) + two
MLPs summed; `proj_in` matmul+bias; then the individual-token-refiner stack
(per-block: affine LayerNorm, MHSA `softmax(qkᵀ·scale)v`, AdaNorm gates, gated
residuals, SiLU FeedForward). Float32 math with a HiFi4 config.
"""

from __future__ import annotations

import math

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"

_MAX_PERIOD = 10000


def _activation_fn(mod):
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


def build(device, torch_module):
    """Extract every sub-weight of the token refiner; return a native ttnn forward."""
    import torch

    m = torch_module
    tte = m.time_text_embed
    proj_in = m.proj_in
    refiner = m.token_refiner

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def lin(linear):
        w = f32(linear.weight.detach().t())
        b = f32(linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None
        return w, b

    def norm_wbe(norm):
        w = f32(norm.weight.detach().reshape(1, 1, -1)) if getattr(norm, "weight", None) is not None else None
        b = f32(norm.bias.detach().reshape(1, 1, -1)) if getattr(norm, "bias", None) is not None else None
        return w, b, float(getattr(norm, "eps", 1e-6))

    # --- time_text_embed (CombinedTimestepTextProjEmbeddings) ---
    tp = tte.time_proj
    half_dim = int(getattr(tp, "num_channels", 256)) // 2
    tte_flip = bool(getattr(tp, "flip_sin_to_cos", True))
    tte_shift = float(getattr(tp, "downscale_freq_shift", 0.0))
    tte_scale = float(getattr(tp, "scale", 1.0))
    exponent = -math.log(_MAX_PERIOD) * torch.arange(0, half_dim, dtype=torch.float32) / (half_dim - tte_shift)
    freq_row = f32((torch.exp(exponent) * tte_scale).reshape(1, half_dim))
    te_w1, te_b1 = lin(tte.timestep_embedder.linear_1)
    te_w2, te_b2 = lin(tte.timestep_embedder.linear_2)
    tx_w1, tx_b1 = lin(tte.text_embedder.linear_1)
    tx_w2, tx_b2 = lin(tte.text_embedder.linear_2)

    pin_w, pin_b = lin(proj_in)

    # --- refiner blocks ---
    blocks = []
    for blk in refiner.refiner_blocks:
        attn = blk.attn
        heads = int(attn.heads)
        inner = int(attn.to_q.out_features)
        dim_head = inner // heads
        scale = float(getattr(attn, "scale", dim_head**-0.5))
        # Fused QKV: one (C -> 3*inner) matmul + slice instead of 3 launches
        # (launch-bound path; a slice is far cheaper than a matmul launch).
        _qw = [attn.to_q, attn.to_k, attn.to_v]
        w_qkv = f32(torch.cat([m.weight.detach() for m in _qw], dim=0).t())
        _hasb = all(m.bias is not None for m in _qw)
        b_qkv = f32(torch.cat([m.bias.detach() for m in _qw], dim=0).reshape(1, -1)) if _hasb else None
        wo, bo = lin(attn.to_out[0])

        # Fused AdaLN gate: keep the (C -> 2*half) linear whole + slice into msa/mlp.
        ada_lin = blk.norm_out.linear
        half = int(ada_lin.out_features) // 2
        aw = ada_lin.weight.detach()
        ab = ada_lin.bias.detach() if ada_lin.bias is not None else None
        w_ada = f32(aw.t())
        b_ada = f32(ab.reshape(1, -1)) if ab is not None else None

        ffnet = blk.ff.net
        act_wrapper = ffnet[0]
        if type(act_wrapper).__name__ == "GELU":
            ff_proj = act_wrapper.proj
            approx = str(getattr(act_wrapper, "approximate", "none"))
            variant = ttnn.GeluVariant.Tanh if approx == "tanh" else ttnn.GeluVariant.Accurate
            ff_act = lambda t, _v=variant: ttnn.gelu(t, variant=_v)
        else:
            ff_proj = getattr(act_wrapper, "proj", act_wrapper)
            ff_act = _activation_fn(getattr(act_wrapper, "activation", act_wrapper))
        ff_lin2 = None
        for module in reversed(list(ffnet)):
            if isinstance(module, torch.nn.Linear):
                ff_lin2 = module
                break
        ff_w1, ff_b1 = lin(ff_proj)
        ff_w2, ff_b2 = lin(ff_lin2)

        blocks.append(
            dict(
                n1=norm_wbe(blk.norm1),
                n2=norm_wbe(blk.norm2),
                w_qkv=w_qkv,
                b_qkv=b_qkv,
                wo=wo,
                bo=bo,
                heads=heads,
                dim_head=dim_head,
                inner=inner,
                scale=scale,
                w_ada=w_ada,
                b_ada=b_ada,
                half=half,
                ff_w1=ff_w1,
                ff_b1=ff_b1,
                ff_act=ff_act,
                ff_w2=ff_w2,
                ff_b2=ff_b2,
            )
        )

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def _linear(x, w, b):
        # Fold bias into the matmul epilogue (launch-bound path: fewer op launches).
        if b is not None:
            return ttnn.linear(x, w, bias=b, compute_kernel_config=compute_config)
        return ttnn.matmul(x, w, compute_kernel_config=compute_config)

    def _ln(x, wbe):
        w, b, eps = wbe
        return ttnn.layer_norm(x, epsilon=eps, weight=w, bias=b, compute_kernel_config=compute_config)

    def _time_text_embed(timestep, pooled):
        ts = _to_f32_device(timestep)
        n = 1
        for d in ts.shape:
            n *= int(d)
        ts = ttnn.reshape(ts, (n, 1))
        a = ttnn.matmul(ts, freq_row, compute_kernel_config=compute_config)
        proj = ttnn.concat([ttnn.cos(a), ttnn.sin(a)] if tte_flip else [ttnn.sin(a), ttnn.cos(a)], dim=-1)
        temb = _linear(proj, te_w1, te_b1)
        temb = ttnn.silu(temb)
        temb = _linear(temb, te_w2, te_b2)
        ptxt = _linear(pooled, tx_w1, tx_b1)
        ptxt = ttnn.silu(ptxt)
        ptxt = _linear(ptxt, tx_w2, tx_b2)
        return ttnn.add(temb, ptxt)

    def _attention(h, blk):
        B = int(h.shape[0])
        L = int(h.shape[1])
        heads, dh, inner, scale = blk["heads"], blk["dim_head"], blk["inner"], blk["scale"]
        qkv = _linear(h, blk["w_qkv"], blk["b_qkv"])  # (B, L, 3*inner), one launch
        q = ttnn.slice(qkv, (0, 0, 0), (B, L, inner))
        k = ttnn.slice(qkv, (0, 0, inner), (B, L, 2 * inner))
        v = ttnn.slice(qkv, (0, 0, 2 * inner), (B, L, 3 * inner))

        def split(t):
            t = ttnn.reshape(t, (B, L, heads, dh))
            return ttnn.permute(t, (0, 2, 1, 3))

        q, k, v = split(q), split(k), split(v)
        kt = ttnn.permute(k, (0, 1, 3, 2))
        scores = ttnn.multiply(ttnn.matmul(q, kt, compute_kernel_config=compute_config), scale)
        attn_w = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(attn_w, v, compute_kernel_config=compute_config)
        out = ttnn.permute(out, (0, 2, 1, 3))
        out = ttnn.reshape(out, (B, L, inner))
        return _linear(out, blk["wo"], blk["bo"])

    def _ada_gate(temb, blk):
        s = ttnn.silu(temb)
        half = blk["half"]
        g = _linear(s, blk["w_ada"], blk["b_ada"])  # (B, 2*half), one launch
        bdim = int(g.shape[0])
        gm = ttnn.slice(g, (0, 0), (bdim, half))
        gp = ttnn.slice(g, (0, half), (bdim, 2 * half))
        return ttnn.reshape(gm, (bdim, 1, half)), ttnn.reshape(gp, (bdim, 1, half))

    def _ff(x, blk):
        y = _linear(x, blk["ff_w1"], blk["ff_b1"])
        y = blk["ff_act"](y)
        y = _linear(y, blk["ff_w2"], blk["ff_b2"])
        return y

    def forward(hidden_states, timestep=None, attention_mask=None, *args, **kwargs):
        if timestep is None:
            timestep = kwargs.get("timestep", args[0] if args else None)
        if timestep is None:
            raise TypeError("hunyuan_video15_token_refiner needs `timestep`")

        x = _to_f32_device(hidden_states)  # (B, L, in_channels)
        pooled = ttnn.mean(x, dim=1)  # (B, in_channels) [no-mask path]
        if len(pooled.shape) == 3:  # squeeze a kept dim if any
            pooled = ttnn.reshape(pooled, (int(pooled.shape[0]), int(pooled.shape[-1])))

        temb = _time_text_embed(timestep, pooled)  # (B, D)
        h = _linear(x, pin_w, pin_b)  # (B, L, D)

        for blk in blocks:
            norm_h = _ln(h, blk["n1"])
            attn_out = _attention(norm_h, blk)
            gate_msa, gate_mlp = _ada_gate(temb, blk)
            h = ttnn.add(h, ttnn.multiply(attn_out, gate_msa))
            ff_out = _ff(_ln(h, blk["n2"]), blk)
            h = ttnn.add(h, ttnn.multiply(ff_out, gate_mlp))
        return h

    return forward


def hunyuan_video15_token_refiner(*args, **kwargs):
    raise RuntimeError(
        "hunyuan_video15_token_refiner requires build(device, torch_module) to bind the "
        "sub-weights; the bare callable has no parameters."
    )
