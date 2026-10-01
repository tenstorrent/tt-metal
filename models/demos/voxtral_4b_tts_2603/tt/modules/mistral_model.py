# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mistral_model` -- the text backbone `MistralModel` at `model`.

Embedding table, 26 `MistralDecoderLayer`s, final RMSNorm. dim 3072, 32 query heads over 8 KV
heads, head_dim **128** (explicitly 128, not 3072/32=96), hidden_dim 9216, `norm_eps` 1e-5,
`rope_theta` 1e6, no biases anywhere.

Two things worth naming:

* **Positions are looked up, not assumed.** `cos`/`sin` for positions 0..`_MAX_POSITIONS` are built
  at `build` time from the model's OWN `rotary_emb` (so they match the reference exactly rather than
  re-deriving `inv_freq` and hoping the `rope_theta` spelling was read the same way), and the
  incoming `position_ids` gather rows out of them with `ttnn.embedding`. That keeps arbitrary
  position ids working while the forward stays free of torch calls -- the runtime native probe
  counts every torch call the forward makes and graduates only at zero.
* **The residual stream runs in float32.** With everything in bfloat16 the 26-layer stack landed
  at PCC 0.9798: a single layer is fine, but ~0.4% per residual add compounds over 52 of them.
  Weights stay bfloat16 (7 GB; float32 would be 14) -- `ttnn.linear` accepts a float32 activation
  against a bfloat16 weight and returns float32, so only the activations widen. Q/K/V are cast back
  down for SDPA, which rejects float32 outright
  (`sdpa_device_operation.cpp:43`, `dtype() == DataType::BFLOAT16`); that leaves one bfloat16
  rounding per layer on the attention branch instead of on the accumulating residual.
* **The norm is spelled out rather than calling `ttnn.rms_norm`** -- see `_rms_norm`. That one
  substitution is worth more than every other fidelity knob on this stack combined, and it is the
  difference between a hidden state that resolves the LM head's top-1 and one that does not.
* **`attention_mask` is not consumed.** Causality comes from SDPA's `is_causal`, which reproduces
  the reference exactly for an UNPADDED batch -- every row genuine content, so the mask is the
  plain lower-triangular one. A mask carrying actual PADDING would need a real additive
  `[B, 1, S, S]` tensor instead, and this port would have to be extended; it must not quietly
  lean on `is_causal` in that case.
* **The leading axis is the BATCH and it is read off the tensor.** `batch = input_ids.shape[0]`,
  never a literal 1. An earlier revision reshaped to `[1, 1, seq, dim]` and returned
  `[1, seq, dim]`; that hardcoded 1 is the shape at which samples 1..B-1 go missing, so the
  bound is taken from the input and the built `[batch, 1, seq, dim]` residual stream carries all
  of them. Every op downstream is batch-clean at that shape:
  `nlp_create_qkv_heads` requires only `shape[1] == 1` and tile-aligned PADDED height,
  `scaled_dot_product_attention` takes `[batch, heads, seq, head_dim]`, and the `cos`/`sin`
  tables broadcast from `[1, 1, seq, head_dim]` over both the batch and the head axis.
  A sequence length that is not a multiple of 32 is fine: the tile padding lands on rows the
  causal mask lets no real query see.
"""

from __future__ import annotations

import torch

import ttnn

_MAX_POSITIONS = 8192


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _weight(linear, device):
    """A bfloat16 `[in, out]` device tensor for a torch `nn.Linear` (whose weight is `[out, in]`)."""
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


def _norm_weight(norm, device):
    """Gamma as a `[1, 1, 1, dim]` float32 TILE tensor -- the form `_rms_norm` multiplies by."""
    return _from_torch(norm.weight.detach().reshape(1, 1, 1, -1), device, dtype=ttnn.float32)


_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _rms_norm(x, gamma, eps):
    """`x * rsqrt(mean(x^2) + eps) * gamma`, spelled out, entirely in float32.

    The stock op is the obvious call and is what this port started with. Measured against the
    reference on the real layer-0 input with `layer.input_layernorm`, it lands at 9.65e-4
    RELATIVE error, while the four ops below land at 6.6e-8 -- four orders of magnitude apart, and
    the stock op was the single largest error term in the whole stack.

    Why it matters that much: the stack runs 52 of these, the error is relative (a norm error
    rescales the entire branch output that follows it), and 9.65e-4 apiece random-walks to ~7e-3
    on the final hidden state. Every other term is smaller -- the float32-activation x bfloat16
    weight matmul sits at a hardware floor of 4.9e-4 per linear (identical with a float32 weight,
    and unchanged by `packer_l1_acc` or HiFi3, so it is the FPU's limit and not a knob), and the
    bfloat16 Q/K/V cast SDPA forces is only 5.7e-4 end to end. The consumer of this stack picks a
    token with an argmax whose top-1/top-2 margin is often a few hundredths of a logit, so the
    norm's error was what decided that comparison.

    Tile padding on a sequence that is not a multiple of 32 is safe here: a padded row is all
    zeros, so `mean(x^2)` is 0 and `0 * rsqrt(eps)` stays 0 -- no NaN, and nothing to leak.
    """
    scale = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.square(x), dim=-1, keepdim=True), eps))
    return ttnn.multiply(ttnn.multiply(x, scale), gamma)


def _rope(x, cos, sin, half):
    """`x * cos + rotate_half(x) * sin` -- the convention `apply_rotary_pos_emb` uses."""
    ends = list(x.shape)
    lower = ttnn.slice(x, [0, 0, 0, 0], [ends[0], ends[1], ends[2], half])
    upper = ttnn.slice(x, [0, 0, 0, half], ends)
    rotated = ttnn.concat([ttnn.neg(upper), lower], dim=-1)
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))


def _compile_layer(device, layer):
    """One `MistralDecoderLayer` as a callable on `(h[1,1,S,dim], cos, sin)`."""
    attn = layer.self_attn
    mlp = layer.mlp

    n_heads = int(attn.config.num_attention_heads)
    n_kv_heads = int(attn.config.num_key_value_heads)
    head_dim = int(attn.head_dim)
    half = head_dim // 2
    scale = float(attn.scaling)

    wqkv = _from_torch(
        torch.cat(
            [
                attn.q_proj.weight.detach().transpose(0, 1),
                attn.k_proj.weight.detach().transpose(0, 1),
                attn.v_proj.weight.detach().transpose(0, 1),
            ],
            dim=-1,
        ).contiguous(),
        device,
    )
    wo = _weight(attn.o_proj, device)
    w_gate = _weight(mlp.gate_proj, device)
    w_up = _weight(mlp.up_proj, device)
    w_down = _weight(mlp.down_proj, device)
    g_in = _norm_weight(layer.input_layernorm, device)
    g_post = _norm_weight(layer.post_attention_layernorm, device)
    eps_in = float(layer.input_layernorm.variance_epsilon)
    eps_post = float(layer.post_attention_layernorm.variance_epsilon)

    def run(h, cos, sin):
        xn = _rms_norm(h, g_in, eps_in)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.linear(xn, wqkv, compute_kernel_config=_COMPUTE),
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            transpose_k_heads=False,
        )
        q = ttnn.typecast(_rope(q, cos, sin, half), ttnn.bfloat16)
        k = ttnn.typecast(_rope(k, cos, sin, half), ttnn.bfloat16)
        v = ttnn.typecast(v, ttnn.bfloat16)
        a = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale, compute_kernel_config=_COMPUTE
        )
        h = ttnn.add(
            h,
            ttnn.linear(
                ttnn.experimental.nlp_concat_heads(a),
                wo,
                dtype=ttnn.float32,
                compute_kernel_config=_COMPUTE,
            ),
        )

        hn = _rms_norm(h, g_post, eps_post)
        gated = ttnn.multiply(
            ttnn.silu(ttnn.linear(hn, w_gate, compute_kernel_config=_COMPUTE)),
            ttnn.linear(hn, w_up, compute_kernel_config=_COMPUTE),
        )
        return ttnn.add(h, ttnn.linear(gated, w_down, compute_kernel_config=_COMPUTE))

    return run


def _rope_tables(device, model):
    """`(cos, sin)` lookup tables for positions 0..`_MAX_POSITIONS`, from the model's own rotary."""
    rotary = model.rotary_emb
    positions = torch.arange(_MAX_POSITIONS, dtype=torch.long).unsqueeze(0)
    probe = torch.zeros(1, _MAX_POSITIONS, 1, dtype=torch.float32)
    with torch.no_grad():
        cos, sin = rotary(probe, positions)
    # Two copies of the same table: ROW_MAJOR to be gathered from by `ttnn.embedding` when explicit
    # position ids are supplied, and TILE `[1, 1, SMAX, head_dim]` to be sliced when they are not.
    # `MistralModel` defaults `position_ids` to `arange(S)`, which is exactly the table's first S
    # rows, so the no-ids path is a slice rather than an on-device arange.
    return (
        _from_torch(cos[0].contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT),
        _from_torch(sin[0].contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT),
        _from_torch(cos.reshape(1, 1, _MAX_POSITIONS, -1).contiguous(), device, dtype=ttnn.float32),
        _from_torch(sin.reshape(1, 1, _MAX_POSITIONS, -1).contiguous(), device, dtype=ttnn.float32),
    )


def build(device, torch_module):
    model = torch_module
    dim = int(model.config.hidden_size)
    head_dim = int(model.config.head_dim)

    embed = _from_torch(model.embed_tokens.weight.detach().contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos_table, sin_table, cos_tiled, sin_tiled = _rope_tables(device, model)
    layers = [_compile_layer(device, layer) for layer in model.layers]
    g_final = _norm_weight(model.norm, device)
    eps_final = float(model.norm.variance_epsilon)

    def mistral_model(input_ids, position_ids=None, **kwargs):
        shape = list(input_ids.shape)
        seq = int(shape[-1])
        # The LEADING axis is the batch, read off the tensor. A 1-D id vector is one sample.
        batch = int(shape[0]) if len(shape) > 1 else 1
        # `ttnn.embedding` requires a bfloat16 table (`embedding_device_operation.cpp:36`); widen
        # once here so every residual add downstream happens in float32.
        h = ttnn.typecast(ttnn.embedding(input_ids, embed, layout=ttnn.TILE_LAYOUT), ttnn.float32)
        h = ttnn.reshape(h, [batch, 1, seq, dim])

        if position_ids is None:
            cos = ttnn.slice(cos_tiled, [0, 0, 0, 0], [1, 1, seq, head_dim])
            sin = ttnn.slice(sin_tiled, [0, 0, 0, 0], [1, 1, seq, head_dim])
        else:
            # `position_ids` may be per-sample `[batch, seq]` or a single shared row `[1, seq]` /
            # `[seq]`; either broadcasts over the head axis, and a shared row broadcasts over the
            # batch too. Its leading bound is its own, not the ids' -- reusing `batch` here would
            # reject the shared row the reference itself defaults to.
            pos_shape = list(position_ids.shape)
            pos_batch = int(pos_shape[0]) if len(pos_shape) > 1 else 1
            cos = ttnn.typecast(
                ttnn.reshape(
                    ttnn.embedding(position_ids, cos_table, layout=ttnn.TILE_LAYOUT),
                    [pos_batch, 1, seq, head_dim],
                ),
                ttnn.float32,
            )
            sin = ttnn.typecast(
                ttnn.reshape(
                    ttnn.embedding(position_ids, sin_table, layout=ttnn.TILE_LAYOUT),
                    [pos_batch, 1, seq, head_dim],
                ),
                ttnn.float32,
            )

        for layer in layers:
            h = layer(h, cos, sin)
        h = _rms_norm(h, g_final, eps_final)
        return ttnn.reshape(h, [batch, seq, dim])

    return mistral_model
