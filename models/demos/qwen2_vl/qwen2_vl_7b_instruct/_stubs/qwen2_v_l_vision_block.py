# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `qwen2_v_l_vision_block` of Qwen/Qwen2-VL-7B-Instruct.

Reference: `visual.blocks.0`, a `Qwen2VLVisionBlock`:

    h = norm1(hidden_states)                                   # LayerNorm
    q, k, v = qkv(h).reshape(seq_len, 3, num_heads, head_dim).unbind(1)
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)          # rotate_half, per mrope table
    # full (non-causal) attention within each `cu_seqlens` chunk independently
    attn_out = proj(concat(attn(q_i, k_i, v_i) for each chunk i))
    hidden_states = hidden_states + attn_out
    h2 = norm2(hidden_states)                                  # LayerNorm
    mlp_out = fc2(quick_gelu(fc1(h2)))
    hidden_states = hidden_states + mlp_out

`hidden_states` here is 2D `(seq_len, embed_dim)` (no leading batch dim) --
that is what `VisionAttention.forward` indexes via `hidden_states.shape[0]`.
"""

from __future__ import annotations

import ttnn

from . import vision_mlp as _vision_mlp_stub

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    block = torch_module
    attn = block.attn
    mlp = block.mlp
    norm1 = block.norm1
    norm2 = block.norm2

    # The block's MLP sub-computation is delegated to the graduated `vision_mlp`
    # port so that stub runs inside the real forward path (fc2(quick_gelu(fc1)),
    # float32 -- identical math to the inline version it replaces).
    mlp_forward = _vision_mlp_stub.build(device, mlp)

    nh = int(attn.num_heads)
    hd = int(attn.head_dim)
    dim = int(attn.dim)
    scaling = float(attn.scaling)

    def _w(t):
        return ttnn.from_torch(t.detach(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    norm1_w, norm1_b = _w(norm1.weight), _w(norm1.bias)
    norm2_w, norm2_b = _w(norm2.weight), _w(norm2.bias)
    eps1, eps2 = float(norm1.eps), float(norm2.eps)

    # `qkv` is a single fused Linear(dim, 3*dim); its output is laid out as
    # [Q-block (dim); K-block (dim); V-block (dim)] since the reference does
    # `.reshape(seq_len, 3, num_heads, head_dim)` on a contiguous (3*dim,) row.
    # Splitting the weight/bias into three plain Linears on the host lets the
    # device side run three ordinary matmuls instead of a fused-then-split op.
    qkv_weight = attn.qkv.weight.detach()
    qkv_bias = attn.qkv.bias.detach()
    q_w, k_w, v_w = qkv_weight[:dim], qkv_weight[dim : 2 * dim], qkv_weight[2 * dim :]
    q_b, k_b, v_b = qkv_bias[:dim], qkv_bias[dim : 2 * dim], qkv_bias[2 * dim :]
    q_w_dev, k_w_dev, v_w_dev = _w(q_w), _w(k_w), _w(v_w)
    q_b_dev, k_b_dev, v_b_dev = _w(q_b.reshape(1, -1)), _w(k_b.reshape(1, -1)), _w(v_b.reshape(1, -1))

    proj_w = _w(attn.proj.weight)
    proj_b = _w(attn.proj.bias.reshape(1, -1))

    def _linear(x, weight, bias=None):
        return ttnn.linear(x, weight, bias=bias, transpose_b=True, memory_config=_DRAM)

    def _rotate_half(x):
        x1 = x[..., : hd // 2]
        x2 = x[..., hd // 2 :]
        return ttnn.concat([ttnn.neg(x2), x1], dim=-1)

    def forward(
        hidden_states, cu_seqlens=None, position_embeddings=None, cos_dev=None, sin_dev=None, bounds=None, **kwargs
    ):
        # Trace/on-device path: when `cos_dev`/`sin_dev` (already (seq_len,1,hd)
        # device buffers) and `bounds` (python list) are supplied, NO host op
        # runs inside -- the block is pure ttnn.
        # This block is stacked 32-deep in the full vision transformer;
        # bfloat16 compute compounds rounding error across layers enough to
        # drop the end-to-end PCC below 0.99 even though any single block's
        # own PCC test passes in isolation at bf16. Run the whole block in
        # float32 so 32-layer composition still clears the 0.99 bar.
        if hidden_states.get_dtype() != ttnn.float32:
            hidden_states = ttnn.typecast(hidden_states, ttnn.float32)

        seq_len = int(hidden_states.shape[0])

        residual = hidden_states
        h = ttnn.layer_norm(hidden_states, weight=norm1_w, bias=norm1_b, epsilon=eps1)

        q = _linear(h, q_w_dev, q_b_dev)
        k = _linear(h, k_w_dev, k_b_dev)
        v = _linear(h, v_w_dev, v_b_dev)

        q = ttnn.reshape(q, (seq_len, nh, hd))
        k = ttnn.reshape(k, (seq_len, nh, hd))
        v = ttnn.reshape(v, (seq_len, nh, hd))

        if cos_dev is None:
            cos_raw, sin_raw = position_embeddings  # each (seq_len, head_dim) torch tensors
            cos_dev = ttnn.reshape(
                ttnn.from_torch(cos_raw.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device),
                (seq_len, 1, hd),
            )
            sin_dev = ttnn.reshape(
                ttnn.from_torch(sin_raw.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device),
                (seq_len, 1, hd),
            )
        q = ttnn.add(ttnn.mul(q, cos_dev), ttnn.mul(_rotate_half(q), sin_dev))
        k = ttnn.add(ttnn.mul(k, cos_dev), ttnn.mul(_rotate_half(k), sin_dev))

        q = ttnn.permute(q, (1, 0, 2))  # (nh, seq_len, hd)
        k = ttnn.permute(k, (1, 0, 2))
        v = ttnn.permute(v, (1, 0, 2))

        if bounds is None:
            bounds = cu_seqlens.tolist() if cu_seqlens is not None else [0, seq_len]

        chunk_outs = []
        for i in range(len(bounds) - 1):
            s, e = int(bounds[i]), int(bounds[i + 1])
            if e <= s:
                continue
            q_c, k_c, v_c = q[:, s:e, :], k[:, s:e, :], v[:, s:e, :]
            scores = ttnn.matmul(q_c, k_c, transpose_b=True, memory_config=_DRAM)
            scores = ttnn.mul(scores, scaling)
            probs = ttnn.softmax(scores, dim=-1)
            chunk_outs.append(ttnn.matmul(probs, v_c, memory_config=_DRAM))

        attn_out = chunk_outs[0] if len(chunk_outs) == 1 else ttnn.concat(chunk_outs, dim=1)
        attn_out = ttnn.reshape(ttnn.permute(attn_out, (1, 0, 2)), (seq_len, nh * hd))
        attn_out = _linear(attn_out, proj_w, proj_b)

        hidden_states = ttnn.add(residual, attn_out)

        residual2 = hidden_states
        h2 = ttnn.layer_norm(hidden_states, weight=norm2_w, bias=norm2_b, epsilon=eps2)
        mlp_out = mlp_forward(h2)  # graduated vision_mlp stub: fc2(quick_gelu(fc1(h2)))

        hidden_states = ttnn.add(residual2, mlp_out)
        return hidden_states

    return forward


def qwen2_v_l_vision_block(*args, **kwargs):
    raise RuntimeError(
        "qwen2_v_l_vision_block requires build(device, torch_module) to bind trained weights; "
        "the bare callable has no parameters."
    )
