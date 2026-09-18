# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `g_p_t2_block` of coqui/XTTS-v2.

Reference submodule: `gpt.gpt.h.0`, a HuggingFace `GPT2Block`:

    residual = x
    h = ln_1(x)
    h = attn(h)                      # multi-head self-attention
    x = h + residual
    residual = x
    h = ln_2(x)
    h = mlp(h)                       # c_fc -> gelu_new -> c_proj
    x = h + residual
    return x

Config (probed): hidden=1024, n_head=16, head_dim=64, LayerNorm eps=1e-5,
attention scaling = head_dim**-0.5, activation = gelu_new (tanh approx).

Attention (`GPT2Attention`, eager): the fused `c_attn` (Conv1D 1024->3072)
output is split into q|k|v BLOCKS (each 1024, contiguous — NOT head-major),
each reshaped to `[1, H, T, head_dim]`. Weights are scaled by head_dim**-0.5;
softmax over keys; `attn @ v`; merge heads; `c_proj` (Conv1D 1024->1024).

IMPORTANT — no causal mask: HF creates the causal mask at the *model* level and
passes it down. This per-block PCC test calls the block with attention_mask=
None, so `eager_attention_forward` applies NO mask → FULL (bidirectional)
attention. We match that (a causal mask would fail PCC here).

Conv1D (GPT2) is `y = x @ weight + bias` with weight `[nx, nf]` (already in
`x @ weight` orientation — no transpose). Captured shapes: in/out `[1, 33, 1024]`.
"""

from __future__ import annotations

import ttnn
from models.demos.xtts_v2.tt.modules.conv1_d import build as _build_conv1d

_LN_EPS = 1e-5


def _to_tile(t):
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT)


def _to_rm(t):
    return ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)


def _split_heads(x, n_heads, head_dim):
    # [1, T, H*hd] -> [1, T, H, hd] -> [1, H, T, hd]
    t = x.shape[1]
    x4 = ttnn.reshape(_to_rm(x), (1, t, n_heads, head_dim))
    return _to_tile(ttnn.permute(x4, (0, 2, 1, 3)))


def _merge_heads(x):
    # [1, H, T, hd] -> [1, T, H, hd] -> [1, T, H*hd]
    x = ttnn.permute(x, (0, 2, 1, 3))
    _, t, h, hd = x.shape
    return _to_tile(ttnn.reshape(_to_rm(x), (1, t, h * hd)))


def build_gpt2_block(device, torch_module):
    """Return a native ttnn forward for a HF GPT2Block, binding trained weights.

    Reused by `g_p_t2_model` to build each layer in the stack.
    """
    import torch

    m = torch_module.float()
    attn = m.attn
    mlp = m.mlp

    n_heads = attn.num_heads
    head_dim = attn.head_dim
    embed_dim = attn.embed_dim
    scaling = float(head_dim) ** -0.5

    def _w(t):
        return ttnn.as_tensor(
            t.detach().contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    ln1_w, ln1_b = _w(m.ln_1.weight), _w(m.ln_1.bias)
    ln2_w, ln2_b = _w(m.ln_2.weight), _w(m.ln_2.bias)

    # GPT2 Conv1D projections (y = x @ weight + bias, weight [nx, nf], no
    # transpose) run through the conv1_d leaf module. Its build binds
    # bf16 TILE weight/bias and its forward does matmul(x, w) + bias — byte
    # identical to the inline version the block used before.
    c_attn_fwd = _build_conv1d(device, attn.c_attn)  # 1024 -> 3072
    c_proj_fwd = _build_conv1d(device, attn.c_proj)  # 1024 -> 1024
    c_fc_fwd = _build_conv1d(device, mlp.c_fc)  # 1024 -> 4096
    mlp_proj_fwd = _build_conv1d(device, mlp.c_proj)  # 4096 -> 1024

    # HiFi4 + fp32 accumulation for the attention score/context matmuls: the AR
    # decoder's greedy argmax is sensitive to bf16 accumulation error over the
    # 30-layer stack, so run these full-fidelity to track the fp32 reference.
    _attn_kernel_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def forward(
        hidden_states,
        *args,
        attn_bias=None,
        kv_out=False,
        k_cache=None,
        v_cache=None,
        cur_pos=None,
        sel_ar=None,
        ones_c=None,
        **kwargs,
    ):
        x = hidden_states

        # --- self-attention ---
        h = ttnn.layer_norm(x, epsilon=_LN_EPS, weight=ln1_w, bias=ln1_b)
        qkv = c_attn_fwd(h)  # [1, T, 3*embed]

        # Fused split + head-reshape + key-transpose in ONE device op (no
        # ROW_MAJOR round-trips). HF's c_attn emits q|k|v as contiguous 1024
        # blocks; reshaping [3*embed] -> [48, head_dim] and slicing [:16]/
        # [16:32]/[32:] recovers exactly those blocks, so this is byte-identical
        # to the old slice + _split_heads path. transpose_key=True yields k as
        # [1, H, hd, T], ready for q @ k^T without a separate transpose.
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            num_heads=n_heads,
            transpose_key=False,
        )  # each [1, H, T, hd]

        if k_cache is not None:
            # KV-cached decode step: hidden is [1,1,embed]; write the new k/v at
            # cur_pos in the [1,H,C,hd] fp32 caches (masked in-place), then attend the
            # single query over the whole cache with keys > cur_pos masked out.
            # scores -> softmax -> context run in fp32 — the fused SDPA also keeps
            # these stages in fp32, and the AR argmax flips on the accumulated bf16
            # rounding of an unfused path. All shapes fixed, so the step is traceable.
            C = int(k_cache.shape[2])
            sel = ttnn.reshape(ttnn.typecast(ttnn.eq(sel_ar, cur_pos), ttnn.float32), [1, 1, C, 1])
            keep = ttnn.subtract(ones_c, sel)
            ttnn.add(
                ttnn.multiply(k_cache, keep), ttnn.multiply(ttnn.typecast(k, ttnn.float32), sel), output_tensor=k_cache
            )
            ttnn.add(
                ttnn.multiply(v_cache, keep), ttnn.multiply(ttnn.typecast(v, ttnn.float32), sel), output_tensor=v_cache
            )
            disallowed = ttnn.reshape(
                ttnn.multiply(ttnn.typecast(ttnn.gt(sel_ar, cur_pos), ttnn.float32), -1e9), [1, 1, 1, C]
            )
            scores = ttnn.matmul(
                ttnn.typecast(q, ttnn.float32), ttnn.transpose(k_cache, -1, -2), compute_kernel_config=_attn_kernel_cfg
            )  # fp32 out (k_cache is fp32)
            scores = ttnn.add(ttnn.multiply(scores, scaling), disallowed)  # scale=1/8 exact
            probs = ttnn.softmax(scores, dim=-1)  # fp32
            attn_out = ttnn.typecast(
                ttnn.matmul(probs, v_cache, compute_kernel_config=_attn_kernel_cfg), ttnn.bfloat16
            )  # [1,H,1,hd]
        else:
            # Fused FlashAttention-2: (q @ k^T) * scale (+ causal mask) -> softmax
            # -> @ v, in ONE device op. Replaces the explicit 5-op path (score
            # matmul, scale-multiply, mask-add, softmax, context matmul) — kills the
            # dispatch-bound tiny 64x64x64 BMMs + the softmax/mul/add. is_causal
            # matches the GPT2Model's lower-triangular bias; the standalone per-block
            # PCC test (attn_bias=None) is FULL (bidirectional) attention.
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=attn_bias is not None,
                scale=scaling,
                compute_kernel_config=_attn_kernel_cfg,
            )  # [1, H, T, hd]

        # Fused merge (inverse of the split above) — one device op, no layout churn.
        attn_out = ttnn.transformer.concatenate_heads(attn_out)  # [1, T, embed]
        attn_out = c_proj_fwd(attn_out)

        x = ttnn.add(attn_out, x)  # residual

        # --- MLP ---
        h = ttnn.layer_norm(x, epsilon=_LN_EPS, weight=ln2_w, bias=ln2_b)
        h = c_fc_fwd(h)
        h = ttnn.gelu(h, fast_and_approximate_mode=True)  # gelu_new (tanh approx)
        h = mlp_proj_fwd(h)

        x = ttnn.add(h, x)  # residual
        if kv_out:
            return x, k, v
        return x

    return forward


def build(device, torch_module):
    return build_gpt2_block(device, torch_module)


def g_p_t2_block(hidden_states, *args, **kwargs):
    raise RuntimeError(
        "g_p_t2_block requires build(device, torch_module) to bind trained "
        "weights; the bare callable has no parameters."
    )
