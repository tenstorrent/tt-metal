# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM-3 weights in the device decomposition (torch only; shared by ``tt/`` and the CPU dataflow).

- **head_dim 80 -> 96, pairs interleaved.** Each head's Q/K/V rows are laid out
  ``[d0, d40, d1, d41, ..., d39, d79, 0 x 16]`` and W_o's columns the same way: the reference's rotate-half
  pairs (i, i + 40) become the adjacent pairs (2i, 2i + 1) that ``rotary_embedding_llama`` rotates (its
  32 x 32 transformation matrix never straddles a tile), and the 16 pad lanes stay zero. The QK RMSNorm
  over 96 lanes sees mean96(x^2) = (80/96) mean80(x^2), so its gain is multiplied by sqrt(80/96) and its
  eps by 80/96. Q.K dot products and SDPA are unchanged by the lane permutation.
- **PerDimScale** ``1.442695 / sqrt(80) * softplus(w)`` is folded into the query_ln gain; SDPA keeps
  ``scale = sqrt(80)``.
- **Pre-norm gains** are folded into the following projections (gain-free RMSNorm before Q/K/V and ff0).
- **V = 1 variate attention** uses ``w_ov = W_o (W_v g_pre)``, one pre-multiplied 1280 x 1280 matrix.

This file must stay importable without the ``models`` package: the reference venv loads it by path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

D_MODEL = 1280
NUM_HEADS = 16
HEAD_DIM = 80
HEAD_DIM_PAD = 96
HALF = HEAD_DIM // 2
NUM_LAYERS = 20
ROPE_MAX_TIMESCALE = 10000.0
RECIPROCAL_OF_SOFTPLUS_0 = 1.442695041
SDPA_SCALE = math.sqrt(HEAD_DIM)
QK_GAIN_PAD = math.sqrt(HEAD_DIM / HEAD_DIM_PAD)
QK_EPS_PAD = HEAD_DIM / HEAD_DIM_PAD


def pad_index() -> torch.Tensor:
    """Slot of each of the 80 head lanes in the 96-lane padded head: lane i < 40 -> 2i, lane 40 + i -> 2i + 1."""
    i = torch.arange(HEAD_DIM)
    return torch.where(i < HALF, 2 * i, 2 * (i - HALF) + 1)


def pad_head_vector(v: torch.Tensor) -> torch.Tensor:
    """(80,) -> (96,) in the padded lane layout, zeros in the pad lanes."""
    out = torch.zeros(HEAD_DIM_PAD, dtype=v.dtype)
    out[pad_index()] = v
    return out


def pad_heads_rows(w: torch.Tensor) -> torch.Tensor:
    """Q/K/V ``nn.Linear`` weight (16*80, in) -> (16*96, in): zero rows in the pad lanes of every head."""
    w = w.reshape(NUM_HEADS, HEAD_DIM, -1)
    out = torch.zeros(NUM_HEADS, HEAD_DIM_PAD, w.shape[-1], dtype=w.dtype)
    out[:, pad_index()] = w
    return out.reshape(NUM_HEADS * HEAD_DIM_PAD, -1)


def pad_heads_cols(w: torch.Tensor) -> torch.Tensor:
    """out_proj weight (out, 16*80) -> (out, 16*96): zero columns in the pad lanes."""
    w = w.reshape(w.shape[0], NUM_HEADS, HEAD_DIM)
    out = torch.zeros(w.shape[0], NUM_HEADS, HEAD_DIM_PAD, dtype=w.dtype)
    out[:, :, pad_index()] = w
    return out.reshape(w.shape[0], -1)


def rope_tables(num_patches: int, dtype=torch.float32):
    """cos/sin (n, 96) for the interleaved pairs: lanes 2i and 2i + 1 hold angle i; pad lanes cos 1, sin 0.
    As in the reference, the timescales are built in fp32 and the angles are computed in ``dtype``
    (``RotaryPositionalEmbedding`` after ``.to(dtype)``)."""
    fraction = 2.0 * torch.arange(HALF, dtype=torch.float32) / HEAD_DIM
    timescale = (1.0 * (ROPE_MAX_TIMESCALE / 1.0) ** fraction).to(dtype)
    pos = torch.arange(num_patches, dtype=torch.int32).to(dtype)
    ang = pos[:, None] / timescale[None, :]
    cos80, sin80 = torch.cos(ang), torch.sin(ang)
    cos = torch.ones(num_patches, HEAD_DIM_PAD, dtype=dtype)
    sin = torch.zeros(num_patches, HEAD_DIM_PAD, dtype=dtype)
    for parity in (0, 1):
        cos[:, parity : 2 * HALF : 2] = cos80
        sin[:, parity : 2 * HALF : 2] = sin80
    return cos, sin


def pair_rotation(n: int = 32) -> torch.Tensor:
    """(n, n) with ``x @ R`` mapping each adjacent pair (a, b) to (-b, a): ``rotary_embedding_llama``'s
    transformation matrix, and the dataflow's rotation."""
    r = torch.zeros(n, n)
    r[torch.arange(0, n, 2), torch.arange(1, n, 2)] = 1.0
    r[torch.arange(1, n, 2), torch.arange(0, n, 2)] = -1.0
    return r


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """(a_i, b_i) -> (-b_i, a_i) for the adjacent lane pairs (2i, 2i + 1)."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    return torch.stack([-pairs[..., 1], pairs[..., 0]], -1).reshape(x.shape)


@dataclass
class AttentionWeights:
    wqkv: torch.Tensor  # (3 * 1536, 1280), pre-norm gain folded; for V = 1 only wv is used
    q_gain: torch.Tensor  # (96,) query_ln gain * PerDimScale * sqrt(80/96)
    k_gain: torch.Tensor  # (96,) key_ln gain * sqrt(80/96)
    wo: torch.Tensor  # (1280, 1536)
    post_gain: torch.Tensor  # (1280,)
    w_ov: torch.Tensor | None = None  # (1280, 1280) = W_o @ (W_v * g_pre): the V = 1 shortcut


@dataclass
class LayerWeights:
    seq: AttentionWeights
    var: AttentionWeights
    ff0: torch.Tensor  # (1280, 1280), pre_ff gain folded
    ff1: torch.Tensor
    post_ff_gain: torch.Tensor


@dataclass
class Weights:
    tok_hidden: torch.Tensor  # (1280, 192)
    tok_out: torch.Tensor  # (1280, 1280)
    tok_res: torch.Tensor  # (1280, 192)
    layers: list = field(default_factory=list)
    head_w: torch.Tensor | None = None  # (576, 1280)
    head_b: torch.Tensor | None = None  # (576,)


def _attention_weights(sd, prefix: str, pre_gain: torch.Tensor, post_gain: torch.Tensor, dtype) -> AttentionWeights:
    g = lambda k: sd[f"{prefix}.{k}"].to(torch.float64)  # noqa: E731
    wq, wk, wv, wo = (
        g(f"{p}_proj.weight") * (pre_gain[None, :] if p != "out" else 1.0) for p in ("query", "key", "value", "out")
    )
    scale = (
        RECIPROCAL_OF_SOFTPLUS_0 / math.sqrt(HEAD_DIM) * torch.nn.functional.softplus(g("per_dim_scale.per_dim_scale"))
    )
    q_gain = pad_head_vector(g("query_ln.weight") * scale) * QK_GAIN_PAD
    k_gain = pad_head_vector(g("key_ln.weight")) * QK_GAIN_PAD
    wqkv = torch.cat([pad_heads_rows(wq), pad_heads_rows(wk), pad_heads_rows(wv)])
    return AttentionWeights(
        wqkv=wqkv.to(dtype),
        q_gain=q_gain.to(dtype),
        k_gain=k_gain.to(dtype),
        wo=pad_heads_cols(wo).to(dtype),
        post_gain=post_gain.to(dtype),
        w_ov=(wo @ wv).to(dtype),
    )


def load_weights(state_dict, dtype=torch.float32, num_layers: int = NUM_LAYERS) -> Weights:
    """Device-decomposition weights from the checkpoint's state dict (``model.safetensors`` keys)."""
    sd = state_dict
    f = lambda k: sd[k].to(torch.float64)  # noqa: E731
    w = Weights(
        tok_hidden=f("pre_transformer_resblock.hidden_layer.weight").to(dtype),
        tok_out=f("pre_transformer_resblock.output_layer.weight").to(dtype),
        tok_res=f("pre_transformer_resblock.residual_layer.weight").to(dtype),
        head_w=f("output_head.weight").to(dtype),
        head_b=f("output_head.bias").to(dtype),
    )
    for i in range(num_layers):
        p = f"transformer_stack.layers.{i}"
        seq = _attention_weights(
            sd, f"{p}.seq_attn", f(f"{p}.pre_seq_attn_ln.weight"), f(f"{p}.post_seq_attn_ln.weight"), dtype
        )
        var = _attention_weights(
            sd, f"{p}.var_attn", f(f"{p}.pre_var_attn_ln.weight"), f(f"{p}.post_var_attn_ln.weight"), dtype
        )
        w.layers.append(
            LayerWeights(
                seq=seq,
                var=var,
                ff0=(f(f"{p}.ff0.weight") * f(f"{p}.pre_ff_ln.weight")[None, :]).to(dtype),
                ff1=f(f"{p}.ff1.weight").to(dtype),
                post_ff_gain=f(f"{p}.post_ff_ln.weight").to(dtype),
            )
        )
    return w
