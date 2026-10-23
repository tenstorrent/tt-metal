# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU torch re-implementation of the TimesFM-3 network in the decomposition the device uses.

Pure torch, so it runs in the reference venv (``check_dataflow.py``) and in python_env. The device
decomposition differs from ``timesfm3.torch`` in these exact rewrites:

- **head_dim 80 -> 96, pairs interleaved.** Each head's lanes are ``[d0, d40, d1, d41, ..., d39, d79,
  0 x 16]``, so the reference's rotate-half pairs (i, i + 40) are the adjacent pairs ``rotary_embedding_llama``
  rotates, and the pad lanes stay zero. The QK RMSNorm over 96 lanes sees mean96(x^2) = (80/96) mean80(x^2),
  so its gain is multiplied by sqrt(80/96) and its eps by 80/96.
- **PerDimScale** ``1.442695 / sqrt(80) * softplus(w)`` is folded into the query_ln gain; SDPA keeps
  ``scale = sqrt(80)``.
- **Pre-norm gains** are folded into the following projections (gain-free RMSNorm before Q/K/V and ff0).
- **V = 1 variate attention** is ``W_o W_v pre_ln(h1)`` (softmax over one key is 1), zero where the key
  is masked (CPU SDPA returns 0 for a fully masked row), as one pre-multiplied 1280 x 1280 matrix.
- Attention masks come from ``num_front_masked`` per row (built once, not per layer).

The weight rewrites live in ``common/decomposition.py`` (shared with ``tt/``); this file holds the network.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from models.experimental.timesfm3_forecast.common.decomposition import *  # noqa: F401,F403
    from models.experimental.timesfm3_forecast.common.decomposition import (
        HEAD_DIM_PAD,
        NUM_HEADS,
        QK_EPS_PAD,
        SDPA_SCALE,
        AttentionWeights,
        LayerWeights,
        Weights,
        rope_tables,
        rotate_pairs,
    )
except ImportError:  # the reference venv loads this file without the models package
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
    from decomposition import *  # noqa: F401,F403
    from decomposition import (
        HEAD_DIM_PAD,
        NUM_HEADS,
        QK_EPS_PAD,
        SDPA_SCALE,
        AttentionWeights,
        LayerWeights,
        Weights,
        rope_tables,
        rotate_pairs,
    )

# Test hooks (check_dataflow.py): the QK-norm eps of the unpadded head (None = machine eps of the dtype),
# the eps factor for the padded head, and the masked-logit fill (-inf: a fully masked query row returns
# 0, as CPU SDPA does; a finite value gives the device's uniform weights instead).
QK_EPS80 = None
QK_EPS_FACTOR = QK_EPS_PAD
MASK_FILL = float("-inf")


def rms_norm(x: torch.Tensor, gain: torch.Tensor | None = None, eps: float | None = None) -> torch.Tensor:
    """``nn.RMSNorm`` (eps None = machine eps of the dtype)."""
    eps = torch.finfo(x.dtype).eps if eps is None else eps
    y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
    return y if gain is None else y * gain


def attend(q, k, v, allowed: torch.Tensor, scale: float) -> torch.Tensor:
    """softmax(scale q k^T, masked) v; a query with no allowed key returns 0 (as CPU SDPA does)."""
    logits = (q @ k.transpose(-1, -2)) * scale
    logits = logits.masked_fill(~allowed, MASK_FILL)
    w = torch.softmax(logits, -1).nan_to_num(0.0)
    return w @ v


@dataclass
class Masks:
    """Attention masks for one forward, built once from ``num_front_masked`` (rows,) and the variate count."""

    rows: int
    variates: int
    num_patches: int
    token_valid: torch.Tensor  # (rows, n) bool
    seq_allowed: torch.Tensor  # (rows, 1, n, n) bool: causal and key not leading padding

    @classmethod
    def build(cls, num_front_masked: torch.Tensor, variates: int, num_patches: int) -> "Masks":
        idx = torch.arange(num_patches)
        valid = idx[None, :] >= num_front_masked[:, None]
        causal = idx[:, None] >= idx[None, :]
        return cls(
            rows=num_front_masked.shape[0],
            variates=variates,
            num_patches=num_patches,
            token_valid=valid,
            seq_allowed=(causal[None] & valid[:, None, :])[:, None],
        )


def split_heads(x: torch.Tensor) -> torch.Tensor:
    """(..., n, 1536) -> (..., 16, n, 96)."""
    *lead, n, _ = x.shape
    return x.reshape(*lead, n, NUM_HEADS, HEAD_DIM_PAD).transpose(-2, -3)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(..., 16, n, 96) -> (..., n, 1536)."""
    *lead, h, n, d = x.shape
    return x.transpose(-2, -3).reshape(*lead, n, h * d)


def qk_norm(q, k, w: AttentionWeights):
    eps = (torch.finfo(q.dtype).eps if QK_EPS80 is None else QK_EPS80) * QK_EPS_FACTOR
    return rms_norm(q, w.q_gain, eps), rms_norm(k, w.k_gain, eps)


def seq_attention(x: torch.Tensor, w: AttentionWeights, masks: Masks, cos, sin, query_tokens=None) -> torch.Tensor:
    """h1 = x + post_ln(SeqAttn(pre_ln(x))). x: (rows, n, 1280). ``query_tokens`` keeps only those queries."""
    xn = rms_norm(x)
    q, k, v = (split_heads(t) for t in (xn @ w.wqkv.T).split(NUM_HEADS * HEAD_DIM_PAD, -1))
    q = q * cos + rotate_pairs(q) * sin
    k = k * cos + rotate_pairs(k) * sin
    q, k = qk_norm(q, k, w)
    allowed = masks.seq_allowed
    if query_tokens is not None:
        q, allowed, x = q[:, :, query_tokens], allowed[:, :, query_tokens], x[:, query_tokens]
    o = merge_heads(attend(q, k, v, allowed, SDPA_SCALE)) @ w.wo.T
    return x + rms_norm(o, w.post_gain)


def var_attention(h1: torch.Tensor, w: AttentionWeights, masks: Masks, token_valid: torch.Tensor) -> torch.Tensor:
    """h2 = h1 + post_ln(VarAttn(pre_ln(h1))). h1: (rows, n, 1280), rows = items * variates."""
    xn = rms_norm(h1)
    if masks.variates == 1:
        o = (xn @ w.w_ov.T) * token_valid[..., None].to(xn.dtype)
        return h1 + rms_norm(o, w.post_gain)
    rows, n, _ = h1.shape
    b, v = rows // masks.variates, masks.variates
    qkv = (xn @ w.wqkv.T).reshape(b, v, n, -1).transpose(1, 2)  # (b, n, v, 3 * 1536)
    q, k, val = (split_heads(t) for t in qkv.split(NUM_HEADS * HEAD_DIM_PAD, -1))  # (b, n, 16, v, 96)
    q, k = qk_norm(q, k, w)
    key_valid = token_valid.reshape(b, v, n).transpose(1, 2)  # (b, n, v)
    allowed = key_valid[:, :, None, None, :].expand(b, n, 1, v, v)
    o = merge_heads(attend(q, k, val, allowed, SDPA_SCALE))  # (b, n, v, 1536)
    o = o.transpose(1, 2).reshape(rows, n, -1) @ w.wo.T
    return h1 + rms_norm(o, w.post_gain)


def ffn(h2: torch.Tensor, lw: LayerWeights) -> torch.Tensor:
    return h2 + rms_norm(torch.relu(rms_norm(h2) @ lw.ff0.T) @ lw.ff1.T, lw.post_ff_gain)


def tokenizer(tokens: torch.Tensor, w: Weights) -> torch.Tensor:
    return torch.relu(tokens @ w.tok_hidden.T) @ w.tok_out.T + tokens @ w.tok_res.T


def layer(x, lw: LayerWeights, masks: Masks, cos, sin, query_tokens=None):
    """One MixingTransformer layer; with ``query_tokens``, everything after the sequence attention's
    K/V runs on those tokens only (the last-layer cut)."""
    h1 = seq_attention(x, lw.seq, masks, cos, sin, query_tokens)
    valid = masks.token_valid if query_tokens is None else masks.token_valid[:, query_tokens]
    sub = masks if query_tokens is None else Masks(masks.rows, masks.variates, h1.shape[1], valid, None)
    h2 = var_attention(h1, lw.var, sub, valid)
    return ffn(h2, lw), (h1, h2)


def forward(
    tokens: torch.Tensor,
    num_front_masked: torch.Tensor,
    variates: int,
    w: Weights,
    readout_tokens=None,
    *,
    last_layer_cut: bool = False,
    capture: dict | None = None,
):
    """tokens (rows, n, 192) -> head output (rows, len(readout_tokens) or n, 576), normalized space."""
    dtype = w.tok_out.dtype
    tokens = tokens.to(dtype)
    n = tokens.shape[1]
    masks = Masks.build(num_front_masked, variates, n)
    cos, sin = rope_tables(n, dtype)
    x = tokenizer(tokens, w)
    if capture is not None:
        capture["transformer_input"] = x
        capture["layers"] = []
    layers = w.layers
    for i, lw in enumerate(layers):
        cut = last_layer_cut and readout_tokens is not None and i == len(layers) - 1
        x, (h1, h2) = layer(x, lw, masks, cos, sin, list(readout_tokens) if cut else None)
        if capture is not None:
            capture["layers"].append({"out": x, "h1": h1, "h2": h2})
    if readout_tokens is not None and not last_layer_cut:
        x = x[:, list(readout_tokens)]
    if capture is not None:
        capture["stack_output"] = x
    return x @ w.head_w.T + w.head_b
