# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN perception head — Stage 3.4.

Drop-in TTNN replacements for the weight-bearing perception-head ops in
``DiffusionDriveModel.forward``:

  * ``_bev_downscale``  — 1×1 Conv2d 512→256        (TtnnConv1x1)
  * ``_status_encoding``— Linear 8→256              (TtnnLinear)
  * ``bev_proj``        — Linear(320→256)+ReLU+LN   (TtnnBevProj)
  * ``_tf_decoder``     — 3-layer nn.TransformerDecoder (TtnnTransformerDecoder)

Each class is a torch ``nn.Module`` with the SAME call signature as the module
it replaces, so ``DiffusionDriveModel.forward`` is untouched.  Heavy compute
runs on TTNN (conv2d / linear / layer_norm / scaled_dot_product_attention);
only the thin tensor glue in ``forward`` (cat/permute/embedding-add) stays on
host until the Stage-3.7 single-graph consolidation.

All math runs in bfloat16; verified PCC ≥ 0.99 vs the PyTorch reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import ttnn
from models.demos.diffusion_drive.tt.ttnn_backbone import _from_ttnn_tile, _to_ttnn_tile
from models.demos.diffusion_drive.tt.ttnn_resnet34 import _ttnn_conv2d, prep_conv_weights

# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------


def _prep_linear(linear: nn.Linear, device):
    """torch Linear → (weight_tt (in,out), bias_tt (1,out) or None) in TILE layout."""
    w = ttnn.from_torch(
        linear.weight.detach().T.contiguous().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    b = None
    if linear.bias is not None:
        b = ttnn.from_torch(
            linear.bias.detach().reshape(1, -1).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
        )
    return w, b


def _prep_layernorm(ln: nn.LayerNorm, device):
    """torch LayerNorm → (gamma (1,d), beta (1,d), eps)."""
    g = ttnn.from_torch(ln.weight.detach().reshape(1, -1).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(ln.bias.detach().reshape(1, -1).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    return g, b, float(ln.eps)


# ---------------------------------------------------------------------------
# Simple drop-ins
# ---------------------------------------------------------------------------


class TtnnConv1x1(nn.Module):
    """1×1 Conv2d on TTNN (drop-in for _bev_downscale).  torch (B,Cin,H,W) → (B,Cout,H,W)."""

    def __init__(self, conv: nn.Conv2d, device) -> None:
        super().__init__()
        self._d = device
        w = conv.weight.detach().to(torch.bfloat16)
        b = (
            conv.bias.detach().to(torch.bfloat16)
            if conv.bias is not None
            else torch.zeros(conv.out_channels, dtype=torch.bfloat16)
        )
        self._w, self._b = prep_conv_weights(w, b)
        self._cout = conv.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, H, W = x.shape
        xt = _to_ttnn_tile(x, B, H, W, Cin, self._d)
        out, Ho, Wo = _ttnn_conv2d(self._d, xt, self._w, self._b, B, H, W, Cin, self._cout, 1, 1, 0)
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        return _from_ttnn_tile(out, B, Ho, Wo, self._cout)


class TtnnLinear(nn.Module):
    """Linear on TTNN (drop-in for _status_encoding).  torch (..., in) → (..., out)."""

    def __init__(self, linear: nn.Linear, device) -> None:
        super().__init__()
        self._d = device
        self._w, self._b = _prep_linear(linear, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = ttnn.from_torch(x.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self._d)
        y = ttnn.linear(xt, self._w, bias=self._b)
        return ttnn.to_torch(y).float()


class TtnnBevProj(nn.Module):
    """Linear(320→256)+ReLU+LayerNorm on TTNN (drop-in for bev_proj Sequential).

    torch (B, N, 320) → (B, N, 256).
    """

    def __init__(self, seq: nn.Sequential, device) -> None:
        super().__init__()
        self._d = device
        # seq == [Linear(320,256), ReLU(inplace), LayerNorm(256)]
        self._lw, self._lb = _prep_linear(seq[0], device)
        self._g, self._beta, self._eps = _prep_layernorm(seq[2], device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = ttnn.from_torch(x.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self._d)
        y = ttnn.linear(xt, self._lw, bias=self._lb)
        y = ttnn.relu(y)
        y = ttnn.layer_norm(y, weight=self._g, bias=self._beta, epsilon=self._eps)
        return ttnn.to_torch(y).float()


# ---------------------------------------------------------------------------
# Transformer decoder (3 × nn.TransformerDecoderLayer, post-norm, ReLU)
# ---------------------------------------------------------------------------


def _split_mha(mha: nn.MultiheadAttention):
    """Split nn.MultiheadAttention packed in_proj into per-projection torch tensors."""
    E = mha.embed_dim
    W = mha.in_proj_weight.detach()
    b = mha.in_proj_bias.detach()
    return {
        "Wq": W[:E],
        "bq": b[:E],
        "Wk": W[E : 2 * E],
        "bk": b[E : 2 * E],
        "Wv": W[2 * E :],
        "bv": b[2 * E :],
        "Wo": mha.out_proj.weight.detach(),
        "bo": mha.out_proj.bias.detach(),
        "nh": mha.num_heads,
        "E": E,
    }


class _TtnnMHA:
    """Multi-head attention on TTNN.  q_in, kv_in are TTNN (B,S,E) tile tensors."""

    def __init__(self, mha: nn.MultiheadAttention, device) -> None:
        self._d = device
        p = _split_mha(mha)
        self._nh = p["nh"]
        self._E = p["E"]
        self._hd = p["E"] // p["nh"]

        def lin(W, b):
            return (
                ttnn.from_torch(W.T.contiguous().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device),
                ttnn.from_torch(b.reshape(1, -1).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device),
            )

        self._wq, self._bq = lin(p["Wq"], p["bq"])
        self._wk, self._bk = lin(p["Wk"], p["bk"])
        self._wv, self._bv = lin(p["Wv"], p["bv"])
        self._wo, self._bo = lin(p["Wo"], p["bo"])

    def _to_heads(self, x, B, S):
        x = ttnn.reshape(x, (B, S, self._nh, self._hd))
        return ttnn.permute(x, (0, 2, 1, 3))  # (B, nh, S, hd)

    def __call__(self, q_in, kv_in, B, Sq, Skv):
        q = ttnn.linear(q_in, self._wq, bias=self._bq)
        k = ttnn.linear(kv_in, self._wk, bias=self._bk)
        v = ttnn.linear(kv_in, self._wv, bias=self._bv)
        q = self._to_heads(q, B, Sq)
        k = self._to_heads(k, B, Skv)
        v = self._to_heads(v, B, Skv)
        o = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)  # (B,nh,Sq,hd)
        o = ttnn.permute(o, (0, 2, 1, 3))  # (B,Sq,nh,hd)
        o = ttnn.reshape(o, (B, Sq, self._E))
        return ttnn.linear(o, self._wo, bias=self._bo)


class _TtnnDecoderLayer:
    """One nn.TransformerDecoderLayer (post-norm, ReLU, dropout=0) on TTNN."""

    def __init__(self, layer: nn.TransformerDecoderLayer, device) -> None:
        self._d = device
        self._self_attn = _TtnnMHA(layer.self_attn, device)
        self._cross_attn = _TtnnMHA(layer.multihead_attn, device)
        self._l1w, self._l1b = _prep_linear(layer.linear1, device)
        self._l2w, self._l2b = _prep_linear(layer.linear2, device)
        self._n1 = _prep_layernorm(layer.norm1, device)
        self._n2 = _prep_layernorm(layer.norm2, device)
        self._n3 = _prep_layernorm(layer.norm3, device)

    @staticmethod
    def _ln(x, n):
        return ttnn.layer_norm(x, weight=n[0], bias=n[1], epsilon=n[2])

    def __call__(self, tgt, memory, B, Sq, Skv):
        sa = self._self_attn(tgt, tgt, B, Sq, Sq)
        tgt = self._ln(ttnn.add(tgt, sa), self._n1)
        ca = self._cross_attn(tgt, memory, B, Sq, Skv)
        tgt = self._ln(ttnn.add(tgt, ca), self._n2)
        ff = ttnn.linear(tgt, self._l1w, bias=self._l1b)
        ff = ttnn.relu(ff)
        ff = ttnn.linear(ff, self._l2w, bias=self._l2b)
        tgt = self._ln(ttnn.add(tgt, ff), self._n3)
        return tgt


class TtnnTransformerDecoder(nn.Module):
    """3-layer TransformerDecoder on TTNN (drop-in for _tf_decoder).

    torch query (B,Sq,E), memory (B,Skv,E) → torch (B,Sq,E).
    """

    def __init__(self, decoder: nn.TransformerDecoder, device) -> None:
        super().__init__()
        self._d = device
        self._layers = [_TtnnDecoderLayer(l, device) for l in decoder.layers]

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B, Sq, _ = query.shape
        Skv = memory.shape[1]
        q = ttnn.from_torch(query.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self._d)
        m = ttnn.from_torch(memory.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self._d)
        for layer in self._layers:
            q = layer(q, m, B, Sq, Skv)
        return ttnn.to_torch(q).float()
