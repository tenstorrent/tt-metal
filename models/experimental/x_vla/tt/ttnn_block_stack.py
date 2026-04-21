# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of the 24-layer TransformerBlock stack inside
`SoftPromptedTransformer` for X-VLA.

Each block is pre-LN:

    x = x + attn(layer_norm(x))
    x = x + mlp (layer_norm(x))

where:
    attn  : Q/K/V linear (fused), SDPA (non-causal), output proj, each with bias.
    mlp   : fc1 (H -> 4H), GELU-tanh, fc2 (4H -> H), both with bias.

The stack is what runs in the hot path. Moving 24 blocks on-chip amortizes
the single torch->ttnn and ttnn->torch transfer per inference.
"""

from __future__ import annotations

from typing import Any, List

import torch
from torch import nn


def _to_tile_tensor(ttnn_mod, t: torch.Tensor, device):
    """Upload a 2-D weight/1-D bias to device DRAM as bfloat16 tiled tensor."""
    return ttnn_mod.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn_mod.bfloat16,
        layout=ttnn_mod.TILE_LAYOUT,
        device=device,
    )


class TTNNTransformerBlockStack(nn.Module):
    """Drop-in replacement for `SoftPromptedTransformer.blocks` — runs the
    entire 24-block residual stack on device in one shot.

    The outer `SoftPromptedTransformer.forward` uses `for block in self.blocks:
    x = block(x)`. We preserve that by exposing this object as a
    `nn.ModuleList`-like iterable of length 1, where the single element is a
    callable that runs all 24 blocks in sequence.
    """

    def __init__(self, torch_blocks: nn.ModuleList, device, num_heads: int) -> None:
        super().__init__()
        import ttnn

        self._ttnn = ttnn
        self.device = device
        self.num_heads = num_heads
        self.num_layers = len(torch_blocks)
        head_dim = torch_blocks[0].attn.head_dim
        self.head_dim_inv_sqrt = float(head_dim) ** -0.5

        # Per-layer weight bundles, each kept on device DRAM.
        self._layers: List[Any] = []
        for blk in torch_blocks:
            self._layers.append(self._bundle_block(blk))

    # -- weight loading ------------------------------------------------------

    def _bundle_block(self, blk: nn.Module) -> dict:
        ttnn = self._ttnn
        dev = self.device

        def bf16(t):
            return _to_tile_tensor(ttnn, t, dev)

        def bfp8(t):
            # Per-block 8-bit exponent-sharing float — half the bytes of bf16.
            # Fine for over-parameterized transformer MLP weights (GPT-3-class
            # accuracy studies show <0.5% drop).
            return ttnn.from_torch(
                t.to(torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev,
            )

        # LayerNorm 1
        ln1_w = bf16(blk.norm1.weight.detach())
        ln1_b = bf16(blk.norm1.bias.detach())
        # QKV stays bf16 — those are the weights producing scores and any
        # precision loss here compounds through softmax. The attention
        # OUTPUT projection runs on post-softmax activations where rounding
        # is less brittle, so it can take bfp8_b.
        qkv_w = bf16(blk.attn.qkv.weight.detach().t().contiguous())
        qkv_b = bf16(blk.attn.qkv.bias.detach())
        proj_w = bfp8(blk.attn.proj.weight.detach().t().contiguous())
        proj_b = bf16(blk.attn.proj.bias.detach())
        # LayerNorm 2
        ln2_w = bf16(blk.norm2.weight.detach())
        ln2_b = bf16(blk.norm2.bias.detach())
        # MLP — by far the biggest weights (H*4H each way). bfp8 here halves
        # their DRAM footprint (for 24 layers: 24 * 2 * 1024*4096 * 2B =
        # 384MB -> 192MB), and matmul FLOPs are typically memory-bound on
        # this shape/batch so it should translate to throughput.
        fc1_w = bfp8(blk.mlp.fc1.weight.detach().t().contiguous())
        fc1_b = bf16(blk.mlp.fc1.bias.detach())
        fc2_w = bfp8(blk.mlp.fc2.weight.detach().t().contiguous())
        fc2_b = bf16(blk.mlp.fc2.bias.detach())

        return dict(
            ln1_w=ln1_w, ln1_b=ln1_b,
            qkv_w=qkv_w, qkv_b=qkv_b,
            proj_w=proj_w, proj_b=proj_b,
            ln2_w=ln2_w, ln2_b=ln2_b,
            fc1_w=fc1_w, fc1_b=fc1_b,
            fc2_w=fc2_w, fc2_b=fc2_b,
        )

    # -- forward -------------------------------------------------------------

    def _block_forward(self, x_tt, wb: dict, head_dim_inv_sqrt: float):
        """Run one pre-LN transformer block on-device.

        Attention is implemented manually (matmul+softmax+matmul) rather than
        calling `ttnn.transformer.scaled_dot_product_attention`, because
        Flash-2 on Blackhole currently requires Q/K/V seq lengths to be
        padded to the tile size and matching, and our seq length is not a
        multiple of 32. Manual SDPA works at any length.
        """
        ttnn = self._ttnn

        # --- Self-attention ------------------------------------------------
        h = ttnn.layer_norm(x_tt, weight=wb["ln1_w"], bias=wb["ln1_b"])
        qkv = ttnn.linear(h, wb["qkv_w"], bias=wb["qkv_b"])
        ttnn.deallocate(h)
        # split heads: [B, S, 3H] -> q,k,v each [B, num_heads, S, head_dim]
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads
        )
        ttnn.deallocate(qkv)
        # split_query_key_value_and_split_heads already returns K pre-transposed
        # to shape [B, H, head_dim, S], so q @ k directly gives attention scores.
        scores = ttnn.matmul(q, k)
        ttnn.deallocate(q); ttnn.deallocate(k)
        scores = ttnn.multiply(scores, head_dim_inv_sqrt)
        probs = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)
        attn_out = ttnn.matmul(probs, v)
        ttnn.deallocate(probs); ttnn.deallocate(v)
        # merge heads: [B, H, S, Dh] -> [B, S, H*Dh]
        attn_out = ttnn.transformer.concatenate_heads(attn_out)
        proj = ttnn.linear(attn_out, wb["proj_w"], bias=wb["proj_b"])
        ttnn.deallocate(attn_out)
        x_tt = ttnn.add(x_tt, proj)
        ttnn.deallocate(proj)

        # --- MLP -----------------------------------------------------------
        h = ttnn.layer_norm(x_tt, weight=wb["ln2_w"], bias=wb["ln2_b"])
        h = ttnn.linear(h, wb["fc1_w"], bias=wb["fc1_b"], activation="gelu")
        h = ttnn.linear(h, wb["fc2_w"], bias=wb["fc2_b"])
        x_tt = ttnn.add(x_tt, h)
        ttnn.deallocate(h)
        return x_tt

    def __iter__(self):
        # Support the `for block in self.blocks: x = block(x)` idiom — we
        # represent the whole stack as a single "block".
        yield self._run_all

    def __len__(self):
        return 1

    def _run_all(self, x_torch: torch.Tensor) -> torch.Tensor:
        """Torch tensor in, torch tensor out; the 24-block stack runs on chip."""
        ttnn = self._ttnn
        x_bf16 = x_torch.to(torch.bfloat16).contiguous()
        x_tt = ttnn.from_torch(
            x_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        for wb in self._layers:
            x_tt = self._block_forward(x_tt, wb, self.head_dim_inv_sqrt)
        out = ttnn.to_torch(x_tt).to(x_torch.dtype)
        ttnn.deallocate(x_tt)
        return out
