# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of the FFN module inside every DaViT block.

DaViT's `SpatialBlock` and `ChannelBlock` each wrap their FFN as

    PreNorm(LayerNorm, Mlp(linear -> GELU(tanh) -> linear), drop_path)

which evaluates as `x + drop_path(Mlp(LN(x)))`. In eval mode drop_path is
identity, so it's just `x + Mlp(LN(x))`.

The Mlp's two linear weights are by far the heaviest tensors in each
DaViT block (dim x 4*dim each way). Across the 12 dual-blocks (24 FFN
modules) this dominates DaViT's parameter footprint.

Each ported FFN takes a torch tensor in and returns a torch tensor out
(plus the unchanged `size` tuple), matching the upstream signature so
the `MySequential` wrapper still works.
"""

from __future__ import annotations

import torch
from torch import nn


def _bf16_tile(ttnn_mod, t: torch.Tensor, device):
    return ttnn_mod.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn_mod.bfloat16, layout=ttnn_mod.TILE_LAYOUT, device=device,
    )


def _bfp8_tile(ttnn_mod, t: torch.Tensor, device):
    return ttnn_mod.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn_mod.bfloat8_b, layout=ttnn_mod.TILE_LAYOUT, device=device,
    )


class TTNNDaViTPreNormFFN(nn.Module):
    """`PreNorm(LN, Mlp)` evaluated end-to-end on the device.

    Replaces a `PreNorm` whose `fn` is an `Mlp` and whose `norm` is a
    `LayerNorm`. `drop_path` is treated as identity (eval-mode behavior).
    """

    def __init__(self, prenorm_torch: nn.Module, device) -> None:
        super().__init__()
        import ttnn

        self._ttnn = ttnn
        self.device = device

        norm = prenorm_torch.norm
        mlp = prenorm_torch.fn

        # Mlp.net is an nn.Sequential of OrderedDict([fc1, act, drop1, norm, fc2, drop2]).
        net = mlp.net
        fc1 = net.fc1
        fc2 = net.fc2

        self.ln_w = _bf16_tile(ttnn, norm.weight.detach(), device)
        self.ln_b = _bf16_tile(ttnn, norm.bias.detach(), device)
        # bfp8_b for the big matmul weights
        self.fc1_w = _bfp8_tile(ttnn, fc1.weight.detach().t().contiguous(), device)
        self.fc1_b = _bf16_tile(
            ttnn,
            (fc1.bias.detach() if fc1.bias is not None
             else torch.zeros(fc1.out_features)),
            device,
        )
        self.fc2_w = _bfp8_tile(ttnn, fc2.weight.detach().t().contiguous(), device)
        self.fc2_b = _bf16_tile(
            ttnn,
            (fc2.bias.detach() if fc2.bias is not None
             else torch.zeros(fc2.out_features)),
            device,
        )

    def forward(self, x: torch.Tensor, size):
        ttnn = self._ttnn
        residual_dtype = x.dtype
        x_tt = _bf16_tile(ttnn, x, self.device)
        h = ttnn.layer_norm(x_tt, weight=self.ln_w, bias=self.ln_b)
        h = ttnn.linear(h, self.fc1_w, bias=self.fc1_b, activation="gelu")
        h = ttnn.linear(h, self.fc2_w, bias=self.fc2_b)
        x_tt = ttnn.add(x_tt, h)
        ttnn.deallocate(h)
        out = ttnn.to_torch(x_tt).to(residual_dtype)
        ttnn.deallocate(x_tt)
        return out, size


def swap_davit_ffns(vision_tower: nn.Module, device) -> int:
    """Walk the DaViT and replace each block's `ffn` PreNorm with the
    on-device version. Returns the number of swaps made.

    DaViT's structure is:
        vision_tower.blocks: ModuleList[stage_seq]
        stage_seq: MySequential of dual_blocks
        dual_block: MySequential with .spatial_block and .channel_block
        spatial_block / channel_block: SpatialBlock / ChannelBlock
        each has .ffn = PreNorm(LayerNorm, Mlp(...))
    """
    swapped = 0
    for stage_seq in vision_tower.blocks:
        for dual_block in stage_seq:
            for sb_name in ("spatial_block", "channel_block"):
                stage_block = getattr(dual_block, sb_name, None)
                if stage_block is None or not hasattr(stage_block, "ffn"):
                    continue
                stage_block.ffn = TTNNDaViTPreNormFFN(stage_block.ffn, device).to(torch.bfloat16)
                swapped += 1
    return swapped
