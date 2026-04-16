# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TT-NN Gaze-LLE: moves the 12-layer DINOv2 backbone encoder onto a single
Blackhole p150a. Patch embedding and the small gaze decoder stay on CPU for
this first port; subsequent iterations migrate them onto device.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F

import ttnn


def _to_device(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


# p150a Blackhole has a 10x13 compute grid; openvla uses 10x12 and it is validated.
_CORE_GRID = ttnn.CoreGrid(y=10, x=13)


class _BlockParams:
    """Holds on-device weights for one DINOv2 block."""

    def __init__(self, block, device):
        # Attention
        self.norm1_w = _to_device(block.norm1.weight.unsqueeze(0), device)
        self.norm1_b = _to_device(block.norm1.bias.unsqueeze(0), device)
        qkv_w = block.attn.qkv.weight.T.contiguous()
        self.qkv_w = _to_device(qkv_w, device)
        self.qkv_b = _to_device(block.attn.qkv.bias.unsqueeze(0), device)
        self.proj_w = _to_device(block.attn.proj.weight.T.contiguous(), device)
        self.proj_b = _to_device(block.attn.proj.bias.unsqueeze(0), device)
        self.ls1 = _to_device(block.ls1.scale_factor.unsqueeze(0), device)

        # MLP
        self.norm2_w = _to_device(block.norm2.weight.unsqueeze(0), device)
        self.norm2_b = _to_device(block.norm2.bias.unsqueeze(0), device)
        self.fc1_w = _to_device(block.mlp.fc1.weight.T.contiguous(), device)
        self.fc1_b = _to_device(block.mlp.fc1.bias.unsqueeze(0), device)
        self.fc2_w = _to_device(block.mlp.fc2.weight.T.contiguous(), device)
        self.fc2_b = _to_device(block.mlp.fc2.bias.unsqueeze(0), device)
        self.ls2 = _to_device(block.ls2.scale_factor.unsqueeze(0), device)


def _dinov2_attention(x, p: _BlockParams, num_heads: int):
    hidden_states = ttnn.layer_norm(x, weight=p.norm1_w, bias=p.norm1_b, epsilon=1e-6)
    qkv = ttnn.linear(hidden_states, p.qkv_w, bias=p.qkv_b, core_grid=_CORE_GRID)
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=num_heads)
    ttnn.deallocate(qkv)

    head_dim = hidden_states.shape[-1] // num_heads
    q = ttnn.mul_(q, 1.0 / (head_dim**0.5))
    scores = ttnn.matmul(q, k, core_grid=_CORE_GRID)
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    probs = ttnn.softmax_in_place(scores, numeric_stable=True)
    ctx = ttnn.matmul(probs, v, core_grid=_CORE_GRID)
    ttnn.deallocate(probs)
    ttnn.deallocate(v)
    ctx = ttnn.transformer.concatenate_heads(ctx)

    out = ttnn.linear(ctx, p.proj_w, bias=p.proj_b, core_grid=_CORE_GRID)
    ttnn.deallocate(ctx)
    out = ttnn.mul(out, p.ls1)
    return ttnn.add(x, out)


def _dinov2_mlp(x, p: _BlockParams):
    hidden_states = ttnn.layer_norm(x, weight=p.norm2_w, bias=p.norm2_b, epsilon=1e-6)
    h = ttnn.linear(hidden_states, p.fc1_w, bias=p.fc1_b, activation="gelu", core_grid=_CORE_GRID)
    h = ttnn.linear(h, p.fc2_w, bias=p.fc2_b, core_grid=_CORE_GRID)
    h = ttnn.mul(h, p.ls2)
    return ttnn.add(x, h)


def _dinov2_block(x, p: _BlockParams, num_heads: int):
    x = _dinov2_attention(x, p, num_heads)
    x = _dinov2_mlp(x, p)
    return x


class TtGazeLLE:
    """
    Hybrid TT-NN / torch Gaze-LLE wrapper.

    Device: DINOv2 encoder blocks + final backbone layer-norm.
    Host (torch CPU): patch embedding, cls/reg/pos embeddings, gaze decoder,
    heatmap head, optional in/out head.
    """

    def __init__(self, ref_model, device, inout: bool = True):
        self.device = device
        self.inout = inout
        self.ref = ref_model
        backbone = ref_model.backbone
        self.cfg = backbone.cfg
        self.num_heads = self.cfg.num_heads
        self.embed_dim = self.cfg.embed_dim
        self.num_reg_tokens = self.cfg.num_register_tokens

        self.block_params = [_BlockParams(blk, device) for blk in backbone.blocks]
        self.final_norm_w = _to_device(backbone.norm.weight.unsqueeze(0), device)
        self.final_norm_b = _to_device(backbone.norm.bias.unsqueeze(0), device)

    @torch.no_grad()
    def _backbone_forward(self, images: torch.Tensor) -> torch.Tensor:
        backbone = self.ref.backbone
        b = images.shape[0]
        patches = backbone.patch_embed_proj(images).flatten(2).transpose(1, 2)  # B,N,D
        cls = backbone.cls_token.expand(b, -1, -1)
        reg = backbone.reg_token.expand(b, -1, -1)
        x = torch.cat([cls, patches], dim=1) + backbone.pos_embed
        x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)

        x_tt = _to_device(x, self.device)
        for p in self.block_params:
            x_tt = _dinov2_block(x_tt, p, self.num_heads)
        x_tt = ttnn.layer_norm(x_tt, weight=self.final_norm_w, bias=self.final_norm_b, epsilon=1e-6)
        x_host = ttnn.to_torch(x_tt).to(torch.float32)
        ttnn.deallocate(x_tt)

        patch_tokens = x_host[:, 1 + self.num_reg_tokens :]
        out_h, out_w = backbone.get_out_size()
        return patch_tokens.view(b, out_h, out_w, -1).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, bboxes: List[Sequence[float]]):
        b = images.shape[0]
        feat = self._backbone_forward(images)

        ref = self.ref
        x = ref.linear(feat) + ref.pos_embed.unsqueeze(0)
        head_maps = torch.stack([ref._bbox_to_head_map(bb) for bb in bboxes]).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(1) * ref.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings

        x = x.flatten(start_dim=2).permute(0, 2, 1)
        if ref.inout:
            inout_tok = ref.inout_token.weight.unsqueeze(0).repeat(b, 1, 1)
            x = torch.cat([inout_tok, x], dim=1)

        x = ref.transformer(x)

        inout_preds = None
        if ref.inout:
            inout_preds = ref.inout_head(x[:, 0, :]).squeeze(-1)
            x = x[:, 1:, :]

        x = x.reshape(b, ref.featmap_h, ref.featmap_w, ref.dim).permute(0, 3, 1, 2)
        x = ref.heatmap_head(x).squeeze(1)
        x = F.interpolate(x.unsqueeze(1), size=ref.out_size, mode="bilinear", align_corners=False).squeeze(1)
        return {"heatmap": x, "inout": inout_preds}
