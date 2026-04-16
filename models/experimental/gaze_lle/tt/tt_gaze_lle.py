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
    """Holds on-device weights for one DINOv2 block.

    LayerScale is folded into the preceding projection (``ls1 -> attn.proj``,
    ``ls2 -> mlp.fc2``) so each block drops two elementwise multiplies.
    """

    def __init__(self, block, device):
        # Attention
        self.norm1_w = _to_device(block.norm1.weight.unsqueeze(0), device)
        self.norm1_b = _to_device(block.norm1.bias.unsqueeze(0), device)
        self.qkv_w = _to_device(block.attn.qkv.weight.T.contiguous(), device)
        self.qkv_b = _to_device(block.attn.qkv.bias.unsqueeze(0), device)
        ls1 = block.ls1.scale_factor.detach()
        proj_w = block.attn.proj.weight.detach() * ls1.unsqueeze(-1)
        proj_b = block.attn.proj.bias.detach() * ls1
        self.proj_w = _to_device(proj_w.T.contiguous(), device)
        self.proj_b = _to_device(proj_b.unsqueeze(0), device)

        # MLP
        self.norm2_w = _to_device(block.norm2.weight.unsqueeze(0), device)
        self.norm2_b = _to_device(block.norm2.bias.unsqueeze(0), device)
        self.fc1_w = _to_device(block.mlp.fc1.weight.T.contiguous(), device)
        self.fc1_b = _to_device(block.mlp.fc1.bias.unsqueeze(0), device)
        ls2 = block.ls2.scale_factor.detach()
        fc2_w = block.mlp.fc2.weight.detach() * ls2.unsqueeze(-1)
        fc2_b = block.mlp.fc2.bias.detach() * ls2
        self.fc2_w = _to_device(fc2_w.T.contiguous(), device)
        self.fc2_b = _to_device(fc2_b.unsqueeze(0), device)


def _dinov2_attention(x, p: _BlockParams, num_heads: int):
    hidden_states = ttnn.layer_norm(x, weight=p.norm1_w, bias=p.norm1_b, epsilon=1e-6)
    qkv = ttnn.linear(hidden_states, p.qkv_w, bias=p.qkv_b, core_grid=_CORE_GRID)
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=num_heads, transpose_key=False
    )
    ttnn.deallocate(qkv)

    head_dim = hidden_states.shape[-1] // num_heads
    ctx = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=1.0 / (head_dim**0.5)
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    ctx = ttnn.transformer.concatenate_heads(ctx)

    out = ttnn.linear(ctx, p.proj_w, bias=p.proj_b, core_grid=_CORE_GRID)
    ttnn.deallocate(ctx)
    return ttnn.add(x, out)


def _dinov2_mlp(x, p: _BlockParams):
    hidden_states = ttnn.layer_norm(x, weight=p.norm2_w, bias=p.norm2_b, epsilon=1e-6)
    h = ttnn.linear(hidden_states, p.fc1_w, bias=p.fc1_b, activation="gelu", core_grid=_CORE_GRID)
    h = ttnn.linear(h, p.fc2_w, bias=p.fc2_b, core_grid=_CORE_GRID)
    return ttnn.add(x, h)


def _dinov2_block(x, p: _BlockParams, num_heads: int):
    x = _dinov2_attention(x, p, num_heads)
    x = _dinov2_mlp(x, p)
    return x


class _GazeBlockParams:
    """Holds on-device weights for one gaze-decoder transformer block (no LayerScale)."""

    def __init__(self, block, device):
        self.norm1_w = _to_device(block.norm1.weight.unsqueeze(0), device)
        self.norm1_b = _to_device(block.norm1.bias.unsqueeze(0), device)
        self.qkv_w = _to_device(block.attn.qkv.weight.T.contiguous(), device)
        self.qkv_b = _to_device(block.attn.qkv.bias.unsqueeze(0), device)
        self.proj_w = _to_device(block.attn.proj.weight.T.contiguous(), device)
        self.proj_b = _to_device(block.attn.proj.bias.unsqueeze(0), device)
        self.norm2_w = _to_device(block.norm2.weight.unsqueeze(0), device)
        self.norm2_b = _to_device(block.norm2.bias.unsqueeze(0), device)
        self.fc1_w = _to_device(block.mlp.fc1.weight.T.contiguous(), device)
        self.fc1_b = _to_device(block.mlp.fc1.bias.unsqueeze(0), device)
        self.fc2_w = _to_device(block.mlp.fc2.weight.T.contiguous(), device)
        self.fc2_b = _to_device(block.mlp.fc2.bias.unsqueeze(0), device)


def _gaze_block(x, p: _GazeBlockParams, num_heads: int):
    # Attention via fused scaled-dot-product-attention kernel.
    hidden_states = ttnn.layer_norm(x, weight=p.norm1_w, bias=p.norm1_b, epsilon=1e-6)
    qkv = ttnn.linear(hidden_states, p.qkv_w, bias=p.qkv_b, core_grid=_CORE_GRID)
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=num_heads, transpose_key=False
    )
    ttnn.deallocate(qkv)
    head_dim = hidden_states.shape[-1] // num_heads
    ctx = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=1.0 / (head_dim**0.5)
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    ctx = ttnn.transformer.concatenate_heads(ctx)
    out = ttnn.linear(ctx, p.proj_w, bias=p.proj_b, core_grid=_CORE_GRID)
    ttnn.deallocate(ctx)
    x = ttnn.add(x, out)

    # MLP
    hidden_states = ttnn.layer_norm(x, weight=p.norm2_w, bias=p.norm2_b, epsilon=1e-6)
    h = ttnn.linear(hidden_states, p.fc1_w, bias=p.fc1_b, activation="gelu", core_grid=_CORE_GRID)
    h = ttnn.linear(h, p.fc2_w, bias=p.fc2_b, core_grid=_CORE_GRID)
    return ttnn.add(x, h)


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

        # --- Gaze decoder: 1x1 conv projection, pos_embed, head_token, transformer, inout head on device
        linear_w = ref_model.linear.weight.squeeze(-1).squeeze(-1).T.contiguous()  # (768,256)
        self.proj_w = _to_device(linear_w, device)
        self.proj_b = _to_device(ref_model.linear.bias.unsqueeze(0), device)

        # pos_embed (dim=256, featmap_h, featmap_w) → (1, H*W, 256)
        pe = ref_model.pos_embed.permute(1, 2, 0).reshape(1, -1, ref_model.dim).contiguous()
        self.gaze_pos_embed = _to_device(pe, device)
        # head_token.weight: (1, 256)
        self.head_token = ref_model.head_token.weight  # kept on host for per-frame head_map mult
        if inout:
            self.inout_token = _to_device(ref_model.inout_token.weight.unsqueeze(0), device)

        self.gaze_block_params = [_GazeBlockParams(blk, device) for blk in ref_model.transformer]

    @torch.no_grad()
    def _backbone_forward_to_patch_tokens(self, images: torch.Tensor) -> "ttnn.Tensor":
        """Run DINOv2 on device, return patch tokens on-device as (B, H*W, embed_dim)."""
        backbone = self.ref.backbone
        b = images.shape[0]
        patches = backbone.patch_embed_proj(images).flatten(2).transpose(1, 2)
        cls = backbone.cls_token.expand(b, -1, -1)
        reg = backbone.reg_token.expand(b, -1, -1)
        x = torch.cat([cls, patches], dim=1) + backbone.pos_embed
        x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)

        x_tt = _to_device(x, self.device)
        for p in self.block_params:
            x_tt = _dinov2_block(x_tt, p, self.num_heads)
        x_tt = ttnn.layer_norm(x_tt, weight=self.final_norm_w, bias=self.final_norm_b, epsilon=1e-6)

        # Slice CLS + register tokens off on device. Fallback to host round-trip if unsupported.
        total_prefix = 1 + self.num_reg_tokens
        shape = x_tt.shape
        try:
            patch_tokens = ttnn.slice(x_tt, [0, total_prefix, 0], [shape[0], shape[1], shape[2]])
            ttnn.deallocate(x_tt)
            return patch_tokens
        except Exception:
            x_host = ttnn.to_torch(x_tt)
            ttnn.deallocate(x_tt)
            patch_tokens = x_host[:, total_prefix:].contiguous()
            return _to_device(patch_tokens, self.device)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, bboxes: List[Sequence[float]]):
        ref = self.ref
        b = images.shape[0]

        feat_tt = self._backbone_forward_to_patch_tokens(images)
        # Project 768 -> 256 on device and add pos_embed.
        x_tt = ttnn.linear(feat_tt, self.proj_w, bias=self.proj_b, core_grid=_CORE_GRID)
        ttnn.deallocate(feat_tt)
        x_tt = ttnn.add(x_tt, self.gaze_pos_embed)

        # Head-map conditioning: head_map (B, H*W) * head_token (256) → (B, H*W, 256).
        head_maps = torch.stack([ref._bbox_to_head_map(bb) for bb in bboxes]).view(b, -1)
        head_contrib = head_maps.unsqueeze(-1) * self.head_token.unsqueeze(0)
        head_contrib_tt = _to_device(head_contrib.contiguous(), self.device)
        x_tt = ttnn.add(x_tt, head_contrib_tt)
        ttnn.deallocate(head_contrib_tt)

        # Prepend in/out token. self.inout_token already shaped (1,1,256); requires B==1.
        if self.inout:
            assert b == 1, "TtGazeLLE currently supports B=1 only"
            x_tt = ttnn.concat([self.inout_token, x_tt], dim=1)

        for gp in self.gaze_block_params:
            x_tt = _gaze_block(x_tt, gp, num_heads=8)

        x_host = ttnn.to_torch(x_tt)
        ttnn.deallocate(x_tt)

        inout_preds = None
        if self.inout:
            inout_preds = ref.inout_head(x_host[:, 0, :].float()).squeeze(-1)
            x_host = x_host[:, 1:, :]

        x_host = x_host.reshape(b, ref.featmap_h, ref.featmap_w, ref.dim).permute(0, 3, 1, 2).float()
        x_host = ref.heatmap_head(x_host).squeeze(1)
        x_host = F.interpolate(x_host.unsqueeze(1), size=ref.out_size, mode="bilinear", align_corners=False).squeeze(1)
        return {"heatmap": x_host, "inout": inout_preds}
