# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TT-NN Gaze-LLE, end-to-end on a single Blackhole p150a chip.

Everything between the first matmul and the final sigmoid runs on device.
Host-side work is pure layout, not inference compute:
  * `(B, 3, H, W) -> (B, num_patches, ps*ps*3)` view+permute+reshape of the image,
  * building the tiny (num_patches,) binary head bbox mask,
  * upload / download,
  * a `(B, num_patches, 4) -> (B, 64, 64)` view on the downloaded heatmap.

Device placement:
  * Patch embedding via a direct `(ps*ps*3, embed_dim)` matmul.
  * Pre-composed [CLS+pos_cls, REG] prefix + patch pos_embed add + concat.
  * DINOv2 ViT-B/14 encoder (12 blocks) + final LayerNorm + slice (CLS+REG).
  * 1x1 projection 768->256, gaze pos + head_map*head_token conditioning.
  * 3 gaze decoder blocks.
  * Fused ConvTranspose2d(256,256,2,2) + Conv2d(256,1,1) + Sigmoid heatmap head,
    expressed as a single (256,4) matmul + scalar add + sigmoid.
  * In/out MLP (256->128->1) with ReLU + Sigmoid.
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

# LoFi matmul for bfp8_b weights: 2x throughput vs HiFi4, small PCC impact.
_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)


class _BlockParams:
    """Holds on-device weights for one DINOv2 block.

    LayerScale is folded into the preceding projection (``ls1 -> attn.proj``,
    ``ls2 -> mlp.fc2``) so each block drops two elementwise multiplies.
    """

    def __init__(self, block, device):
        # Attention
        self.norm1_w = _to_device(block.norm1.weight.unsqueeze(0), device)
        self.norm1_b = _to_device(block.norm1.bias.unsqueeze(0), device)
        self.qkv_w = _to_device(block.attn.qkv.weight.T.contiguous(), device, dtype=ttnn.bfloat8_b)
        self.qkv_b = _to_device(block.attn.qkv.bias.unsqueeze(0), device)
        ls1 = block.ls1.scale_factor.detach()
        proj_w = block.attn.proj.weight.detach() * ls1.unsqueeze(-1)
        proj_b = block.attn.proj.bias.detach() * ls1
        self.proj_w = _to_device(proj_w.T.contiguous(), device, dtype=ttnn.bfloat8_b)
        self.proj_b = _to_device(proj_b.unsqueeze(0), device)

        # MLP
        self.norm2_w = _to_device(block.norm2.weight.unsqueeze(0), device)
        self.norm2_b = _to_device(block.norm2.bias.unsqueeze(0), device)
        self.fc1_w = _to_device(block.mlp.fc1.weight.T.contiguous(), device, dtype=ttnn.bfloat8_b)
        self.fc1_b = _to_device(block.mlp.fc1.bias.unsqueeze(0), device)
        ls2 = block.ls2.scale_factor.detach()
        fc2_w = block.mlp.fc2.weight.detach() * ls2.unsqueeze(-1)
        fc2_b = block.mlp.fc2.bias.detach() * ls2
        self.fc2_w = _to_device(fc2_w.T.contiguous(), device, dtype=ttnn.bfloat8_b)
        self.fc2_b = _to_device(fc2_b.unsqueeze(0), device)


def _dinov2_attention(x, p: _BlockParams, num_heads: int):
    hidden_states = ttnn.layer_norm(x, weight=p.norm1_w, bias=p.norm1_b, epsilon=1e-6)
    head_dim = hidden_states.shape[-1] // num_heads
    qkv = ttnn.linear(
        hidden_states, p.qkv_w, bias=p.qkv_b, core_grid=_CORE_GRID, compute_kernel_config=_LOFI
    )
    ttnn.deallocate(hidden_states)
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=num_heads, transpose_key=False
    )
    ttnn.deallocate(qkv)
    ctx = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=1.0 / (head_dim**0.5)
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    ctx = ttnn.transformer.concatenate_heads(ctx)

    out = ttnn.linear(ctx, p.proj_w, bias=p.proj_b, core_grid=_CORE_GRID, compute_kernel_config=_LOFI)
    ttnn.deallocate(ctx)
    return ttnn.add(x, out)


def _dinov2_mlp(x, p: _BlockParams):
    hidden_states = ttnn.layer_norm(x, weight=p.norm2_w, bias=p.norm2_b, epsilon=1e-6)
    h = ttnn.linear(
        hidden_states, p.fc1_w, bias=p.fc1_b, activation="gelu",
        core_grid=_CORE_GRID, compute_kernel_config=_LOFI,
    )
    ttnn.deallocate(hidden_states)
    h2 = ttnn.linear(h, p.fc2_w, bias=p.fc2_b, core_grid=_CORE_GRID, compute_kernel_config=_LOFI)
    ttnn.deallocate(h)
    return ttnn.add(x, h2)


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
    """Gaze-LLE inference entirely on a single Blackhole p150a chip (see module docstring)."""

    def __init__(self, ref_model, device, inout: bool = True):
        self.device = device
        self.inout = inout
        self.ref = ref_model
        backbone = ref_model.backbone
        self.cfg = backbone.cfg
        self.num_heads = self.cfg.num_heads
        self.embed_dim = self.cfg.embed_dim
        self.num_reg_tokens = self.cfg.num_register_tokens
        self.patch_size = backbone.patch_size
        self.img_size = backbone.img_size
        self.num_patches_side = self.img_size // self.patch_size
        self.num_patches = self.num_patches_side ** 2
        self.dim = ref_model.dim  # decoder hidden 256
        self.featmap_h = ref_model.featmap_h
        self.featmap_w = ref_model.featmap_w
        self.out_size = ref_model.out_size

        self.block_params = [_BlockParams(blk, device) for blk in backbone.blocks]
        self.final_norm_w = _to_device(backbone.norm.weight.unsqueeze(0), device)
        self.final_norm_b = _to_device(backbone.norm.bias.unsqueeze(0), device)

        # --- Patch embed as a single on-device matmul. The equivalent torch op is
        # Conv2d(3, embed_dim, k=patch_size, s=patch_size, padding=0). We flatten each
        # patch to (patch_size*patch_size*3) features and run one (588, 768) matmul.
        # Host pre-reshapes the input tensor (pure layout, not inference compute).
        pe_w = backbone.patch_embed_proj.weight.detach()  # (embed_dim, 3, ps, ps)
        w_flat = pe_w.permute(2, 3, 1, 0).reshape(-1, self.embed_dim).contiguous()  # (ps*ps*3, embed_dim)
        self.patch_embed_w = _to_device(w_flat, device, dtype=ttnn.bfloat8_b)
        self.patch_embed_b = _to_device(backbone.patch_embed_proj.bias.detach().unsqueeze(0), device)

        # --- Pre-composed [CLS + pos_cls, REG] prefix + standalone pos_patches.
        cls_tok = backbone.cls_token.detach()
        reg_tok = backbone.reg_token.detach()
        pos_cls = backbone.pos_embed[:, :1].detach()
        pos_patches = backbone.pos_embed[:, 1:].detach().contiguous()
        prefix = torch.cat([cls_tok + pos_cls, reg_tok], dim=1).contiguous()
        self.prefix_tt = _to_device(prefix, device)
        self.pos_patches_tt = _to_device(pos_patches, device)

        # --- Gaze decoder 1x1 projection (768 -> 256) and on-device pos_embed + head_token.
        linear_w = ref_model.linear.weight.squeeze(-1).squeeze(-1).T.contiguous()
        self.proj_w = _to_device(linear_w, device)
        self.proj_b = _to_device(ref_model.linear.bias.unsqueeze(0), device)

        pe = ref_model.pos_embed.permute(1, 2, 0).reshape(1, -1, self.dim).contiguous()
        self.gaze_pos_embed_tt = _to_device(pe, device)
        self.head_token_tt = _to_device(ref_model.head_token.weight.unsqueeze(0), device)  # (1, 1, 256)

        if inout:
            self.inout_token = _to_device(ref_model.inout_token.weight.unsqueeze(0), device)
            # Reference inout_head = Sequential(Linear 256->128, ReLU, Linear 128->1, Sigmoid)
            self.inout_fc1_w = _to_device(ref_model.inout_head[0].weight.T.contiguous(), device)
            self.inout_fc1_b = _to_device(ref_model.inout_head[0].bias.unsqueeze(0), device)
            self.inout_fc2_w = _to_device(ref_model.inout_head[2].weight.T.contiguous(), device)
            self.inout_fc2_b = _to_device(ref_model.inout_head[2].bias.unsqueeze(0), device)

        # --- Fused heatmap head. ConvTranspose2d(k=2, s=2) bias + Conv2d(256->1, k=1, bias=False)
        # is algebraically equivalent to a single per-pixel matmul with weight shape
        # (256, 2, 2) and a scalar bias. We emit a (256, 4) matmul and reshape afterwards.
        ct_w = ref_model.heatmap_head[0].weight.detach()  # (in=256, out=256, kH=2, kW=2)
        ct_b = ref_model.heatmap_head[0].bias.detach()
        c1_w = ref_model.heatmap_head[1].weight.detach().squeeze(-1).squeeze(-1)  # (1, 256)
        w_fused = torch.einsum('ko,ioab->ikab', c1_w, ct_w).squeeze(1)  # (256, 2, 2)
        b_fused = (c1_w @ ct_b).squeeze().item()  # scalar
        w_fused_2d = w_fused.reshape(self.dim, 4).contiguous()
        self.heatmap_w = _to_device(w_fused_2d, device)
        # Broadcast-add scalar bias by uploading a (1, 1, 1) constant.
        self.heatmap_b_tt = _to_device(torch.full((1, 1, 1), float(b_fused)), device)

        self.gaze_block_params = [_GazeBlockParams(blk, device) for blk in ref_model.transformer]

    @staticmethod
    def _reshape_image_for_matmul(images: torch.Tensor, num_patches_side: int, patch_size: int) -> torch.Tensor:
        """(B, 3, H, W) → (B, num_patches, ps*ps*3). Pure layout (permute+reshape)."""
        b = images.shape[0]
        n = num_patches_side
        ps = patch_size
        return (
            images.view(b, 3, n, ps, n, ps)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b, n * n, ps * ps * 3)
            .contiguous()
        )

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, bboxes: List[Sequence[float]]):
        ref = self.ref
        b = images.shape[0]
        assert b == 1, "TtGazeLLE currently supports B=1 only"

        # ---- 1. Upload pre-patched image. A single pure-layout permute+reshape on host
        # turns the (B, 3, H, W) image into (B, num_patches, ps*ps*3) so the device side
        # is a clean matmul with no fold / layout-conversion overhead.
        patches_host = self._reshape_image_for_matmul(images, self.num_patches_side, self.patch_size)
        patches_tt = _to_device(patches_host, self.device)

        # ---- 2. Patch embed (on device): (B, N, 588) @ (588, 768) + bias.
        patches_tt = ttnn.linear(
            patches_tt,
            self.patch_embed_w,
            bias=self.patch_embed_b,
            core_grid=_CORE_GRID,
            compute_kernel_config=_LOFI,
        )

        # ---- 3. Add DINOv2 pos_embed for patch tokens, then prepend [CLS+pos_cls, REG].
        patches_tt = ttnn.add(patches_tt, self.pos_patches_tt)
        x_tt = ttnn.concat([self.prefix_tt, patches_tt], dim=1)
        ttnn.deallocate(patches_tt)

        # ---- 4. DINOv2 backbone.
        for bp in self.block_params:
            x_tt = _dinov2_block(x_tt, bp, self.num_heads)
        x_tt = ttnn.layer_norm(x_tt, weight=self.final_norm_w, bias=self.final_norm_b, epsilon=1e-6)

        # ---- 5. Drop CLS + register tokens on device.
        total_prefix = 1 + self.num_reg_tokens
        shp = x_tt.shape
        feat_tt = ttnn.slice(x_tt, [0, total_prefix, 0], [shp[0], shp[1], shp[2]])
        ttnn.deallocate(x_tt)

        # ---- 6. Gaze decoder: project 768→256, add pos_embed, multiply head_map by head_token.
        x_tt = ttnn.linear(feat_tt, self.proj_w, bias=self.proj_b, core_grid=_CORE_GRID)
        ttnn.deallocate(feat_tt)
        x_tt = ttnn.add(x_tt, self.gaze_pos_embed_tt)

        head_map = torch.stack([ref._bbox_to_head_map(bb) for bb in bboxes]).view(b, -1, 1).contiguous()
        head_map_tt = _to_device(head_map, self.device)
        head_contrib = ttnn.mul(head_map_tt, self.head_token_tt)  # broadcast → (B, 1024, 256)
        ttnn.deallocate(head_map_tt)
        x_tt = ttnn.add(x_tt, head_contrib)
        ttnn.deallocate(head_contrib)

        # ---- 7. Prepend in/out token, run 3 gaze blocks, split outputs.
        if self.inout:
            x_tt = ttnn.concat([self.inout_token, x_tt], dim=1)

        for gp in self.gaze_block_params:
            x_tt = _gaze_block(x_tt, gp, num_heads=8)

        inout_preds_tt = None
        if self.inout:
            seq = x_tt.shape[1]
            inout_tok = ttnn.slice(x_tt, [0, 0, 0], [b, 1, self.dim])
            patch_out = ttnn.slice(x_tt, [0, 1, 0], [b, seq, self.dim])
            ttnn.deallocate(x_tt)

            # ---- 8a. In/out head on device: Linear+ReLU → Linear → Sigmoid.
            h = ttnn.linear(
                inout_tok, self.inout_fc1_w, bias=self.inout_fc1_b, activation="relu"
            )
            ttnn.deallocate(inout_tok)
            h = ttnn.linear(h, self.inout_fc2_w, bias=self.inout_fc2_b)
            inout_preds_tt = ttnn.sigmoid(h)
        else:
            patch_out = x_tt

        # ---- 8b. Fused heatmap head: (B,1024,256) @ (256,4) + scalar bias + sigmoid.
        hm = ttnn.linear(patch_out, self.heatmap_w, bias=None, core_grid=_CORE_GRID)
        ttnn.deallocate(patch_out)
        hm = ttnn.add(hm, self.heatmap_b_tt)
        hm = ttnn.sigmoid(hm)

        # ---- 9. Download the small outputs. The (B, 1024, 4) → (B, 64, 64) re-interleave is
        # a pure view on contiguous memory.
        heatmap_compact = ttnn.to_torch(hm).to(torch.float32)  # (B, 1024, 4)
        ttnn.deallocate(hm)
        heatmap = (
            heatmap_compact.view(b, self.featmap_h, self.featmap_w, 2, 2)
            .permute(0, 1, 3, 2, 4)
            .reshape(b, self.featmap_h * 2, self.featmap_w * 2)
        )
        if (self.featmap_h * 2, self.featmap_w * 2) != self.out_size:
            heatmap = F.interpolate(
                heatmap.unsqueeze(1), size=self.out_size, mode="bilinear", align_corners=False
            ).squeeze(1)

        inout_preds = None
        if self.inout:
            inout_preds = ttnn.to_torch(inout_preds_tt).to(torch.float32).reshape(b)
            ttnn.deallocate(inout_preds_tt)

        return {"heatmap": heatmap, "inout": inout_preds}
