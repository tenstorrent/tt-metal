# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference helpers for Swin-L submodule PCC tests.
Loads mmdet backbone and provides per-submodule forward functions
that return (input, output) pairs for comparison with TTNN.

All submodule helpers return tensors in NHWC [B, H, W, C] format
to match TTNN conventions (except where noted).

Standalone — works with any mmdet checkpoint containing a Swin-L backbone.
"""

from typing import List, Tuple

import torch


def _load_backbone(config_path: str, checkpoint_path: str):
    """Load mmdet model and return backbone (eval mode, no grad)."""
    from mmdet.apis import init_detector

    model = init_detector(str(config_path), str(checkpoint_path), device="cpu")
    model.eval()
    return model.backbone


class SwinLReference:
    """
    Wraps mmdet Swin-L backbone and exposes per-submodule forward
    for PCC testing against TTNN.
    """

    def __init__(self, config_path: str, checkpoint_path: str):
        self.backbone = _load_backbone(config_path, checkpoint_path)

    @torch.no_grad()
    def get_patch_embed_output(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Run patch embedding.
        Input: [B, 3, H, W]
        Returns: [B, H', W', C] in NHWC format, and (H', W') tuple.
        """
        out, hw = self.backbone.patch_embed(x)  # [B, H*W, C]
        B, N, C = out.shape
        H, W = hw
        return out.view(B, H, W, C), (H, W)

    @torch.no_grad()
    def get_stage_input(self, x: torch.Tensor, target_stage: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Run backbone up to (but not including) target_stage.
        Returns the input tensor for that stage in NHWC [B, H, W, C] format.
        """
        out, hw = self.backbone.patch_embed(x)
        out = self.backbone.drop_after_pos(out)
        H, W = hw

        for s in range(target_stage):
            stage = self.backbone.stages[s]
            out, hw, _, _ = stage(out, hw)
            H, W = hw

        B, N, C = out.shape
        return out.view(B, H, W, C), (H, W)

    @torch.no_grad()
    def forward_attention(
        self, x_nhwc: torch.Tensor, hw: Tuple[int, int], stage_idx: int, block_idx: int
    ) -> torch.Tensor:
        """
        Run one attention sublayer (norm1 -> ShiftWindowMSA) on input.
        Input/output: [B, H, W, C] NHWC.
        """
        B, H, W, C = x_nhwc.shape
        x_flat = x_nhwc.view(B, H * W, C)
        blk = self.backbone.stages[stage_idx].blocks[block_idx]
        normed = blk.norm1(x_flat)
        attn_out = blk.attn(normed, hw)
        return attn_out.view(B, H, W, C)

    @torch.no_grad()
    def forward_ffn(self, x_nhwc: torch.Tensor, hw: Tuple[int, int], stage_idx: int, block_idx: int) -> torch.Tensor:
        """
        Run one FFN sublayer (norm2 -> FFN layers only, no identity) on input.
        Note: mmdet FFN.forward() adds an internal residual (identity + layers(x)).
        We call .layers directly so this tests just the two linears + GELU,
        matching our TTNN TtSwinMLP which also has no internal residual.
        Input/output: [B, H, W, C] NHWC.
        """
        B, H, W, C = x_nhwc.shape
        x_flat = x_nhwc.view(B, H * W, C)
        blk = self.backbone.stages[stage_idx].blocks[block_idx]
        normed = blk.norm2(x_flat)
        ffn_out = blk.ffn.layers(normed)  # layers only, skip identity addition
        return ffn_out.view(B, H, W, C)

    @torch.no_grad()
    def forward_block(self, x_nhwc: torch.Tensor, hw: Tuple[int, int], stage_idx: int, block_idx: int) -> torch.Tensor:
        """
        Run one full Swin block via the actual mmdet block forward.
        Input/output: [B, H, W, C] NHWC.
        """
        B, H, W, C = x_nhwc.shape
        x_flat = x_nhwc.view(B, H * W, C)
        blk = self.backbone.stages[stage_idx].blocks[block_idx]
        out = blk(x_flat, hw)
        return out.view(B, H, W, C)

    @torch.no_grad()
    def forward_patch_merge(
        self, x_nhwc: torch.Tensor, hw: Tuple[int, int], stage_idx: int
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Run patch merging (downsample) at end of stage_idx.
        Input: [B, H, W, C] NHWC.
        Returns: [B, H', W', C'] NHWC, (H', W').
        """
        B, H, W, C = x_nhwc.shape
        x_flat = x_nhwc.view(B, H * W, C)
        ds = self.backbone.stages[stage_idx].downsample
        out, new_hw = ds(x_flat, hw)
        nH, nW = new_hw
        return out.view(B, nH, nW, out.shape[-1]), new_hw

    @torch.no_grad()
    def forward_backbone(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Full backbone forward. Returns list of 4 NCHW feature maps.
        """
        feats = self.backbone(x)
        return list(feats)
