# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN end-to-end DINO-5scale Swin-L model.

Full pipeline: Image → Backbone → Neck → Pre-Transformer → Encoder →
               Pre-Decoder → Decoder → Heads → Post-Processing (NMS)

Usage:
    model = TtDINO(backbone_params, neck_params, encoder_params, decoder_params, device)
    detections = model.forward_image(image_tensor)  # [B, 3, H, W] float32
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import ttnn
from loguru import logger

from models.experimental.swin_l.tt.tt_backbone import TtSwinLBackbone
from models.experimental.dino_5scale_swin_l.tt.tt_neck import TtDINONeck
from models.experimental.dino_5scale_swin_l.tt.tt_encoder import TtDINOEncoder
from models.experimental.dino_5scale_swin_l.tt.tt_decoder import (
    TtDINODecoder,
    TtRegBranch,
    inverse_sigmoid_ttnn,
)

# Tile alignment for ttnn.concat (TILE layout requires aligned dims)
TILE_ALIGN = 32


def _tile_aligned_hw(H: int, W: int) -> int:
    """Return padded H*W: tile-aligned and divisible by H (for encoder value reshape)."""
    hw = H * W
    padded = (hw + TILE_ALIGN - 1) // TILE_ALIGN * TILE_ALIGN
    if padded % H != 0:
        padded = ((hw + H - 1) // H) * H
        padded = (padded + TILE_ALIGN - 1) // TILE_ALIGN * TILE_ALIGN
    return padded


def sine_positional_encoding(
    H: int,
    W: int,
    num_feats: int = 128,
    temperature: float = 20,
    normalize: bool = True,
    scale: float = 2.0 * math.pi,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute 2D sine positional encoding for a single feature map.
    Matches mmdet SinePositionalEncoding with mask=None.

    Returns: [1, 256, H, W] torch float32 tensor.
    """
    x_embed = torch.arange(1, W + 1, dtype=torch.float32).view(1, 1, -1).repeat(1, H, 1)
    y_embed = torch.arange(1, H + 1, dtype=torch.float32).view(1, -1, 1).repeat(1, 1, W)

    if normalize:
        y_embed = (y_embed) / (y_embed[:, -1:, :] + eps) * scale
        x_embed = (x_embed) / (x_embed[:, :, -1:] + eps) * scale
    dim_t = torch.arange(num_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(1, H, W, -1)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(1, H, W, -1)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos


def sine_positional_encoding_ttnn(
    device,
    H: int,
    W: int,
    num_feats: int = 128,
    temperature: float = 20,
    scale: float = 2.0 * math.pi,
    eps: float = 1e-6,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    dim_t_cache=None,
):
    x_embed = ttnn.arange(1, W + 1, 1, dtype=dtype, device=device, memory_config=memory_config)
    x_embed = ttnn.reshape(x_embed, (1, 1, W))
    x_embed = ttnn.multiply(ttnn.divide(x_embed, W + eps), scale)
    x_embed = ttnn.repeat(x_embed, (1, H, 1))
    y_embed = ttnn.arange(1, H + 1, 1, dtype=dtype, device=device, memory_config=memory_config)
    y_embed = ttnn.reshape(y_embed, (1, H, 1))
    y_embed = ttnn.multiply(ttnn.divide(y_embed, H + eps), scale)
    y_embed = ttnn.repeat(y_embed, (1, 1, W))
    if dim_t_cache is not None:
        dim_t = dim_t_cache
    else:
        dim_t = ttnn.arange(0, num_feats, 1, dtype=dtype, device=device, memory_config=memory_config)
        dim_t = ttnn.floor(ttnn.divide(dim_t, 2.0, memory_config=memory_config))
        dim_t = ttnn.multiply(dim_t, 2.0, memory_config=memory_config)
        dim_t = ttnn.divide(dim_t, float(num_feats), memory_config=memory_config)
        dim_t = ttnn.pow(float(temperature), dim_t, memory_config=memory_config)
    x_embed = ttnn.reshape(x_embed, (1, H, W, 1))
    pos_x = ttnn.divide(x_embed, dim_t)
    y_embed = ttnn.reshape(y_embed, (1, H, W, 1))
    pos_y = ttnn.divide(y_embed, dim_t)
    sin_x = ttnn.sin(pos_x[:, :, :, 0::2])
    cos_x = ttnn.cos(pos_x[:, :, :, 1::2])
    sin_x = ttnn.unsqueeze(sin_x, -1)
    cos_x = ttnn.unsqueeze(cos_x, -1)
    pos_x = ttnn.concat([sin_x, cos_x], dim=-1)
    pos_x = ttnn.reshape(pos_x, (1, H, W, num_feats))
    sin_y = ttnn.sin(pos_y[:, :, :, 0::2])
    cos_y = ttnn.cos(pos_y[:, :, :, 1::2])
    sin_y = ttnn.unsqueeze(sin_y, -1)
    cos_y = ttnn.unsqueeze(cos_y, -1)
    pos_y = ttnn.concat([sin_y, cos_y], dim=-1)
    pos_y = ttnn.reshape(pos_y, (1, H, W, num_feats))
    pos = ttnn.concat([pos_y, pos_x], dim=-1)
    pos = ttnn.permute(pos, (0, 3, 1, 2))
    pos = ttnn.reshape(pos, (1, num_feats * 2, H * W))
    pos = ttnn.permute(pos, (0, 2, 1))
    return ttnn.to_layout(pos, ttnn.ROW_MAJOR_LAYOUT)


def _linspace_ttnn(start, end, steps, device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    if steps <= 1:
        return ttnn.full((1,), start, dtype=dtype, device=device, memory_config=memory_config)
    idx = ttnn.arange(0, steps, dtype=dtype, device=device, memory_config=memory_config)
    step_size = (end - start) / (steps - 1)
    return ttnn.add(ttnn.multiply(idx, step_size), start)


def gen_encoder_output_proposals_ttnn(
    device,
    memory_tt: ttnn.Tensor,
    spatial_shapes,
    B: int,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    hw_list = spatial_shapes.tolist() if hasattr(spatial_shapes, "tolist") else list(spatial_shapes)
    proposals_list = []
    for lvl, hw in enumerate(hw_list):
        H, W = int(hw[0]), int(hw[1])
        if H * W == 0:
            continue
        grid_x = _linspace_ttnn(0, W - 1, W, device, memory_config=memory_config)
        grid_x = ttnn.reshape(grid_x, (1, W))
        grid_x = ttnn.add(grid_x, 0.5)
        grid_x = ttnn.divide(grid_x, float(W))
        grid_x = ttnn.repeat(grid_x, (H, 1))
        grid_x = ttnn.reshape(grid_x, (1, H * W))
        grid_y = _linspace_ttnn(0, H - 1, H, device, memory_config=memory_config)
        grid_y = ttnn.reshape(grid_y, (H, 1))
        grid_y = ttnn.add(grid_y, 0.5)
        grid_y = ttnn.divide(grid_y, float(H))
        grid_y = ttnn.repeat(grid_y, (1, W))
        grid_y = ttnn.reshape(grid_y, (1, H * W))
        wh_val = 0.05 * (2.0**lvl)
        wh = ttnn.full((1, H * W), wh_val, dtype=ttnn.bfloat16, device=device, memory_config=memory_config)
        level_proposal = ttnn.stack([grid_x, grid_y, wh, wh], dim=-1)
        level_proposal = ttnn.repeat(level_proposal, (B, 1, 1))
        proposals_list.append(level_proposal)
    output_proposals = ttnn.concat(proposals_list, dim=1)
    for t in proposals_list:
        ttnn.deallocate(t)
    gt_lo = ttnn.gt(output_proposals, 0.01)
    lt_hi = ttnn.lt(output_proposals, 0.99)
    both = ttnn.logical_and(gt_lo, lt_hi)
    valid_sum = ttnn.sum(both, dim=-1, keepdim=True, memory_config=memory_config)
    four = ttnn.full(
        (output_proposals.shape[0], output_proposals.shape[1], 1),
        4.0,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=memory_config,
    )
    output_proposals_valid = ttnn.eq(valid_sum, four)
    ttnn.deallocate(gt_lo)
    ttnn.deallocate(lt_hi)
    ttnn.deallocate(both)
    ttnn.deallocate(valid_sum)
    ttnn.deallocate(four)
    one_minus = ttnn.add(ttnn.neg(output_proposals), 1.0)
    ratio = ttnn.divide(output_proposals, ttnn.clamp(one_minus, min=1e-6))
    ttnn.deallocate(one_minus)
    log_proposals = ttnn.log(ttnn.clamp(ratio, min=1e-6))
    ttnn.deallocate(ratio)
    inf_tensor = ttnn.full(log_proposals.shape, 1e4, dtype=ttnn.bfloat16, device=device, memory_config=memory_config)
    inf_broadcast = ttnn.where(output_proposals_valid, log_proposals, inf_tensor)
    ttnn.deallocate(log_proposals)
    ttnn.deallocate(inf_tensor)
    return inf_broadcast, output_proposals_valid


def gen_encoder_output_proposals(
    memory_torch: torch.Tensor,
    spatial_shapes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = memory_torch.shape[0]
    proposals = []
    for lvl, (H, W) in enumerate(spatial_shapes.tolist()):
        H, W = int(H), int(W)
        scale = torch.tensor([[W, H]], dtype=torch.float32).view(1, 1, 1, 2)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H - 1, H, dtype=torch.float32),
            torch.linspace(0, W - 1, W, dtype=torch.float32),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
        proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
        proposals.append(proposal)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).sum(-1, keepdim=True) == 4
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))
    return output_proposals, output_proposals_valid


class TtDINO:
    """
    Full DINO-5scale Swin-L model for inference on Tenstorrent hardware.

    Complete pipeline: Image → Backbone → Neck → Encoder → Decoder → Heads → NMS.

    Args:
        backbone_params: weights from load_backbone_weights (or None to skip backbone)
        neck_params: weights from load_neck_weights (or None to skip neck)
        encoder_params: weights from load_encoder_weights
        decoder_params: weights from load_decoder_weights
        device: ttnn device
        attn_masks: precomputed shifted-window attention masks (from compute_attn_masks)
    """

    def __init__(
        self,
        encoder_params: dict,
        decoder_params: dict,
        device: ttnn.Device,
        backbone_params: Optional[dict] = None,
        neck_params: Optional[dict] = None,
        attn_masks: Optional[list] = None,
        num_queries: int = 900,
        num_classes: int = 80,
        num_levels: int = 5,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        encoder_num_layers: int = 6,
        decoder_num_layers: int = 6,
        pe_temperature: float = 20,
        embed_dim: int = 192,
        depths: Tuple[int, ...] = (2, 2, 18, 2),
        backbone_num_heads: Tuple[int, ...] = (6, 12, 24, 48),
        window_size: int = 12,
        in_channels: Tuple[int, ...] = (192, 384, 768, 1536),
    ):
        self.device = device
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.embed_dims = embed_dims
        self.pe_temperature = pe_temperature
        self.decoder_num_layers = decoder_num_layers

        # --- Backbone (optional) ---
        self.backbone = None
        if backbone_params is not None:
            self.backbone = TtSwinLBackbone(
                device,
                backbone_params,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=backbone_num_heads,
                window_size=window_size,
                attn_masks=attn_masks,
            )

        # --- Neck (optional) ---
        self.neck = None
        if neck_params is not None:
            self.neck = TtDINONeck(device, neck_params, in_channels=in_channels)

        # --- Encoder ---
        self.encoder = TtDINOEncoder(
            encoder_params,
            device,
            num_layers=encoder_num_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )

        self.level_embed = encoder_params["level_embed"]

        # --- Decoder ---
        self.decoder = TtDINODecoder(
            decoder_params,
            device,
            num_layers=decoder_num_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )

        pd = decoder_params["_torch_pre_decoder"]
        self.memory_trans_fc_w_torch = pd["memory_trans_fc_w"]
        self.memory_trans_fc_b_torch = pd["memory_trans_fc_b"]
        self.memory_trans_norm_w_torch = pd["memory_trans_norm_w"]
        self.memory_trans_norm_b_torch = pd["memory_trans_norm_b"]
        self.query_embedding_torch = pd["query_embedding"]
        self.cls_enc_w_torch = pd["cls_enc_w"]
        self.cls_enc_b_torch = pd["cls_enc_b"]
        self.reg_enc_layers_torch = [(layer["weight"], layer["bias"]) for layer in pd["reg_enc_layers"]]

        self._decoder_params = decoder_params
        self.cls_branches = decoder_params["cls_branches"]
        self.reg_branches_head = [
            TtRegBranch(decoder_params["reg_branches"][i], device) for i in range(decoder_num_layers)
        ]

        self._dram_cfg = ttnn.DRAM_MEMORY_CONFIG

    def pre_transformer_tt(
        self,
        mlvl_feats_tt: List[ttnn.Tensor],
    ) -> Dict[str, Any]:
        """
        Flatten and positional encoding on device; concat on device with padding
        so that each level's H*W is tile-aligned (ttnn.concat requires TILE).
        Slices out valid [B, N, 256] before returning so encoder/pre_decoder see unpadded sequence.

        Returns feat_flatten, feat_pos as ttnn tensors [B, N, 256]; level_start_index is unpadded.
        """
        logger.info("Pre-transformer (TT): flatten + pad + PE + concat on device...")
        sh0 = tuple(mlvl_feats_tt[0].shape)
        B, C = int(sh0[0]), int(sh0[1])
        feat_padded_list: List[ttnn.Tensor] = []
        pos_padded_list: List[ttnn.Tensor] = []
        spatial_shapes_list: List[List[int]] = []
        padded_hw_list: List[int] = []
        hw_list: List[int] = []

        for lvl, feat_tt in enumerate(mlvl_feats_tt):
            sh = tuple(feat_tt.shape)
            H, W = int(sh[2]), int(sh[3])
            hw = H * W
            padded_hw = _tile_aligned_hw(H, W)
            spatial_shapes_list.append([H, W])
            padded_hw_list.append(padded_hw)
            hw_list.append(hw)

            # Flatten on device: (B, C, H, W) -> (B, H*W, C)
            feat_tt = ttnn.to_layout(feat_tt, ttnn.ROW_MAJOR_LAYOUT)
            feat_tt = ttnn.permute(feat_tt, (0, 2, 3, 1))
            feat_flat_tt = ttnn.reshape(feat_tt, (B, hw, C))
            if padded_hw > hw:
                pad_count = padded_hw - hw
                zeros_tt = ttnn.zeros(
                    (B, pad_count, C),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=self._dram_cfg,
                )
                feat_flat_tt = ttnn.concat([feat_flat_tt, zeros_tt], dim=1)
                ttnn.deallocate(zeros_tt)
            feat_flat_tt = ttnn.to_layout(feat_flat_tt, ttnn.TILE_LAYOUT)
            feat_padded_list.append(feat_flat_tt)

            # PE on device: [1, H*W, 256] + level_embed, then pad and expand to B
            pos_tt = sine_positional_encoding_ttnn(
                self.device,
                int(H),
                int(W),
                num_feats=self.embed_dims // 2,
                temperature=self.pe_temperature,
                memory_config=self._dram_cfg,
            )
            level_slice = self.level_embed[lvl : lvl + 1, :]  # [1, 256]
            pos_tt = ttnn.add(pos_tt, level_slice, memory_config=self._dram_cfg)
            if padded_hw > hw:
                pad_count = padded_hw - hw
                zeros_pos = ttnn.zeros(
                    (1, pad_count, C),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=self._dram_cfg,
                )
                pos_tt = ttnn.concat([pos_tt, zeros_pos], dim=1)
                ttnn.deallocate(zeros_pos)
            pos_tt = ttnn.repeat(pos_tt, (B, 1, 1))
            pos_tt = ttnn.to_layout(pos_tt, ttnn.TILE_LAYOUT)
            pos_padded_list.append(pos_tt)

        # Concat on device (all levels are tile-aligned)
        feat_flatten_padded = ttnn.concat(feat_padded_list, dim=1)
        feat_pos_padded = ttnn.concat(pos_padded_list, dim=1)
        for t in feat_padded_list:
            ttnn.deallocate(t)
        for t in pos_padded_list:
            ttnn.deallocate(t)

        N = sum(hw_list)
        N_padded = sum(padded_hw_list)
        spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long)
        # Slice out valid [B, N, C] so encoder/pre_decoder see unpadded sequence
        padded_starts = [0] + list(torch.tensor(padded_hw_list, dtype=torch.long).cumsum(0)[:-1].tolist())
        if N_padded > N:
            feat_segments = []
            pos_segments = []
            for lvl in range(len(hw_list)):
                start = padded_starts[lvl]
                end = start + hw_list[lvl]
                feat_segments.append(ttnn.slice(feat_flatten_padded, [0, start, 0], [B, end, C]))
                pos_segments.append(ttnn.slice(feat_pos_padded, [0, start, 0], [B, end, C]))
            feat_flatten_tt = ttnn.concat(feat_segments, dim=1)
            feat_pos_tt = ttnn.concat(pos_segments, dim=1)
            ttnn.deallocate(feat_flatten_padded)
            ttnn.deallocate(feat_pos_padded)
            for t in feat_segments:
                ttnn.deallocate(t)
            for t in pos_segments:
                ttnn.deallocate(t)
        else:
            feat_flatten_tt = feat_flatten_padded
            feat_pos_tt = feat_pos_padded

        level_start_index = torch.cat(
            [
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            ]
        )
        valid_ratios = torch.ones(B, len(mlvl_feats_tt), 2, dtype=torch.float32)

        logger.info(f"Pre-transformer (TT): feat_flatten [B={B}, N={N}], " f"spatial_shapes {spatial_shapes.tolist()}")
        return {
            "feat_flatten": feat_flatten_tt,
            "feat_pos": feat_pos_tt,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
        }

    def pre_transformer(
        self,
        mlvl_feats: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Process multi-scale feature maps before encoder.

        Generates sine positional encodings, adds level embeddings, flattens
        features, and computes spatial metadata.

        Args:
            mlvl_feats: list of 5 NCHW torch tensors from neck [B, 256, H_i, W_i]

        Returns dict with:
            feat_flatten: torch [B, N, 256]
            feat_pos: torch [B, N, 256] (positional encoding + level embed)
            spatial_shapes: torch [num_levels, 2]
            level_start_index: torch [num_levels]
            valid_ratios: torch [B, num_levels, 2]
        """
        logger.info("Pre-transformer: generating positional encodings...")

        # Level embed to torch for host computation
        level_embed_torch = ttnn.to_torch(self.level_embed).float()  # [num_levels, 256]

        feat_flatten_list = []
        lvl_pos_embed_list = []
        spatial_shapes_list = []

        for lvl, feat in enumerate(mlvl_feats):
            B, C, H, W = feat.shape
            spatial_shapes_list.append([H, W])

            feat_flat = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            feat_flatten_list.append(feat_flat)

            pos_embed = sine_positional_encoding(
                H,
                W,
                num_feats=self.embed_dims // 2,
                temperature=self.pe_temperature,
            )
            pos_flat = pos_embed.flatten(2).permute(0, 2, 1)  # [1, H*W, 256]
            lvl_pos_embed = pos_flat + level_embed_torch[lvl].view(1, 1, -1)
            lvl_pos_embed_list.append(lvl_pos_embed.expand(B, -1, -1))

        feat_flatten = torch.cat(feat_flatten_list, dim=1)
        feat_pos = torch.cat(lvl_pos_embed_list, dim=1)
        spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long)
        level_start_index = torch.cat(
            [
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            ]
        )
        valid_ratios = feat_flatten.new_ones(B, len(mlvl_feats), 2)

        logger.info(f"Pre-transformer: feat_flatten {feat_flatten.shape}, " f"spatial_shapes {spatial_shapes.tolist()}")

        return {
            "feat_flatten": feat_flatten,
            "feat_pos": feat_pos,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
        }

    def pre_decoder(
        self,
        memory_tt: ttnn.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Query selection: generate proposals from encoder memory, select top-K.

        Entire pipeline runs on HOST in float32 to match the PyTorch reference
        exactly. This is critical because the top-K operation (selecting 900 from
        ~89K proposals) is extremely sensitive to small score differences — even
        bfloat16 rounding in the cls_branch linear can cause different queries.

        Pipeline (all host, float32):
          1. gen_encoder_output_proposals → proposal boxes
          2. memory_trans_fc + memory_trans_norm → transformed memory
          3. cls_branches[6] → classification scores → top-K selection
          4. reg_branches[6] + proposals → reference points
          5. query_embedding → initial query content
        """
        logger.info("Pre-decoder: generating encoder output proposals (host float32)...")
        memory_torch = ttnn.to_torch(memory_tt).float()
        N = spatial_shapes.prod(1).sum().item()
        memory_torch = memory_torch[:, :N, :]
        bs = memory_torch.shape[0]

        output_proposals, output_proposals_valid = gen_encoder_output_proposals(
            memory_torch,
            spatial_shapes,
        )

        output_memory = memory_torch.masked_fill(~output_proposals_valid, 0.0)

        # memory_trans_fc + memory_trans_norm — host float32
        logger.info("Pre-decoder: memory_trans_fc + norm (host float32)...")
        output_memory = torch.nn.functional.linear(
            output_memory,
            self.memory_trans_fc_w_torch,
            self.memory_trans_fc_b_torch,
        )
        output_memory = torch.nn.functional.layer_norm(
            output_memory,
            [self.embed_dims],
            weight=self.memory_trans_norm_w_torch,
            bias=self.memory_trans_norm_b_torch,
        )

        # cls_branches[6] — host float32: [B, N, 256] → [B, N, 80]
        logger.info("Pre-decoder: cls_branches[6] scoring (host float32)...")
        enc_cls = torch.nn.functional.linear(
            output_memory,
            self.cls_enc_w_torch,
            self.cls_enc_b_torch,
        )

        # reg_branches[6] — host float32: [B, N, 256] → [B, N, 4]
        logger.info("Pre-decoder: reg_branches[6] proposals (host float32)...")
        reg_out = output_memory
        for i, (w, b) in enumerate(self.reg_enc_layers_torch):
            reg_out = torch.nn.functional.linear(reg_out, w, b)
            if i < len(self.reg_enc_layers_torch) - 1:
                reg_out = torch.nn.functional.relu(reg_out)

        enc_coords_unact = reg_out + output_proposals

        # top-K selection — host float32
        logger.info("Pre-decoder: top-K selection (K=%d, host float32)...", self.num_queries)
        topk_indices = torch.topk(
            enc_cls.max(-1)[0],
            k=self.num_queries,
            dim=1,
        )[1]

        topk_coords_unact = torch.gather(
            enc_coords_unact,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4),
        )
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()

        query = self.query_embedding_torch[:, None, :].repeat(1, bs, 1).transpose(0, 1)
        query_tt = ttnn.from_torch(
            query.to(torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
        )

        logger.info(f"Pre-decoder: selected {self.num_queries} queries, " f"reference_points {reference_points.shape}")

        return {
            "query": query_tt,
            "reference_points": reference_points,
            "topk_score": torch.gather(
                enc_cls,
                1,
                topk_indices.unsqueeze(-1).repeat(1, 1, self.num_classes),
            ),
            "topk_coords": reference_points,
            "topk_indices": topk_indices,
        }

    def pre_decoder_ttnn(
        self,
        memory_tt: ttnn.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> Dict[str, Any]:
        N = int(spatial_shapes.prod(1).sum().item())
        B = memory_tt.shape[0]
        mem = (
            ttnn.slice(memory_tt, [0, 0, 0], [B, N, self.embed_dims], memory_config=self._dram_cfg)
            if memory_tt.shape[1] > N
            else memory_tt
        )

        proposals_host, valid_host = gen_encoder_output_proposals(torch.zeros(B, N, self.embed_dims), spatial_shapes)
        output_proposals = ttnn.from_torch(
            proposals_host.to(torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._dram_cfg,
        )
        output_proposals_valid = ttnn.from_torch(
            valid_host.to(torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._dram_cfg,
        )

        zeros_mem = ttnn.zeros_like(mem, memory_config=self._dram_cfg)
        output_memory = ttnn.where(output_proposals_valid, mem, zeros_mem)
        ttnn.deallocate(zeros_mem)
        ttnn.deallocate(output_proposals_valid)

        mfc = self._decoder_params["memory_trans_fc"]
        output_memory = ttnn.linear(
            output_memory,
            mfc["weight"],
            bias=mfc["bias"],
            memory_config=self._dram_cfg,
        )
        mnorm = self._decoder_params["memory_trans_norm"]
        output_memory = ttnn.layer_norm(output_memory, weight=mnorm["weight"], bias=mnorm["bias"])

        cls6 = self._decoder_params["cls_branches"][6]
        enc_cls = ttnn.linear(
            output_memory,
            cls6["weight"],
            bias=cls6["bias"],
            memory_config=self._dram_cfg,
        )

        reg6 = self._decoder_params["reg_branches"][6]
        reg_out = output_memory
        for i, layer in enumerate(reg6):
            reg_out = ttnn.linear(
                reg_out,
                layer["weight"],
                bias=layer["bias"],
                memory_config=self._dram_cfg,
            )
            if i < len(reg6) - 1:
                reg_out = ttnn.relu(reg_out)
        enc_coords_unact = ttnn.add(reg_out, output_proposals)
        ttnn.deallocate(output_proposals)
        ttnn.deallocate(reg_out)

        enc_cls_t = ttnn.to_torch(enc_cls).float()[:, :N, :]
        ttnn.deallocate(enc_cls)
        output_memory_t = ttnn.to_torch(output_memory).float()[:, :N, :]
        ttnn.deallocate(output_memory)

        coarse_cls = torch.nn.functional.linear(output_memory_t, self.cls_enc_w_torch, self.cls_enc_b_torch)
        topk_indices = torch.topk(coarse_cls.max(-1)[0], k=self.num_queries, dim=1)[1]

        enc_coords_t = ttnn.to_torch(enc_coords_unact).float()[:, :N, :]
        ttnn.deallocate(enc_coords_unact)
        topk_coords_unact = torch.gather(enc_coords_t, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        reference_points = topk_coords_unact.sigmoid()
        reference_points_tt = ttnn.from_torch(
            reference_points.to(torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._dram_cfg,
        )

        qemb = self._decoder_params["query_embedding"]
        qemb = ttnn.reshape(qemb, (1, self.num_queries, self.embed_dims))
        query_tt = ttnn.repeat(qemb, (B, 1, 1))

        if memory_tt.shape[1] > N:
            ttnn.deallocate(mem)
        return {
            "query": query_tt,
            "reference_points": reference_points_tt,
            "topk_score": None,
            "topk_coords": reference_points_tt,
            "topk_indices": topk_indices,
        }

    def forward_heads(
        self,
        hidden_states: List[ttnn.Tensor],
        references: List[torch.Tensor | ttnn.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply classification and regression heads to decoder outputs.
        All head computation (cls linear, reg branch, inverse_sigmoid, add, sigmoid)
        is done on device; results are stacked on device and converted to torch once at the end.

        Args:
            hidden_states: list of 6 ttnn [B, num_queries, 256] (normed decoder outputs)
            references: list of 7 [B, num_queries, 4]: references[0] torch, references[1:] ttnn

        Returns:
            all_cls: [num_layers, B, num_queries, num_classes] float32
            all_coords: [num_layers, B, num_queries, 4] float32 (sigmoid coords)
        """
        logger.info("Detection heads: computing class logits and bbox coords on device...")
        all_cls_tt: List[ttnn.Tensor] = []
        all_coords_tt: List[ttnn.Tensor] = []

        for layer_id in range(len(hidden_states)):
            hidden_state = hidden_states[layer_id]
            reference = references[layer_id]

            # Classification: Linear(256, num_classes) on device
            cls_w = self.cls_branches[layer_id]["weight"]
            cls_b = self.cls_branches[layer_id]["bias"]
            cls_out_tt = ttnn.linear(
                hidden_state,
                cls_w,
                bias=cls_b,
            )
            all_cls_tt.append(cls_out_tt)

            # Regression: reg_branch on device
            reg_out_tt = self.reg_branches_head[layer_id](hidden_state)
            # Reference to device, inverse_sigmoid on device, then add + sigmoid on device
            # references[0] is torch (initial); references[1:] are ttnn from decoder
            if isinstance(reference, torch.Tensor):
                ref_tt = ttnn.from_torch(
                    reference.to(torch.bfloat16),
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=self._dram_cfg,
                )
                ref_inv_tt = inverse_sigmoid_ttnn(ref_tt, eps=1e-3)
                ttnn.deallocate(ref_tt)
            else:
                ref_tt = reference
                ref_inv_tt = inverse_sigmoid_ttnn(ref_tt, eps=1e-3)
            coords_tt = ttnn.sigmoid(ttnn.add(reg_out_tt, ref_inv_tt))
            ttnn.deallocate(ref_inv_tt)
            ttnn.deallocate(reg_out_tt)
            all_coords_tt.append(coords_tt)
            logger.info(f"  Head layer {layer_id}: cls and coords on device")

        # Stack on device: [num_layers, B, num_queries, C]
        stacked_cls_tt = ttnn.stack(all_cls_tt, dim=0)
        stacked_coords_tt = ttnn.stack(all_coords_tt, dim=0)
        for t in all_cls_tt:
            ttnn.deallocate(t)
        for t in all_coords_tt:
            ttnn.deallocate(t)

        # Single host transfer and convert to float32
        _cls_t: torch.Tensor = ttnn.to_torch(stacked_cls_tt)
        all_cls = _cls_t.float()[:, :, : self.num_queries, :]
        _coords_t: torch.Tensor = ttnn.to_torch(stacked_coords_tt)
        all_coords = _coords_t.float()[:, :, : self.num_queries, :]
        ttnn.deallocate(stacked_cls_tt)
        ttnn.deallocate(stacked_coords_tt)
        logger.info(f"Detection heads done: cls {all_cls.shape}, coords {all_coords.shape}")

        return all_cls, all_coords

    def __call__(
        self,
        mlvl_feats_tt: List[ttnn.Tensor],
        profile_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        DINO inference from neck features → detections.

        Args:
            mlvl_feats: list of 5 NCHW torch tensors from neck [B, 256, H_i, W_i]

        Returns dict with:
            all_cls_scores: [num_layers, B, num_queries, num_classes]
            all_bbox_preds: [num_layers, B, num_queries, 4]
        """
        pre_trans = self.pre_transformer_tt(mlvl_feats_tt)

        # feat_tt = ttnn.from_torch(
        #     pre_trans["feat_flatten"],
        #     device=self.device,
        #     layout=ttnn.TILE_LAYOUT,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )
        # feat_pos_tt = ttnn.from_torch(
        #     pre_trans["feat_pos"],
        #     device=self.device,
        #     layout=ttnn.TILE_LAYOUT,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )

        logger.info("Running encoder...")
        memory_tt = self.encoder(
            feat=pre_trans["feat_flatten"],
            feat_pos=pre_trans["feat_pos"],
            feat_mask=None,
            spatial_shapes=pre_trans["spatial_shapes"],
            level_start_index=pre_trans["level_start_index"],
            valid_ratios=pre_trans["valid_ratios"],
        )
        ttnn.deallocate(pre_trans["feat_flatten"])
        ttnn.deallocate(pre_trans["feat_pos"])

        pre_dec = self.pre_decoder_ttnn(memory_tt, pre_trans["spatial_shapes"])

        logger.info("Running decoder...")
        hidden_states, references = self.decoder(
            query=pre_dec["query"],
            value=memory_tt,
            key_padding_mask=None,
            self_attn_mask=None,
            reference_points=pre_dec["reference_points"],
            spatial_shapes=pre_trans["spatial_shapes"],
            level_start_index=pre_trans["level_start_index"],
            valid_ratios=pre_trans["valid_ratios"],
        )

        all_cls, all_coords = self.forward_heads(hidden_states, references)

        return {
            "all_cls_scores": all_cls,
            "all_bbox_preds": all_coords,
        }

    def forward_image(
        self,
        image: torch.Tensor,
        return_intermediates: bool = False,
        profile_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full end-to-end inference from raw image tensor.

        Args:
            image: [B, 3, H, W] float32 torch tensor (normalized to ImageNet stats)
            return_intermediates: if True, also return backbone/neck/encoder outputs

        Returns dict with:
            all_cls_scores: [num_layers, B, num_queries, num_classes]
            all_bbox_preds: [num_layers, B, num_queries, 4] (cx, cy, w, h) in [0,1]
            (if return_intermediates):
                backbone_feats: list of 4 NCHW torch tensors
                neck_feats: list of 5 NCHW torch tensors
                encoder_memory: [B, N, 256] torch tensor
                decoder_hidden_states: list of 6 torch [B, 900, 256]
                decoder_references: list of 7 torch [B, 900, 4]
            profile_mode: if True, ReadDeviceProfiler after each stage (slower, for debugging)
        """
        assert self.backbone is not None, "Backbone not initialized — pass backbone_params"
        assert self.neck is not None, "Neck not initialized — pass neck_params"

        # --- Backbone ---
        logger.info(f"Backbone: input {image.shape}")
        image_tt = ttnn.from_torch(
            image,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        backbone_feats_tt = self.backbone(image_tt)
        ttnn.synchronize_device(self.device)
        if profile_mode:
            ttnn.ReadDeviceProfiler(self.device)
        logger.info(f"Backbone: {len(backbone_feats_tt)} feature maps")

        backbone_feats_torch = None
        if return_intermediates:
            # Backbone returns NHWC [B,H,W,C]; reference uses NCHW [B,C,H,W]. Permute for PCC.
            backbone_feats_torch = [
                ttnn.to_torch(ttnn.from_device(bf)).float().permute(0, 3, 1, 2) for bf in backbone_feats_tt
            ]

        # --- Neck ---
        logger.info("Neck: ChannelMapper...")
        neck_feats_tt = self.neck(backbone_feats_tt)
        ttnn.synchronize_device(self.device)
        if profile_mode:
            ttnn.ReadDeviceProfiler(self.device)
        logger.info(f"Neck: {len(neck_feats_tt)} output levels")

        neck_feats_torch = None
        if return_intermediates:
            neck_feats_torch = [ttnn.to_torch(ttnn.from_device(nf)).float() for nf in neck_feats_tt]
        for bf in backbone_feats_tt:
            ttnn.deallocate(bf)

        # --- Pre-transformer (TT-optimized: flatten + PE on device) → Encoder → ... ---
        pre_trans = self.pre_transformer_tt(neck_feats_tt)
        for nf in neck_feats_tt:
            ttnn.deallocate(nf)

        feat_tt = pre_trans["feat_flatten"]
        feat_pos_tt = pre_trans["feat_pos"]

        logger.info("Running encoder...")
        memory_tt = self.encoder(
            feat=feat_tt,
            feat_pos=feat_pos_tt,
            feat_mask=None,
            spatial_shapes=pre_trans["spatial_shapes"],
            level_start_index=pre_trans["level_start_index"],
            valid_ratios=pre_trans["valid_ratios"],
        )
        ttnn.deallocate(feat_tt)
        ttnn.deallocate(feat_pos_tt)
        ttnn.synchronize_device(self.device)
        if profile_mode:
            ttnn.ReadDeviceProfiler(self.device)

        encoder_memory_torch = None
        if return_intermediates:
            N = pre_trans["spatial_shapes"].prod(1).sum().item()
            encoder_memory_torch = ttnn.to_torch(memory_tt).float()[:, :N, :]

        pre_dec = self.pre_decoder_ttnn(memory_tt, pre_trans["spatial_shapes"])

        logger.info("Running decoder...")
        hidden_states, references = self.decoder(
            query=pre_dec["query"],
            value=memory_tt,
            key_padding_mask=None,
            self_attn_mask=None,
            reference_points=pre_dec["reference_points"],
            spatial_shapes=pre_trans["spatial_shapes"],
            level_start_index=pre_trans["level_start_index"],
            valid_ratios=pre_trans["valid_ratios"],
        )

        ttnn.synchronize_device(self.device)
        if profile_mode:
            ttnn.ReadDeviceProfiler(self.device)

        all_cls, all_coords = self.forward_heads(hidden_states, references)

        result = {
            "all_cls_scores": all_cls,
            "all_bbox_preds": all_coords,
        }

        if return_intermediates:
            N_q = self.num_queries
            result["backbone_feats"] = backbone_feats_torch
            result["neck_feats"] = neck_feats_torch
            result["encoder_memory"] = encoder_memory_torch
            result["decoder_hidden_states"] = [ttnn.to_torch(hs).float()[:, :N_q, :] for hs in hidden_states]
            result["decoder_references"] = [
                ttnn.to_torch(r).float()[:, :N_q, :] if isinstance(r, ttnn.Tensor) else r[:, :N_q, :]
                for r in references
            ]
            result["topk_indices"] = pre_dec.get("topk_indices")

        return result

    @staticmethod
    def postprocess(
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        img_shape: Tuple[int, int],
        score_thr: float = 0.3,
        nms_thr: float = 0.8,
        max_per_img: int = 300,
    ) -> Dict[str, torch.Tensor]:
        """
        Post-process detection outputs from the last decoder layer.

        Converts (cx, cy, w, h) normalized coords to (x1, y1, x2, y2) pixel coords
        and applies score thresholding + NMS.

        Args:
            cls_scores: [B, num_queries, num_classes] logits
            bbox_preds: [B, num_queries, 4] (cx, cy, w, h) in [0, 1]
            img_shape: (H, W) of the input image
            score_thr: minimum confidence score
            nms_thr: NMS IoU threshold
            max_per_img: max detections per image

        Returns dict with:
            boxes: [N, 4] (x1, y1, x2, y2) in pixel coords
            scores: [N] confidence scores
            labels: [N] class labels (0-indexed)
        """
        from torchvision.ops import batched_nms

        H, W = img_shape
        bs = cls_scores.shape[0]

        all_boxes, all_scores, all_labels = [], [], []

        for b in range(bs):
            scores = cls_scores[b].sigmoid()  # [num_queries, num_classes]
            bboxes = bbox_preds[b]  # [num_queries, 4] cx, cy, w, h

            # Convert to x1, y1, x2, y2 pixel coords
            cx, cy, bw, bh = bboxes.unbind(-1)
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

            # Flatten across classes
            max_scores, max_labels = scores.max(dim=-1)  # [num_queries]

            # Score threshold
            keep = max_scores > score_thr
            boxes_xyxy = boxes_xyxy[keep]
            max_scores = max_scores[keep]
            max_labels = max_labels[keep]

            if boxes_xyxy.numel() == 0:
                all_boxes.append(boxes_xyxy)
                all_scores.append(max_scores)
                all_labels.append(max_labels)
                continue

            # NMS
            nms_keep = batched_nms(boxes_xyxy, max_scores, max_labels, nms_thr)
            nms_keep = nms_keep[:max_per_img]

            all_boxes.append(boxes_xyxy[nms_keep])
            all_scores.append(max_scores[nms_keep])
            all_labels.append(max_labels[nms_keep])

        return {
            "boxes": all_boxes[0] if bs == 1 else all_boxes,
            "scores": all_scores[0] if bs == 1 else all_scores,
            "labels": all_labels[0] if bs == 1 else all_labels,
        }
