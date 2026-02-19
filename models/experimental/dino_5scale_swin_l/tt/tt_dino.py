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
from typing import Dict, List, Optional, Tuple

import torch
import ttnn
from loguru import logger

from models.experimental.swin_l.tt.tt_backbone import TtSwinLBackbone
from models.experimental.dino_5scale_swin_l.tt.tt_neck import TtDINONeck
from models.experimental.dino_5scale_swin_l.tt.tt_encoder import TtDINOEncoder
from models.experimental.dino_5scale_swin_l.tt.tt_decoder import (
    TtDINODecoder,
    TtRegBranch,
    inverse_sigmoid_torch,
)


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

    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [1, 256, H, W]
    return pos


def gen_encoder_output_proposals(
    memory_torch: torch.Tensor,
    spatial_shapes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate encoder output proposals (inference path: no memory_mask).
    Matches mmdet DINO.gen_encoder_output_proposals with memory_mask=None.

    Args:
        memory_torch: [B, N, 256] float32 encoder output
        spatial_shapes: [num_levels, 2] (H, W)

    Returns:
        output_proposals: [B, N, 4] inverse-sigmoid proposals (cx, cy, w, h)
        output_proposals_valid: [B, N, 1] bool mask of valid proposals
    """
    bs = memory_torch.shape[0]
    proposals = []
    _cur = 0
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
        _cur += H * W

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

        # --- Pre-decoder weights (original float32 from checkpoint) ---
        # Top-K selection is extremely sensitive to precision — using original
        # float32 weights (not bfloat16-quantized) matches the PyTorch reference.
        pd = decoder_params["_torch_pre_decoder"]
        self.memory_trans_fc_w_torch = pd["memory_trans_fc_w"]  # [out=256, in=256]
        self.memory_trans_fc_b_torch = pd["memory_trans_fc_b"]  # [256]
        self.memory_trans_norm_w_torch = pd["memory_trans_norm_w"]  # [256]
        self.memory_trans_norm_b_torch = pd["memory_trans_norm_b"]  # [256]
        self.query_embedding_torch = pd["query_embedding"]  # [900, 256]
        self.cls_enc_w_torch = pd["cls_enc_w"]  # [out=80, in=256]
        self.cls_enc_b_torch = pd["cls_enc_b"]  # [80]
        self.reg_enc_layers_torch = [(layer["weight"], layer["bias"]) for layer in pd["reg_enc_layers"]]

        # Device-side weights still needed for detection heads
        self.cls_branches = decoder_params["cls_branches"]
        self.reg_branches_head = [
            TtRegBranch(decoder_params["reg_branches"][i], device) for i in range(decoder_num_layers)
        ]

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

    def forward_heads(
        self,
        hidden_states: List[ttnn.Tensor],
        references: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply classification and regression heads to decoder outputs.

        Args:
            hidden_states: list of 6 ttnn [B, num_queries, 256] (normed decoder outputs)
            references: list of 7 torch [B, num_queries, 4] (initial + per-layer)

        Returns:
            all_cls: [num_layers, B, num_queries, num_classes] float32
            all_coords: [num_layers, B, num_queries, 4] float32 (sigmoid coords)
        """
        logger.info("Detection heads: computing class logits and bbox coords...")
        all_cls = []
        all_coords = []

        for layer_id in range(len(hidden_states)):
            hidden_state = hidden_states[layer_id]
            reference = references[layer_id]

            # Classification: Linear(256, 80) on device
            cls_w = self.cls_branches[layer_id]["weight"]
            cls_b = self.cls_branches[layer_id]["bias"]
            cls_out_tt = ttnn.linear(hidden_state, cls_w, bias=cls_b)
            cls_out = ttnn.to_torch(cls_out_tt).float()[:, : self.num_queries, :]
            ttnn.deallocate(cls_out_tt)

            # Regression: reg_branch on device + inverse_sigmoid(reference)
            reg_out_tt = self.reg_branches_head[layer_id](hidden_state)
            reg_out = ttnn.to_torch(reg_out_tt).float()[:, : self.num_queries, :]
            ttnn.deallocate(reg_out_tt)

            ref_inv = inverse_sigmoid_torch(reference, eps=1e-3)
            coords = (reg_out + ref_inv).sigmoid()

            all_cls.append(cls_out)
            all_coords.append(coords)
            logger.info(f"  Head layer {layer_id}: cls {cls_out.shape}, coords {coords.shape}")

        all_cls = torch.stack(all_cls, dim=0)
        all_coords = torch.stack(all_coords, dim=0)
        logger.info(f"Detection heads done: cls {all_cls.shape}, coords {all_coords.shape}")

        return all_cls, all_coords

    def __call__(
        self,
        mlvl_feats: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        DINO inference from neck features → detections.

        Args:
            mlvl_feats: list of 5 NCHW torch tensors from neck [B, 256, H_i, W_i]

        Returns dict with:
            all_cls_scores: [num_layers, B, num_queries, num_classes]
            all_bbox_preds: [num_layers, B, num_queries, 4]
        """
        pre_trans = self.pre_transformer(mlvl_feats)

        feat_tt = ttnn.from_torch(
            pre_trans["feat_flatten"].to(torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
        )
        feat_pos_tt = ttnn.from_torch(
            pre_trans["feat_pos"].to(torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
        )

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

        pre_dec = self.pre_decoder(memory_tt, pre_trans["spatial_shapes"])

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
        """
        assert self.backbone is not None, "Backbone not initialized — pass backbone_params"
        assert self.neck is not None, "Neck not initialized — pass neck_params"

        # --- Backbone ---
        logger.info(f"Backbone: input {image.shape}")
        image_tt = ttnn.from_torch(
            image,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        backbone_feats_tt = self.backbone(image_tt)
        ttnn.ReadDeviceProfiler(self.device)
        logger.info(f"Backbone: {len(backbone_feats_tt)} feature maps")

        backbone_feats_torch = None
        if return_intermediates:
            backbone_feats_torch = [ttnn.to_torch(ttnn.from_device(bf)).float() for bf in backbone_feats_tt]

        # --- Neck ---
        logger.info("Neck: ChannelMapper...")
        neck_feats_tt = self.neck(backbone_feats_tt)
        ttnn.ReadDeviceProfiler(self.device)
        logger.info(f"Neck: {len(neck_feats_tt)} output levels")

        neck_feats_torch = []
        for i, nf in enumerate(neck_feats_tt):
            nf_torch = ttnn.to_torch(ttnn.from_device(nf)).float()
            neck_feats_torch.append(nf_torch)
            ttnn.deallocate(nf)
        for bf in backbone_feats_tt:
            ttnn.deallocate(bf)

        # --- Pre-transformer → Encoder → Pre-decoder → Decoder → Heads ---
        pre_trans = self.pre_transformer(neck_feats_torch)

        feat_tt = ttnn.from_torch(
            pre_trans["feat_flatten"].to(torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
        )
        feat_pos_tt = ttnn.from_torch(
            pre_trans["feat_pos"].to(torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
        )

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
        ttnn.ReadDeviceProfiler(self.device)

        encoder_memory_torch = None
        if return_intermediates:
            N = pre_trans["spatial_shapes"].prod(1).sum().item()
            encoder_memory_torch = ttnn.to_torch(memory_tt).float()[:, :N, :]

        pre_dec = self.pre_decoder(memory_tt, pre_trans["spatial_shapes"])

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
            result["decoder_references"] = references
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
