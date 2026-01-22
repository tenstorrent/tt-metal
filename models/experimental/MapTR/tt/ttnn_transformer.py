# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
import numpy as np
from scipy.ndimage import rotate as scipy_rotate
from typing import List, Optional, Tuple


class TtConvFuser:
    """TTNN Convolution-based feature fuser for multi-modal fusion."""

    def __init__(self, params, device: ttnn.Device):
        self.device = device
        self.params = params

    def __call__(self, inputs: List[ttnn.Tensor]) -> ttnn.Tensor:
        # Concatenate inputs along channel dimension
        concatenated = ttnn.concat(inputs, dim=1)

        # Apply convolution
        conv_output = ttnn.conv2d(
            input=concatenated,
            weight=self.params.conv.weight,
            bias=self.params.conv.bias,
            padding=[1, 1],
            stride=[1, 1],
            dilation=[1, 1],
            groups=1,
        )

        # Apply batch normalization
        bn_output = ttnn.batch_norm(
            conv_output,
            running_mean=self.params.batch_norm.running_mean,
            running_var=self.params.batch_norm.running_var,
            weight=self.params.batch_norm.weight,
            bias=self.params.batch_norm.bias,
            epsilon=1e-5,
            momentum=0.1,
            training=False,
        )

        # Apply ReLU activation
        return ttnn.relu(bn_output)


class TtMapTRPerceptionTransformer:
    """TTNN MapTR Perception Transformer Implementation."""

    def __init__(
        self,
        params,
        device: ttnn.Device,
        encoder,
        decoder,
        embed_dims: int = 256,
        num_feature_levels: int = 1,
        num_cams: int = 6,
        rotate_prev_bev: bool = True,
        use_shift: bool = True,
        use_can_bus: bool = True,
        can_bus_norm: bool = True,
        use_cams_embeds: bool = True,
        rotate_center: List[int] = None,
        fuser: Optional[TtConvFuser] = None,
    ):
        if rotate_center is None:
            rotate_center = [100, 100]

        self.device = device
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.rotate_center = rotate_center
        self.fuser = fuser

        # Check if using attention-based BEV encoder
        self.use_attn_bev = hasattr(encoder, "layers")

    def attn_bev_encode(
        self,
        mlvl_feats: List[ttnn.Tensor],
        bev_queries: ttnn.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: List[float] = None,
        bev_pos: ttnn.Tensor = None,
        prev_bev: ttnn.Tensor = None,
        **kwargs,
    ) -> ttnn.Tensor:
        """TTNN BEV feature encoding using attention-based encoder."""
        if grid_length is None:
            grid_length = [0.512, 0.512]

        bs = mlvl_feats[0].shape[0]

        # Expand BEV queries
        bev_queries = ttnn.unsqueeze(bev_queries, 1)
        bev_queries = ttnn.repeat(bev_queries, (1, bs, 1))

        # Reshape BEV position
        bev_pos = ttnn.reshape(bev_pos, [bev_pos.shape[0], bev_pos.shape[1], bev_pos.shape[2] * bev_pos.shape[3]])
        bev_pos = ttnn.permute(bev_pos, (2, 0, 1))

        # Convert to torch for tensor creation (following vadv2 pattern)
        bev_queries_torch = ttnn.to_torch(bev_queries)

        # Obtain rotation angle and shift with ego motion
        img_metas = kwargs.get("img_metas", [{}])
        delta_x = np.array([each.get("can_bus", np.zeros(18))[0] for each in img_metas])
        delta_y = np.array([each.get("can_bus", np.zeros(18))[1] for each in img_metas])
        ego_angle = np.array([each.get("can_bus", np.zeros(18))[-2] / np.pi * 180 for each in img_metas])

        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift

        # Create shift tensor (vadv2 pattern: use new_tensor then permute)
        shift = bev_queries_torch.new_tensor([shift_x, shift_y]).permute(1, 0)  # [bs, 2]
        shift = ttnn.from_torch(shift, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # Handle previous BEV features with rotation
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = ttnn.permute(prev_bev, (1, 0, 2))
            if self.rotate_prev_bev:
                prev_bev_torch = ttnn.to_torch(prev_bev)
                for i in range(bs):
                    rotation_angle = img_metas[i].get("can_bus", np.zeros(18))[-1]
                    tmp_prev_bev = prev_bev_torch[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    # Use scipy for rotation (operates on numpy)
                    tmp_prev_bev_np = tmp_prev_bev.float().numpy()
                    tmp_prev_bev_np = scipy_rotate(
                        tmp_prev_bev_np, rotation_angle, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0
                    )
                    tmp_prev_bev = torch.from_numpy(tmp_prev_bev_np).permute(1, 2, 0).reshape(bev_h * bev_w, -1)
                    prev_bev_torch[:, i] = tmp_prev_bev
                prev_bev = ttnn.from_torch(
                    prev_bev_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
                )

        # Add CAN bus signals (vadv2 pattern: direct linear layers)
        can_bus = bev_queries_torch.new_tensor([each.get("can_bus", np.zeros(18)) for each in img_metas])
        can_bus = ttnn.from_torch(can_bus, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        bev_queries = ttnn.from_torch(
            bev_queries_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # CAN bus MLP: Linear -> ReLU -> Linear -> ReLU -> LayerNorm
        can_bus = ttnn.linear(can_bus, self.params.can_bus_mlp["0"].weight, bias=self.params.can_bus_mlp["0"].bias)
        can_bus = ttnn.relu(can_bus)
        can_bus = ttnn.linear(can_bus, self.params.can_bus_mlp["2"].weight, bias=self.params.can_bus_mlp["2"].bias)
        can_bus = ttnn.relu(can_bus)

        if self.can_bus_norm:
            can_bus = ttnn.layer_norm(
                can_bus,
                weight=self.params.can_bus_mlp.norm.weight,
                bias=self.params.can_bus_mlp.norm.bias,
            )

        # Reshape and add to bev_queries
        can_bus = ttnn.reshape(can_bus, (1, can_bus.shape[0], can_bus.shape[1]))
        if self.use_can_bus:
            bev_queries = bev_queries + can_bus

        # Process multi-level features
        feat_flatten = []
        spatial_shapes = []

        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            # Flatten and permute features
            feat = ttnn.reshape(feat, (bs, num_cam, c, h * w))
            feat = ttnn.permute(feat, (1, 0, 3, 2))

            # Add camera embeddings
            if self.use_cams_embeds:
                cam_embeds = self.params.cams_embeds
                cam_embeds = ttnn.reshape(cam_embeds, (cam_embeds.shape[0], 1, 1, cam_embeds.shape[1]))
                feat = feat + cam_embeds

            # Add level embeddings
            level_embeds = self.params.level_embeds
            level_embeds = level_embeds[lvl : lvl + 1, :]
            level_embeds = ttnn.reshape(level_embeds, (1, 1, level_embeds.shape[0], level_embeds.shape[-1]))
            feat = feat + level_embeds

            feat_flatten.append(feat)

        # Concatenate all features
        feat_flatten = ttnn.concat(feat_flatten, 2)
        feat_flatten = ttnn.permute(feat_flatten, (0, 2, 1, 3))  # (num_cam, H*W, bs, embed_dims)

        # Convert spatial shapes to tensor (vadv2 pattern)
        spatial_shapes_tensor = torch.as_tensor(spatial_shapes, dtype=torch.long, device="cpu")
        spatial_shapes_ttnn = ttnn.from_torch(
            spatial_shapes_tensor, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

        # Level start index (simple zeros for single level)
        level_start_index = ttnn.zeros((1,), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=self.device)

        # Call encoder
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes_ttnn,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs,
        )

        return bev_embed

    def lss_bev_encode(
        self,
        mlvl_feats: List[ttnn.Tensor],
        prev_bev: ttnn.Tensor = None,
        **kwargs,
    ) -> ttnn.Tensor:
        """TTNN BEV feature encoding using LSS-based encoder."""
        assert len(mlvl_feats) == 1, "Currently we only support single level feat in LSS"
        images = mlvl_feats[0]
        img_metas = kwargs.get("img_metas", [])

        bev_embed = self.encoder(images, img_metas)
        bs, c, _, _ = bev_embed.shape

        bev_embed = ttnn.reshape(bev_embed, [bs, c, -1])
        bev_embed = ttnn.permute(bev_embed, [0, 2, 1])

        return bev_embed

    def get_bev_features(
        self,
        mlvl_feats: List[ttnn.Tensor],
        lidar_feat: Optional[ttnn.Tensor],
        bev_queries: ttnn.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: List[float] = None,
        bev_pos: ttnn.Tensor = None,
        prev_bev: ttnn.Tensor = None,
        **kwargs,
    ) -> ttnn.Tensor:
        """TTNN BEV feature extraction."""
        if grid_length is None:
            grid_length = [0.512, 0.512]

        if self.use_attn_bev:
            bev_embed = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs,
            )
        else:
            bev_embed = self.lss_bev_encode(
                mlvl_feats,
                prev_bev=prev_bev,
                **kwargs,
            )

        # Fuse with LiDAR features if available
        if lidar_feat is not None and self.fuser is not None:
            bs = mlvl_feats[0].shape[0]

            # Reshape BEV features for fusion
            bev_embed = ttnn.reshape(bev_embed, [bs, bev_h, bev_w, -1])
            bev_embed = ttnn.permute(bev_embed, [0, 3, 1, 2])

            # Process LiDAR features
            lidar_feat = ttnn.permute(lidar_feat, [0, 1, 3, 2])
            lidar_feat = ttnn.interpolate(lidar_feat, size=(bev_h, bev_w), mode="bicubic", align_corners=False)

            # Fuse features
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = ttnn.reshape(fused_bev, [bs, -1, bev_h * bev_w])
            fused_bev = ttnn.permute(fused_bev, [0, 2, 1])
            bev_embed = fused_bev

        return bev_embed

    def __call__(
        self,
        mlvl_feats: List[ttnn.Tensor],
        lidar_feat: Optional[ttnn.Tensor],
        bev_queries: ttnn.Tensor,
        object_query_embed: ttnn.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: List[float] = None,
        bev_pos: ttnn.Tensor = None,
        reg_branches: Optional[List] = None,
        cls_branches: Optional[List] = None,
        prev_bev: ttnn.Tensor = None,
        **kwargs,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Forward pass."""
        if grid_length is None:
            grid_length = [0.512, 0.512]

        # Get BEV features
        bev_embed = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs,
        )  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].shape[0]

        # Log BEV embedding for debugging
        import logging

        logger = logging.getLogger(__name__)
        bev_embed_torch = ttnn.to_torch(bev_embed)
        logger.info(
            f"[TT Transformer] BEV embed shape: {bev_embed_torch.shape}, sample: {bev_embed_torch.flatten()[:5].tolist()}"
        )

        # Split object query embeddings into position and content
        object_query_embed = ttnn.to_layout(object_query_embed, layout=ttnn.ROW_MAJOR_LAYOUT)
        query_pos, query = ttnn.split(object_query_embed, self.embed_dims, dim=1)

        # Expand queries for batch dimension
        query_pos = ttnn.unsqueeze(query_pos, 0)
        query_pos = ttnn.expand(query_pos, (bs, -1, -1))
        query_pos = ttnn.to_layout(query_pos, layout=ttnn.TILE_LAYOUT)

        query = ttnn.unsqueeze(query, 0)
        query = ttnn.expand(query, (bs, -1, -1))

        # Calculate reference points
        reference_points = ttnn.linear(
            query_pos, self.params.reference_points.weight, bias=self.params.reference_points.bias
        )
        reference_points = ttnn.sigmoid(reference_points)
        init_reference_out = reference_points

        # Permute for decoder input
        query = ttnn.permute(query, (1, 0, 2))
        query_pos = ttnn.permute(query_pos, (1, 0, 2))
        bev_embed = ttnn.permute(bev_embed, (1, 0, 2))

        # Create spatial shapes and level start index (vadv2 pattern)
        spatial_shapes = torch.tensor([[bev_h, bev_w]], device="cpu")
        spatial_shapes = ttnn.from_torch(
            spatial_shapes, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        level_start_index = ttnn.zeros((1,), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=self.device)

        # Call decoder
        # NOTE: decoder expects map_reg_branches, not reg_branches
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            map_reg_branches=reg_branches,  # Fix: was reg_branches, decoder expects map_reg_branches
            cls_branches=cls_branches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs,
        )

        # Log decoder outputs for debugging
        inter_states_torch = ttnn.to_torch(inter_states)
        inter_refs_torch = ttnn.to_torch(inter_references)
        logger.info(
            f"[TT Transformer] Decoder inter_states shape: {inter_states_torch.shape}, sample: {inter_states_torch.flatten()[:5].tolist()}"
        )
        logger.info(
            f"[TT Transformer] Decoder inter_refs shape: {inter_refs_torch.shape}, sample: {inter_refs_torch.flatten()[:5].tolist()}"
        )

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
