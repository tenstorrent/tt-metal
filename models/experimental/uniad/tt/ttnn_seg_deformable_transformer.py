# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.uniad.tt.ttnn_detr_transformer_encoder import TtDetrTransformerEncoder
from models.experimental.uniad.tt.ttnn_detr_transformer_decoder import TtDeformableDetrTransformerDecoder


class TtSegDeformableTransformer:
    def __init__(self, device, params, as_two_stage=False, num_feature_levels=4, two_stage_num_proposals=300, **kwargs):
        super().__init__()
        self.fp16_enabled = False
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dims = 256
        self.device = device
        self.params = params
        self.level_embeds = params.level_embeds
        self.encoder = TtDetrTransformerEncoder(
            params=params.encoder,
            device=device,
        )

        self.decoder = TtDeformableDetrTransformerDecoder(
            params=params.decoder,
            device=device,
            num_layers=6,
            embed_dim=256,
            num_heads=8,
            params_branches=kwargs["params_branches"],
        )
        # level_start_index is a constant scalar `0` consumed by encoder
        # and decoder for every forward — allocate it once.
        self._level_start_index = ttnn.zeros((1,), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=self.device)
        # spatial_shapes, valid_ratios, and reference_points are all
        # functions of the multi-level feature shapes (constant for
        # fixed-input UniAD inference) and the all-zeros mask. Cache
        # the device tensors on first call. The cache key is the tuple
        # of (h, w) per level, which is what changes if the input
        # resolution does.
        self._spatial_shapes_cache = None
        self._valid_ratios_cache = None
        self._reference_points_cache = None
        self._cache_key = None

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            H = int(H)
            W = int(W)
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask, device=None):
        _, H, W = mask.shape
        one_tensor = ttnn.ones(mask.shape, layout=ttnn.TILE_LAYOUT, device=self.device)
        neg_mask = ttnn.subtract(one_tensor, mask)
        valid_H = ttnn.sum(neg_mask[:, :, 0], dim=1)
        valid_W = ttnn.sum(neg_mask[:, 0, :], dim=1)
        valid_ratio_h = ttnn.divide(valid_H, H)
        valid_ratio_w = ttnn.divide(valid_W, W)
        valid_ratio = ttnn.stack([valid_ratio_w, valid_ratio_h], dim=-1)
        return valid_ratio

    def forward(
        self,
        mlvl_feats,
        mlvl_masks,
        query_embed,
        mlvl_pos_embeds,
        reg_branches=None,
        cls_branches=None,
        level_embeds=None,
        **kwargs,
    ):
        assert self.as_two_stage or query_embed is not None
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = ttnn.reshape(feat, (feat.shape[0], feat.shape[1], -1))
            feat = ttnn.permute(feat, (0, 2, 1))
            mask = ttnn.reshape(mask, (mask.shape[0], -1))
            pos_embed = ttnn.reshape(pos_embed, (pos_embed.shape[0], pos_embed.shape[1], -1))
            pos_embed = ttnn.permute(pos_embed, (0, 2, 1))
            out = ttnn.reshape(self.level_embeds[lvl], (1, 1, -1))
            out = ttnn.to_layout(out, layout=ttnn.TILE_LAYOUT)
            lvl_pos_embed = pos_embed + out
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = ttnn.concat(feat_flatten, 1)
        mask_flatten = ttnn.concat(mask_flatten, 1)
        lvl_pos_embed_flatten = ttnn.concat(lvl_pos_embed_flatten, 1)

        # spatial_shapes / valid_ratios / reference_points are all functions
        # of the multi-level shape tuple and the all-zeros mask, both
        # frame-invariant for fixed-input UniAD inference. Lazy-cache the
        # device tensors. The original code rebuilt them every forward,
        # including a `to_torch → CPU get_reference_points → from_torch`
        # round-trip that costs ~10 ms (and blocks trace capture).
        #
        # The previous code also computed `prod/cumsum/cumsum_excl_last` on
        # spatial_shapes but never used the result — dead work, dropped.
        # Use the constant scalar zero allocated in __init__ for
        # level_start_index.
        level_start_index = self._level_start_index
        cache_key = tuple(spatial_shapes)
        if self._cache_key != cache_key:
            spatial_shapes_torch = torch.as_tensor(spatial_shapes, dtype=torch.long)
            valid_ratios_list = [self.get_valid_ratio(m, device=self.device) for m in mlvl_masks]
            valid_ratios_ttnn = ttnn.stack(valid_ratios_list, dim=1)
            valid_ratios_torch = ttnn.to_torch(valid_ratios_ttnn)
            reference_points_torch = self.get_reference_points(spatial_shapes_torch, valid_ratios_torch, device="cpu")
            self._reference_points_cache = ttnn.from_torch(
                reference_points_torch, device=self.device, layout=ttnn.TILE_LAYOUT
            )
            self._spatial_shapes_cache = ttnn.from_torch(
                spatial_shapes_torch,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            self._valid_ratios_cache = ttnn.from_torch(valid_ratios_torch, device=self.device, layout=ttnn.TILE_LAYOUT)
            self._cache_key = cache_key
        spatial_shapes = self._spatial_shapes_cache
        valid_ratios = self._valid_ratios_cache
        reference_points = self._reference_points_cache
        feat_flatten = ttnn.permute(feat_flatten, (1, 0, 2))
        lvl_pos_embed_flatten = ttnn.permute(lvl_pos_embed_flatten, (1, 0, 2))
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )
        memory = ttnn.permute(memory, (1, 0, 2))
        bs, _, c = memory.shape

        query_pos = query_embed[:, :c, ...]
        query = query_embed[:, c:, ...]
        query_pos = ttnn.unsqueeze(query_pos, 0)
        query_pos = ttnn.expand(query_pos, (bs, -1, -1))
        query = ttnn.unsqueeze(query, 0)
        query = ttnn.expand(query, (bs, -1, -1))
        query_pos = ttnn.to_layout(query_pos, ttnn.TILE_LAYOUT)
        reference_points = ttnn.linear(
            query_pos, self.params.reference_points.weight, bias=self.params.reference_points.bias
        )
        reference_points = ttnn.sigmoid(reference_points)
        init_reference_out = reference_points

        query = ttnn.permute(query, (1, 0, 2))
        memory = ttnn.permute(memory, (1, 0, 2))
        query_pos = ttnn.permute(query_pos, (1, 0, 2))

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs,
        )

        inter_references_out = inter_references

        return (
            (memory, lvl_pos_embed_flatten, mask_flatten, query_pos),
            inter_states,
            init_reference_out,
            inter_references_out,
            None,
            None,
        )
