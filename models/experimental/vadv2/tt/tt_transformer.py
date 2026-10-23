# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import numpy as np
from torchvision.transforms.functional import rotate
from models.experimental.vadv2.tt.tt_encoder import TtBEVFormerEncoder
from models.experimental.vadv2.tt.tt_decoder import TtDetectionTransformerDecoder, TtMapDetectionTransformerDecoder


class TtVADPerceptionTransformer:
    def __init__(
        self,
        params,
        params_branches,
        device,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        encoder=None,
        decoder=None,
        map_decoder=None,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        map_num_vec=50,
        map_num_pts_per_vec=10,
        **kwargs,
    ):
        super(TtVADPerceptionTransformer, self).__init__(**kwargs)
        point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        _dim_ = 256
        _pos_dim_ = _dim_ // 2
        _ffn_dim_ = _dim_ * 2
        self.device = device
        self.params = params
        self.params_branches = (params_branches,)
        self.encoder = TtBEVFormerEncoder(
            params.encoder,
            device,
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            embed_dims=_dim_,
            num_heads=4,
            dilation=1,
            kernel_size=(3, 5),
            im2col_step=192,
            feedforward_channels=_ffn_dim_,
            ffn_dropout=0.1,
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        )
        if decoder is not None:
            self.decoder = TtDetectionTransformerDecoder(
                num_layers=3,
                embed_dim=_dim_,
                num_heads=8,
                params=params.decoder,
                params_branches=params_branches,
                device=self.device,
            )
        else:
            self.decoder = None
        if map_decoder is not None:
            self.map_decoder = TtMapDetectionTransformerDecoder(
                num_layers=3,
                embed_dim=_dim_,
                num_heads=8,
                params=params.map_decoder,
                params_branches=params_branches,
                device=self.device,
            )
        else:
            self.map_decoder = None

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.two_stage_num_proposals = two_stage_num_proposals
        self.rotate_center = rotate_center
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec

        # Caches for the static spatial_shapes / level_start_index tensors
        # built per forward. Same values every time; trace-replay-blocking
        # `torch.tensor + ttnn.from_torch` upload only needed once.
        self._encoder_spatial_shapes_cache = {}  # keyed by tuple of (h, w) per level
        self._decoder_spatial_shapes_cache = {}  # keyed by (bev_h, bev_w)
        self._decoder_level_start_index_cache = None  # ttnn.zeros((1,)) uint32 TILE
        self._map_decoder_level_start_index_cache = None  # ttnn.zeros((1,)) bfloat16 TILE

        # Persistent device buffers for the only dynamic per-call uploads in
        # the warm path (shift / can_bus, built from img_metas). Allocated on
        # first call; in-place updated via copy_host_to_device_tensor on
        # subsequent calls. When `_skip_dynamic_upload` is True the inline
        # update is bypassed (caller must seed buffers before begin_trace_capture).
        self._shift_buffer = None
        self._can_bus_buffer = None
        self._skip_dynamic_upload = False

        # Encoder's level_start_index is a constant (0,) uint32 tensor. ttnn.zeros
        # does a host->device write of the fill, which is forbidden inside trace
        # capture — build once and reuse.
        self._encoder_level_start_index_cache = None

    def update_dynamic_inputs(self, img_metas, bev_h, bev_w, grid_length=[0.512, 0.512]):
        """Build shift / can_bus on host and write them into persistent device buffers.

        First call allocates the buffers via ttnn.from_torch. Subsequent calls reuse
        the same device handles and update contents via copy_host_to_device_tensor —
        the trace replay reads from these handles. Must be called outside any
        trace-capture / execute_trace region.
        """
        delta_x = np.array([each["can_bus"][0] for each in img_metas])
        delta_y = np.array([each["can_bus"][1] for each in img_metas])
        ego_angle = np.array([each["can_bus"][-2] / np.pi * 180 for each in img_metas])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift_torch = torch.tensor([shift_x, shift_y], dtype=torch.float32).permute(1, 0)
        can_bus_torch = torch.tensor([each["can_bus"] for each in img_metas], dtype=torch.float32)

        if self._shift_buffer is None:
            self._shift_buffer = ttnn.from_torch(
                shift_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            self._can_bus_buffer = ttnn.from_torch(
                can_bus_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        else:
            shift_host = ttnn.from_torch(shift_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            can_bus_host = ttnn.from_torch(can_bus_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(shift_host, self._shift_buffer)
            ttnn.copy_host_to_device_tensor(can_bus_host, self._can_bus_buffer)

    def attn_bev_encode(
        self,
        params,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        shift=None,
        can_bus=None,
        **kwargs,
    ):
        bs = mlvl_feats[0].shape[0]

        bev_queries = ttnn.unsqueeze(bev_queries, 1)
        bev_queries = ttnn.repeat(bev_queries, (1, bs, 1))
        # The previous code roundtripped bev_queries through torch (line 117 +
        # 154) as a side effect of using `.new_tensor(...)` to build shift /
        # can_bus. The roundtrip also re-created bev_queries as a fresh
        # TILE / DRAM INTERLEAVED tensor; replicate that explicitly.
        bev_queries = ttnn.to_layout(bev_queries, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        bev_pos = ttnn.reshape(bev_pos, [bev_pos.shape[0], bev_pos.shape[1], bev_pos.shape[2] * bev_pos.shape[3]])
        bev_pos = ttnn.permute(bev_pos, (2, 0, 1))
        # Populate persistent shift / can_bus device buffers from img_metas. Bypassed
        # when a caller has already seeded the buffers (e.g. trace capture / replay).
        if not self._skip_dynamic_upload:
            self.update_dynamic_inputs(kwargs["img_metas"], bev_h, bev_w, grid_length)
        # Clone so the encoder's ttnn.deallocate(shift) does not free the persistent buffer.
        shift = ttnn.clone(self._shift_buffer)

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = ttnn.permute(prev_bev, (1, 0, 2))
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = kwargs["img_metas"][i]["can_bus"][-1]
                    tmp_prev_bev = prev_bev[:, i]
                    tmp_prev_bev = ttnn.reshape(tmp_prev_bev, (bev_h, bev_w, -1))
                    tmp_prev_bev = ttnn.permute(tmp_prev_bev, (2, 0, 1))
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = ttnn.permute(tmp_prev_bev, (1, 2, 0))
                    tmp_prev_bev = ttnn.reshape(tmp_prev_bev, (bev_h * bev_w, 1, -1))
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals (sourced from the persistent buffer; see update_dynamic_inputs)
        # Clone so downstream ops can't free the persistent buffer.
        can_bus = ttnn.clone(self._can_bus_buffer)

        can_bus = ttnn.linear(can_bus, params.can_bus_mlp["0"].weight, bias=params.can_bus_mlp["0"].bias)
        can_bus = ttnn.relu(can_bus)
        can_bus = ttnn.linear(can_bus, params.can_bus_mlp["1"].weight, bias=params.can_bus_mlp["1"].bias)
        can_bus = ttnn.relu(can_bus)
        if self.can_bus_norm:
            can_bus = ttnn.layer_norm(
                can_bus,
                weight=self.params.can_bus_mlp.norm.weight,
                bias=self.params.can_bus_mlp.norm.bias,
            )
        can_bus = ttnn.reshape(can_bus, (1, can_bus.shape[0], can_bus.shape[1]))
        # [None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = ttnn.reshape(feat, (feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3] * feat.shape[4]))
            feat = ttnn.permute(feat, (1, 0, 3, 2))
            # ss
            if self.use_cams_embeds:
                cam_embeds = params.cams_embeds
                cam_embeds = ttnn.reshape(cam_embeds, (cam_embeds.shape[0], 1, 1, cam_embeds.shape[1]))
                feat = feat + cam_embeds
                ttnn.deallocate(cam_embeds)
            level_embeds = params.level_embeds
            level_embeds = level_embeds[lvl : lvl + 1, :]
            level_embeds = ttnn.reshape(level_embeds, (1, 1, level_embeds.shape[0], level_embeds.shape[-1]))
            feat = feat + level_embeds
            ttnn.deallocate(level_embeds)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = ttnn.concat(feat_flatten, 2)
        # spatial_shapes is built from the feat shapes which are static across
        # warm calls — cache the device tensor.
        ss_key = tuple(spatial_shapes)
        spatial_shapes_dev = self._encoder_spatial_shapes_cache.get(ss_key)
        if spatial_shapes_dev is None:
            spatial_shapes_t = torch.as_tensor(list(ss_key), dtype=torch.long, device="cpu")
            spatial_shapes_dev = ttnn.from_torch(
                spatial_shapes_t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
            self._encoder_spatial_shapes_cache[ss_key] = spatial_shapes_dev
        spatial_shapes = spatial_shapes_dev

        if self._encoder_level_start_index_cache is None:
            self._encoder_level_start_index_cache = ttnn.zeros(
                (1,), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        level_start_index = self._encoder_level_start_index_cache
        feat_flatten = ttnn.permute(feat_flatten, (0, 2, 1, 3))  # (num_cam, H*W, bs, embed_dims)
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs,
        )
        return bev_embed

    def get_bev_features(
        self,
        params,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        **kwargs,
    ):
        bev_embed = self.attn_bev_encode(
            params,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs,
        )

        return bev_embed

    def __call__(
        self,
        mlvl_feats,
        bev_queries,
        object_query_embed,
        map_query_embed,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        map_reg_branches=None,
        map_cls_branches=None,
        prev_bev=None,
        shift=None,
        can_bus=None,
        **kwargs,
    ):
        bev_embed = self.get_bev_features(
            self.params,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            shift=shift,
            can_bus=can_bus,
            **kwargs,
        )  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].shape[0]
        object_query_embed = ttnn.to_layout(object_query_embed, layout=ttnn.ROW_MAJOR_LAYOUT)
        query_pos, query = ttnn.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = ttnn.unsqueeze(query_pos, 0)
        query_pos = ttnn.expand(query_pos, (bs, -1, -1))
        query_pos = ttnn.to_layout(query_pos, layout=ttnn.TILE_LAYOUT)
        query = ttnn.unsqueeze(query, 0)
        query = ttnn.expand(query, (bs, -1, -1))
        reference_points = ttnn.linear(
            query_pos, self.params.reference_points.weight, bias=self.params.reference_points.bias
        )
        reference_points = ttnn.sigmoid(reference_points)
        init_reference_out = reference_points
        map_query_embed = ttnn.to_layout(map_query_embed, layout=ttnn.ROW_MAJOR_LAYOUT)

        map_query_pos, map_query = ttnn.split(map_query_embed, self.embed_dims, dim=1)
        map_query_pos = ttnn.unsqueeze(map_query_pos, 0)
        map_query_pos = ttnn.expand(map_query_pos, (bs, -1, -1))

        map_query_pos = ttnn.to_layout(map_query_pos, layout=ttnn.TILE_LAYOUT)
        map_query = ttnn.unsqueeze(map_query, 0)
        map_query = ttnn.expand(map_query, (bs, -1, -1))
        map_reference_points = ttnn.linear(
            map_query_pos, self.params.map_reference_points.weight, bias=self.params.map_reference_points.bias
        )
        map_reference_points = ttnn.sigmoid(map_reference_points)
        map_init_reference_out = map_reference_points

        query = ttnn.permute(query, (1, 0, 2))
        query_pos = ttnn.permute(query_pos, (1, 0, 2))
        map_query = ttnn.permute(map_query, (1, 0, 2))
        map_query_pos = ttnn.permute(map_query_pos, (1, 0, 2))
        bev_embed = ttnn.permute(bev_embed, (1, 0, 2))

        if self.decoder is not None:
            dec_key = (bev_h, bev_w, "bfloat16")
            spatial_shapes = self._decoder_spatial_shapes_cache.get(dec_key)
            if spatial_shapes is None:
                spatial_shapes_t = torch.tensor([[bev_h, bev_w]], device="cpu")
                spatial_shapes = ttnn.from_torch(
                    spatial_shapes_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                )
                self._decoder_spatial_shapes_cache[dec_key] = spatial_shapes
            if self._decoder_level_start_index_cache is None:
                self._decoder_level_start_index_cache = ttnn.zeros(
                    (1,), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=self.device
                )
            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=bev_embed,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=spatial_shapes,
                level_start_index=self._decoder_level_start_index_cache,
                **kwargs,
            )
            inter_references_out = inter_references
        else:
            inter_states = ttnn.unsqueeze(query, 0)
            inter_references_out = ttnn.unsqueeze(reference_points, 0)

        if self.map_decoder is not None:
            # [L, Q, B, D], [L, B, Q, D]
            map_key = (bev_h, bev_w, "bfloat16_map")
            spatial_shapes = self._decoder_spatial_shapes_cache.get(map_key)
            if spatial_shapes is None:
                spatial_shapes_t = torch.tensor([[bev_h, bev_w]], device="cpu")
                spatial_shapes = ttnn.from_torch(
                    spatial_shapes_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                )
                self._decoder_spatial_shapes_cache[map_key] = spatial_shapes
            if self._map_decoder_level_start_index_cache is None:
                self._map_decoder_level_start_index_cache = ttnn.zeros(
                    (1,), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
                )
            map_inter_states, map_inter_references = self.map_decoder(
                query=map_query,
                key=None,
                value=bev_embed,
                query_pos=map_query_pos,
                reference_points=map_reference_points,
                reg_branches=map_reg_branches,
                cls_branches=map_cls_branches,
                spatial_shapes=spatial_shapes,
                level_start_index=self._map_decoder_level_start_index_cache,
                **kwargs,
            )
            map_inter_references_out = map_inter_references
        else:
            map_inter_states = ttnn.unsqueeze(map_query, 0)
            map_inter_references_out = ttnn.unsqueeze(map_reference_points, 0)

        return (
            bev_embed,
            inter_states,
            init_reference_out,
            inter_references_out,
            map_inter_states,
            map_init_reference_out,
            map_inter_references_out,
        )
