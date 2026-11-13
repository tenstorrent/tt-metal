# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import copy
import warnings
import ttnn
from models.experimental.vadv1.tt.tt_temporal_self_attention import TtTemporalSelfAttention
from models.experimental.vadv1.tt.tt_spatial_cross_attention import TtSpatialCrossAttention
from models.experimental.vadv1.tt.tt_ffn import TtFFN
from models.experimental.vadv1.tt.tt_point_sampling import camera_rows, point_sampling_3d_to_2d_ttnn, world_points_ttnn
from models.experimental.vadv1.tt.tt_sca_rebatch_plan import SCARebatchPlanner


class TtBEVFormerEncoder:
    def __init__(
        self,
        params,
        device,
        bev_h,
        bev_w,
        feat_hw,
        img_shape,
        num_layers=3,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        embed_dims=256,
        num_heads=4,
        dilation=1,
        kernel_size=(3, 5),
        im2col_step=192,
        feedforward_channels=512,
        ffn_dropout=0.1,
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
    ):
        super(TtBEVFormerEncoder, self).__init__()
        self.device = device
        self.params = params
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.num_layers = num_layers
        self.fp16_enabled = False

        transformer_layers = dict(
            attn_cfgs=[
                dict(
                    type="TemporalSelfAttention",
                    embed_dims=embed_dims,
                    num_levels=1,
                    spatial_shapes_hw=[(bev_h, bev_w)],
                ),
                dict(
                    type="SpatialCrossAttention",
                    pc_range=pc_range,
                    spatial_shapes_hw=[tuple(feat_hw)],
                    attention=dict(
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        num_levels=1,
                        im2col_step=im2col_step,
                    ),
                    embed_dims=embed_dims,
                ),
            ],
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
        )

        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(TtBEVFormerLayer(self.device, params.layers[f"layer{i}"], **transformer_layers))

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.img_shape = tuple(img_shape)
        # Built on the host in float32, as the reference computes them: in bfloat16 the BEV cell
        # centres round by up to 0.4 cells.
        ref_3d = self.reference_points_3d(bev_h, bev_w, pc_range[5] - pc_range[2], num_points_in_pillar)
        self.ref_2d = ttnn.from_torch(
            self.reference_points_2d(bev_h, bev_w),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # As BEVFormer: the pillar grid in world coordinates, which every frame projects to its
        # cameras on device, and one rebatch planner, whose capacity is the rig's high-water mark.
        self.world_points = world_points_ttnn(ref_3d.permute(0, 2, 1, 3), pc_range, device)
        self.rebatch_planner = SCARebatchPlanner(device)
        self._cameras = None

    @staticmethod
    def reference_points_3d(H, W, Z, num_points_in_pillar):
        """(1, num_points_in_pillar, H * W, 3) pillar points in [0, 1], float32."""
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        return ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)[None]

    @staticmethod
    def reference_points_2d(H, W):
        """(1, H * W, 1, 2) BEV cell centres in [0, 1], float32."""
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H), torch.linspace(0.5, W - 0.5, W), indexing="ij")
        ref_2d = torch.stack((ref_x.reshape(-1)[None] / W, ref_y.reshape(-1)[None] / H), -1)
        return ref_2d.unsqueeze(2)

    def projection_rows(self, lidar2img):
        """Host projection rows (camera_rows) of the cameras' (num_cams, 4, 4) lidar2img
        over the pillar grid, (1, num_cams, 9, width) float32."""
        lidar2img = torch.as_tensor(np.asarray(lidar2img), dtype=torch.float32).reshape(1, -1, 4, 4)
        grid = self.world_points
        return camera_rows(lidar2img, self.img_shape, grid.num_queries, grid.num_points)

    def project_cameras(self, projection):
        """Project the pillar grid through the frame's projection rows on device, then size the
        rebatch plan: the only readback is each camera's visible count, and only a frame needing
        more rows than the planner holds grows it. Call before the forward, outside any trace."""
        reference_points_cam, bev_mask = point_sampling_3d_to_2d_ttnn(
            reference_points=None,
            pc_range=None,
            lidar2img=None,
            device=self.device,
            world_points=self.world_points,
            projection=projection,
            # Projected in float32; bfloat16 is the only dtype the rebatch plan's gather takes.
            output_dtype=ttnn.bfloat16,
        )
        visibility = self.rebatch_planner.visibility(bev_mask)
        self.rebatch_planner.size(visibility)
        self._cameras = (reference_points_cam, visibility)

    def __call__(
        self,
        bev_query,
        key,
        value,
        *args,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        valid_ratios=None,
        prev_bev=None,
        shift=None,
        **kwargs,
    ):
        """shift: the ego translation in BEV fractions, float32 ROW_MAJOR (bs, 1, 1, 2)."""
        output = bev_query
        intermediate = []

        assert (bev_h, bev_w) == (self.bev_h, self.bev_w)
        assert bev_query.shape[1] == 1, "reference points are built for bs=1"
        assert self._cameras is not None, "project_cameras() with this frame's projection before the forward"
        ref_2d = self.ref_2d
        # Device work only: the planner was sized in project_cameras().
        reference_points_cam, visibility = self._cameras
        rebatch_plan = self.rebatch_planner.build(reference_points_cam, visibility)
        self._cameras = None

        bev_query = ttnn.permute(bev_query, (1, 0, 2))
        bev_pos = ttnn.permute(bev_pos, (1, 0, 2))
        bs, len_bev, _ = bev_query.shape
        # Every BEV map's float32 reference points, previous BEV first: the previous one's moved
        # by the ego shift, (bs, Q, 2, 2), as the temporal self-attention samples the two maps.
        if prev_bev is not None:
            prev_bev = ttnn.permute(prev_bev, (1, 0, 2))
            prev_bev = ttnn.stack([prev_bev, bev_query], 1)
            prev_bev = ttnn.reshape(prev_bev, (bs * 2, len_bev, -1))
            shifted_ref_2d = ttnn.add(ref_2d, shift)
            hybird_ref_2d = ttnn.concat([shifted_ref_2d, ref_2d], dim=2)
            ttnn.deallocate(shifted_ref_2d)
        else:
            hybird_ref_2d = ttnn.concat([ref_2d, ref_2d], dim=2)
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                bev_h=bev_h,
                bev_w=bev_w,
                prev_bev=prev_bev,
                rebatch_plan=rebatch_plan,
                **kwargs,
            )
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)
        ttnn.deallocate(hybird_ref_2d)

        if self.return_intermediate:
            stacked = ttnn.stack(intermediate)
            for it in intermediate:
                ttnn.deallocate(it)
            return stacked

        return output


class TtBEVFormerLayer:
    def __init__(
        self,
        device,
        params,
        attn_cfgs,
        feedforward_channels,
        ffn_dropout=0.0,
        operation_order=None,
        ffn_num_fcs=2,
        **kwargs,
    ):
        super(TtBEVFormerLayer, self).__init__()
        self.params = params
        self.device = device
        self.attn_cfgs = attn_cfgs
        self.feedforward_channels = feedforward_channels
        self.ffn_dropout = ffn_dropout
        self.operation_order = operation_order
        self.ffn_num_fcs = ffn_num_fcs
        self.fp16_enabled = False
        self.batch_first = True
        self.attentions = []
        index = 0
        for operation_name in self.operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first
                if attn_cfgs[index]["type"] == "TemporalSelfAttention":
                    type = attn_cfgs[index].pop("type")
                    attention = TtTemporalSelfAttention(device, params.attentions[f"attn0"], **attn_cfgs[index])
                    attn_cfgs[index]["type"] = "TemporalSelfAttention"
                elif attn_cfgs[index]["type"] == "SpatialCrossAttention":
                    type = attn_cfgs[index].pop("type")
                    attention = TtSpatialCrossAttention(device, params.attentions[f"attn1"], **attn_cfgs[index])
                    attn_cfgs[index]["type"] = "SpatialCrossAttention"

                self.attentions.append(attention)
                index += 1

        self.pre_norm = operation_order[0] == "norm"

        self.embed_dims = self.attentions[0].embed_dims

        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.ffns = []
        num_ffns = operation_order.count("ffn")

        for i in range(num_ffns):
            self.ffns.append(TtFFN(params.ffn[f"ffn{i}"], self.device))

        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

    def __call__(
        self,
        query,
        key,
        value,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        mask=None,
        prev_bev=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, ttnn.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = ttnn.layer_norm(
                    query,
                    weight=self.params.norms[f"norm{norm_index}"].weight,
                    bias=self.params.norms[f"norm{norm_index}"].bias,
                )
                norm_index += 1

            # spatial cross attention
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
