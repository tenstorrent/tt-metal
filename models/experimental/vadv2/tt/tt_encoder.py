# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import copy
import warnings
import ttnn
from models.experimental.vadv2.tt.tt_temporal_self_attention import TtTemporalSelfAttention
from models.experimental.vadv2.tt.tt_spatial_cross_attention import TtSpatialCrossAttention
from models.experimental.vadv2.tt.tt_ffn import TtFFN


class TtBEVFormerEncoder:
    def __init__(
        self,
        params,
        device,
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
                dict(type="TemporalSelfAttention", embed_dims=embed_dims, num_levels=1),
                dict(
                    type="SpatialCrossAttention",
                    pc_range=pc_range,
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

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim="3d", bs=1, device="cuda", dtype=torch.float):
        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == "3d":
            zs = (
                torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device)
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d

    def point_sampling(self, reference_points, pc_range, img_metas):  # TODO Handle fp32
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32), reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps
        )

        reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
        reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )
        # TODO handle
        bev_mask = torch.nan_to_num(bev_mask)
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    # TODO Handle fp16
    def __call__(
        self,
        bev_query,
        key,
        value,
        *args,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=0.0,
        **kwargs,
    ):
        bev_query = ttnn.to_torch(bev_query)
        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )

        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim="2d", bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype
        )

        reference_points_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, kwargs["img_metas"])

        bev_mask = ttnn.from_torch(bev_mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        ref_2d = ttnn.from_torch(ref_2d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        bev_query = ttnn.from_torch(bev_query, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

        shift_ref_2d = ttnn.clone(ref_2d, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shift = ttnn.reshape(shift, (shift.shape[0], 1, 1, shift.shape[1]))
        shift_ref_2d = shift_ref_2d + shift

        bev_query = ttnn.permute(bev_query, (1, 0, 2))
        bev_pos = ttnn.permute(bev_pos, (1, 0, 2))
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = ttnn.permute(prev_bev, (1, 0, 2))
            prev_bev = ttnn.stack([prev_bev, bev_query], 1)
            prev_bev = ttnn.reshape(prev_bev, (bs * 2, len_bev, -1))
            hybird_ref_2d = ttnn.stack([shift_ref_2d, ref_2d], 1)
            hybird_ref_2d = ttnn.reshape(hybird_ref_2d, (bs * 2, len_bev, num_bev_level, 2))
        else:
            hybird_ref_2d = ttnn.stack([ref_2d, ref_2d], 1)
            hybird_ref_2d = ttnn.reshape(hybird_ref_2d, (bs * 2, len_bev, num_bev_level, 2))
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs,
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return ttnn.stack(intermediate)

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
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        **kwargs,
    ):
        bev_mask = kwargs.get("bev_mask", None)
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
                spatial_shapes_1 = torch.tensor([[bev_h, bev_w]])
                spatial_shapes_1 = ttnn.from_torch(
                    spatial_shapes_1, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                )
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
                    spatial_shapes=spatial_shapes_1,
                    level_start_index=level_start_index,
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
                ttnn.deallocate(self.params.norms[f"norm{norm_index}"].weight)
                ttnn.deallocate(self.params.norms[f"norm{norm_index}"].bias)
                norm_index += 1

            # spaital cross attention
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)

                ffn_index += 1

        return query
