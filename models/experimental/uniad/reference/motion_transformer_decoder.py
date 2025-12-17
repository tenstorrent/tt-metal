# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import copy
import warnings
from einops import rearrange
from torch.nn import Sequential

from models.experimental.uniad.reference.decoder import multi_scale_deformable_attn_pytorch, FFN
from models.experimental.uniad.reference.utils import norm_points, pos2posemb2d, trajectory_coordinate_transform


class MotionDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_steps=1,
        sample_index=-1,
        im2col_step=64,
        bev_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size=[0.2, 0.2, 8],
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False
        self.bev_range = bev_range

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_steps = num_steps
        self.sample_index = sample_index
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_steps * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_steps * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = Sequential(
            nn.Linear(num_steps * embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True)
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        spatial_shapes=None,
        level_start_index=None,
        bbox_results=None,
        reference_trajs=None,
        flag="decoder",
        **kwargs,
    ):
        bs, num_agent, num_mode, _ = query.shape
        num_query = num_agent * num_mode
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        query = torch.flatten(query, start_dim=1, end_dim=2)

        value = value.permute(1, 0, 2)
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_steps, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_steps, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_steps, self.num_levels, self.num_points
        )

        if reference_trajs.shape[-1] == 2:
            reference_trajs = reference_trajs[:, :, :, [self.sample_index], :, :]
            reference_trajs_ego = self.agent_coords_to_ego_coords(copy.deepcopy(reference_trajs), bbox_results).detach()
            reference_trajs_ego = torch.flatten(reference_trajs_ego, start_dim=1, end_dim=2)
            reference_trajs_ego = reference_trajs_ego[:, :, None, :, :, None, :]
            reference_trajs_ego[..., 0] -= self.bev_range[0]
            reference_trajs_ego[..., 1] -= self.bev_range[1]
            reference_trajs_ego[..., 0] /= self.bev_range[3] - self.bev_range[0]
            reference_trajs_ego[..., 1] /= self.bev_range[4] - self.bev_range[1]
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_trajs_ego + sampling_offsets / offset_normalizer[None, None, None, None, :, None, :]
            )

            sampling_locations = rearrange(
                sampling_locations, "bs nq nh ns nl np c -> bs nq ns nh nl np c"
            )  # permute([0,1,3,2,4,5,6])
            attention_weights = rearrange(
                attention_weights, "bs nq nh ns nl np -> bs nq ns nh nl np"
            )  # .permute([0,1,3,2,4,5])
            sampling_locations = sampling_locations.reshape(
                bs, num_query * self.num_steps, self.num_heads, self.num_levels, self.num_points, 2
            )
            attention_weights = attention_weights.reshape(
                bs, num_query * self.num_steps, self.num_heads, self.num_levels, self.num_points
            )

        else:
            raise ValueError(
                f"Last dim of reference_trajs must be" f" 2 or 4, but get {reference_trajs.shape[-1]} instead."
            )
        if torch.cuda.is_available() and value.is_cuda:
            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step
            )
        output = output.view(bs, num_query, self.num_steps, -1)
        output = torch.flatten(output, start_dim=2, end_dim=3)
        output = self.output_proj(output)
        output = output.view(bs, num_agent, num_mode, -1)
        return output + identity

    def agent_coords_to_ego_coords(self, reference_trajs, bbox_results):
        batch_size = len(bbox_results)
        reference_trajs_ego = []
        for i in range(batch_size):
            boxes_3d, scores, labels, bbox_index, mask = bbox_results[i]
            det_centers = boxes_3d.gravity_center.to(reference_trajs.device)
            batch_reference_trajs = reference_trajs[i]
            batch_reference_trajs += det_centers[:, None, None, None, :2]
            reference_trajs_ego.append(batch_reference_trajs)
        return torch.stack(reference_trajs_ego)

    def rot_2d(self, yaw):
        sy, cy = torch.sin(yaw), torch.cos(yaw)
        out = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).permute([2, 0, 1])
        return out


class MotionTransformerAttentionLayer(nn.Module):
    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        deprecated_args = dict(feedforward_channels="feedforward_channels", ffn_num_fcs="num_fcs")
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments "
                    f"to a dict named `ffn_cfgs`. ",
                    DeprecationWarning,
                )
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__()

        self.batch_first = batch_first

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
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first
                attention = MotionDeformableAttention(
                    num_steps=12, embed_dims=256, num_levels=1, num_heads=8, num_points=4, sample_index=-1
                )
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = nn.ModuleList()

        num_ffns = operation_order.count("ffn")
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ffn_cfgs
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if "embed_dims" not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]["embed_dims"] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]["embed_dims"] == self.embed_dims
            self.ffns.append(FFN(embed_dims=256))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(nn.LayerNorm(self.embed_dims))

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
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
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
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


class MapInteraction(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, batch_first=True, norm_cfg=None, init_cfg=None):
        super().__init__()

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def forward(self, query, key, query_pos=None, key_pos=None):
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # N, A, P, D -> N*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        mem = key.expand(B * A, -1, -1)
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query


class TrackAgentInteraction(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, batch_first=True, norm_cfg=None, init_cfg=None):
        super().__init__()

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def forward(self, query, key, query_pos=None, key_pos=None):
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        mem = key.expand(B * A, -1, -1)
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query


class IntentionInteraction(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, batch_first=True, norm_cfg=None, init_cfg=None):
        super().__init__()

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def forward(self, query):
        B, A, P, D = query.shape
        # B, A, P, D -> B*A,P, D
        rebatch_x = torch.flatten(query, start_dim=0, end_dim=1)
        rebatch_x = self.interaction_transformer(rebatch_x)
        out = rebatch_x.view(B, A, P, D)
        return out


class MotionTransformerDecoder(nn.Module):
    def __init__(self, pc_range=None, embed_dims=256, transformerlayers=None, num_layers=3, **kwargs):
        super(MotionTransformerDecoder, self).__init__()
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.intention_interaction_layers = IntentionInteraction()
        self.track_agent_interaction_layers = nn.ModuleList([TrackAgentInteraction() for i in range(self.num_layers)])
        self.map_interaction_layers = nn.ModuleList([MapInteraction() for i in range(self.num_layers)])
        self.bev_interaction_layers = nn.ModuleList(
            [
                MotionTransformerAttentionLayer(
                    batch_first=True,
                    attn_cfgs=[
                        {
                            "type": "MotionDeformableAttention",
                            "num_steps": 12,
                            "embed_dims": 256,
                            "num_levels": 1,
                            "num_heads": 8,
                            "num_points": 4,
                            "sample_index": -1,
                        }
                    ],
                    feedforward_channels=512,
                    operation_order=("cross_attn", "norm", "ffn", "norm"),
                )
                for i in range(self.num_layers)
            ]
        )

        self.static_dynamic_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.dynamic_embed_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 3, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.in_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.out_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 4, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )

    def forward(
        self,
        track_query,
        lane_query,
        track_query_pos=None,
        lane_query_pos=None,
        track_bbox_results=None,
        bev_embed=None,
        reference_trajs=None,
        traj_reg_branches=None,
        agent_level_embedding=None,
        scene_level_ego_embedding=None,
        scene_level_offset_embedding=None,
        learnable_embed=None,
        agent_level_embedding_layer=None,
        scene_level_ego_embedding_layer=None,
        scene_level_offset_embedding_layer=None,
        **kwargs,
    ):
        intermediate = []
        intermediate_reference_trajs = []

        B, _, P, D = agent_level_embedding.shape
        track_query_bc = track_query.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)
        track_query_pos_bc = track_query_pos.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)

        agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)
        static_intention_embed = agent_level_embedding + scene_level_offset_embedding + learnable_embed
        reference_trajs_input = reference_trajs.unsqueeze(4).detach()

        query_embed = torch.zeros_like(static_intention_embed)
        for lid in range(self.num_layers):
            dynamic_query_embed = self.dynamic_embed_fuser(
                torch.cat([agent_level_embedding, scene_level_offset_embedding, scene_level_ego_embedding], dim=-1)
            )

            query_embed_intention = self.static_dynamic_fuser(
                torch.cat([static_intention_embed, dynamic_query_embed], dim=-1)
            )  # (B, A, P, D)

            query_embed = self.in_query_fuser(torch.cat([query_embed, query_embed_intention], dim=-1))

            track_query_embed = self.track_agent_interaction_layers[lid](
                query_embed, track_query, query_pos=track_query_pos_bc, key_pos=track_query_pos
            )

            map_query_embed = self.map_interaction_layers[lid](
                query_embed, lane_query, query_pos=track_query_pos_bc, key_pos=lane_query_pos
            )

            bev_query_embed = self.bev_interaction_layers[lid](
                query_embed,
                value=bev_embed,
                query_pos=track_query_pos_bc,
                bbox_results=track_bbox_results,
                reference_trajs=reference_trajs_input,
                **kwargs,
            )

            query_embed = [track_query_embed, map_query_embed, bev_query_embed, track_query_bc + track_query_pos_bc]
            query_embed = torch.cat(query_embed, dim=-1)
            query_embed = self.out_query_fuser(query_embed)

            if traj_reg_branches is not None:
                tmp = traj_reg_branches[lid](query_embed)
                bs, n_agent, n_modes, n_steps, _ = reference_trajs.shape
                tmp = tmp.view(bs, n_agent, n_modes, n_steps, -1)

                tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
                new_reference_trajs = torch.zeros_like(reference_trajs)
                new_reference_trajs = tmp[..., :2]
                reference_trajs = new_reference_trajs.detach()
                reference_trajs_input = reference_trajs.unsqueeze(4)  # BS NUM_AGENT NUM_MODE 12 NUM_LEVEL  2

                ep_offset_embed = reference_trajs.detach()
                ep_ego_embed = (
                    trajectory_coordinate_transform(
                        reference_trajs.unsqueeze(2),
                        track_bbox_results,
                        with_translation_transform=True,
                        with_rotation_transform=False,
                    )
                    .squeeze(2)
                    .detach()
                )
                ep_agent_embed = (
                    trajectory_coordinate_transform(
                        reference_trajs.unsqueeze(2),
                        track_bbox_results,
                        with_translation_transform=False,
                        with_rotation_transform=True,
                    )
                    .squeeze(2)
                    .detach()
                )

                agent_level_embedding = agent_level_embedding_layer(
                    pos2posemb2d(norm_points(ep_agent_embed[..., -1, :], self.pc_range))
                )
                scene_level_ego_embedding = scene_level_ego_embedding_layer(
                    pos2posemb2d(norm_points(ep_ego_embed[..., -1, :], self.pc_range))
                )
                scene_level_offset_embedding = scene_level_offset_embedding_layer(
                    pos2posemb2d(norm_points(ep_offset_embed[..., -1, :], self.pc_range))
                )

                intermediate.append(query_embed)
                intermediate_reference_trajs.append(reference_trajs)

        return torch.stack(intermediate), torch.stack(intermediate_reference_trajs)
