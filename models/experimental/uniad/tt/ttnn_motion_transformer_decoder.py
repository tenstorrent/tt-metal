# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import copy
import warnings

import ttnn

from models.experimental.uniad.tt.ttnn_deformable_attention import multi_scale_deformable_attn_pytorch
from models.experimental.uniad.tt.ttnn_utils import trajectory_coordinate_transform, norm_points, pos2posemb2d
from models.experimental.uniad.tt.ttnn_ffn import TtFFN
from models.experimental.uniad.tt.ttnn_interaction import (
    TtIntentionInteraction,
    TtMapInteraction,
    TtTrackAgentInteraction,
)


class TtMotionDeformableAttention:
    def __init__(
        self,
        parameters,
        device,
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
        self.device = device
        self.parameters = parameters
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
        self.sampling_offsets = ttnn.linear
        self.attention_weights = ttnn.linear
        self.value_proj = ttnn.linear
        self.output_proj = [
            ttnn.linear,
            ttnn.layer_norm,
        ]

    def __call__(
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
        query = ttnn.reshape(query, (query.shape[0], query.shape[1] * query.shape[2], query.shape[3]))

        value = ttnn.permute(value, (1, 0, 2))
        bs, num_value, _ = value.shape
        assert ttnn.sum(spatial_shapes[:, 0] * spatial_shapes[:, 1]) == num_value

        value = self.value_proj(
            value,
            self.parameters.value_proj.weight,
            bias=self.parameters.value_proj.bias,
            dtype=ttnn.bfloat16,
        )
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))
        sampling_offsets = self.sampling_offsets(
            query,
            self.parameters.sampling_offsets.weight,
            bias=self.parameters.sampling_offsets.bias,
            dtype=ttnn.bfloat16,
        )
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_steps, self.num_levels, self.num_points, 2)
        )
        attention_weights = self.attention_weights(
            query,
            self.parameters.attention_weights.weight,
            bias=self.parameters.attention_weights.bias,
            dtype=ttnn.bfloat16,
        )
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_steps, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, dim=-1)

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_steps, self.num_levels, self.num_points)
        )

        if reference_trajs.shape[-1] == 2:
            reference_trajs = reference_trajs[:, :, :, self.sample_index :, :, :]
            reference_trajs_ego = self.agent_coords_to_ego_coords(ttnn.clone(reference_trajs), bbox_results)
            reference_trajs_ego = ttnn.reshape(
                reference_trajs_ego,
                (
                    reference_trajs_ego.shape[0],
                    reference_trajs_ego.shape[1] * reference_trajs_ego.shape[2],
                    reference_trajs_ego.shape[3],
                    reference_trajs_ego.shape[4],
                    reference_trajs_ego.shape[5],
                ),
            )
            reference_trajs_ego = ttnn.reshape(
                reference_trajs_ego,
                (
                    reference_trajs_ego.shape[0],
                    reference_trajs_ego.shape[1],
                    1,
                    reference_trajs_ego.shape[2],
                    reference_trajs_ego.shape[3],
                    1,
                    reference_trajs_ego.shape[4],
                ),
            )
            device = reference_trajs_ego.device()
            reference_trajs_ego = ttnn.to_torch(reference_trajs_ego)
            # TODO Raised issue for this operation - <https://github.com/tenstorrent/tt-metal/issues/25517>
            reference_trajs_ego[..., 0] -= self.bev_range[0]
            reference_trajs_ego[..., 1] -= self.bev_range[1]
            reference_trajs_ego[..., 0] /= self.bev_range[3] - self.bev_range[0]
            reference_trajs_ego[..., 1] /= self.bev_range[4] - self.bev_range[1]
            reference_trajs_ego = ttnn.from_torch(
                reference_trajs_ego, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
            offset_normalizer = ttnn.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)

            # we are making 7D to 5D to support ttnn add and ttnn div
            sampling_offsets = ttnn.squeeze(sampling_offsets, dim=0)
            sampling_offsets = ttnn.squeeze(sampling_offsets, dim=-3)
            reference_trajs_ego = ttnn.squeeze(reference_trajs_ego, dim=0)
            reference_trajs_ego = ttnn.squeeze(reference_trajs_ego, dim=-3)

            sampling_locations = ttnn.add(
                reference_trajs_ego,
                ttnn.div(
                    sampling_offsets,
                    ttnn.reshape(offset_normalizer, (1, 1, 1, offset_normalizer.shape[0], offset_normalizer.shape[1])),
                ),
            )

            sampling_locations = ttnn.unsqueeze(sampling_locations, dim=0)
            sampling_locations = ttnn.unsqueeze(sampling_locations, dim=-3)

            sampling_locations = ttnn.permute(
                sampling_locations, (0, 1, 3, 2, 4, 5, 6)  # "bs nq nh ns nl np c -> bs nq ns nh nl np c"
            )
            attention_weights = ttnn.permute(attention_weights, (0, 1, 3, 2, 4, 5))
            sampling_locations = ttnn.reshape(
                sampling_locations,
                (bs, num_query * self.num_steps, self.num_heads, self.num_levels, self.num_points, 2),
            )
            attention_weights = ttnn.reshape(
                attention_weights, (bs, num_query * self.num_steps, self.num_heads, self.num_levels, self.num_points)
            )

        else:
            raise ValueError(
                f"Last dim of reference_trajs must be" f" 2 or 4, but get {reference_trajs.shape[-1]} instead."
            )

        output = multi_scale_deformable_attn_pytorch(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
            self.device,
        )

        output = ttnn.reshape(output, (bs, num_query, self.num_steps, -1))
        output = ttnn.reshape(output, (output.shape[0], output.shape[1], output.shape[2] * output.shape[3]))
        for i in range(2):
            if self.output_proj[i] == ttnn.linear:
                output = self.output_proj[i](
                    output,
                    self.parameters.output_proj[i].weight,
                    bias=self.parameters.output_proj[i].bias,
                    dtype=ttnn.bfloat16,
                )
            else:
                output = self.output_proj[i](
                    output, weight=self.parameters.output_proj[i].weight, bias=self.parameters.output_proj[i].bias
                )
        output = ttnn.relu(output)

        output = ttnn.reshape(output, (bs, num_agent, num_mode, -1))
        return output + identity

    def agent_coords_to_ego_coords(self, reference_trajs, bbox_results):
        batch_size = len(bbox_results)
        reference_trajs_ego = []
        for i in range(batch_size):
            boxes_3d, scores, labels, bbox_index, mask = bbox_results[i]
            det_centers = boxes_3d.gravity_center
            batch_reference_trajs = reference_trajs[i]
            temp = det_centers[:, :2]
            batch_reference_trajs += ttnn.reshape(temp, (temp.shape[0], 1, 1, 1, temp.shape[1]))
            reference_trajs_ego.append(batch_reference_trajs)
        return ttnn.stack(reference_trajs_ego, dim=0)


class TtMotionTransformerAttentionLayer:
    def __init__(
        self,
        parameters,
        device,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
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
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = []

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first
                attention = TtMotionDeformableAttention(
                    parameters=parameters.attentions[index],
                    device=device,
                    num_steps=12,
                    embed_dims=256,
                    num_levels=1,
                    num_heads=8,
                    num_points=4,
                    sample_index=-1,
                )

                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = []

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
            self.ffns.append(TtFFN(params=parameters.ffns[str(ffn_index)], device=device))

        self.norms = []
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(ttnn.layer_norm)

    def __call__(
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


class TtMotionTransformerDecoder:
    def __init__(
        self, parameters, device, pc_range=None, embed_dims=256, transformerlayers=None, num_layers=3, **kwargs
    ):
        self.parameters = parameters
        self.device = device
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.intention_interaction_layers = TtIntentionInteraction(
            parameters=parameters.intention_interaction_layers, device=device
        )
        self.track_agent_interaction_layers = [
            TtTrackAgentInteraction(parameters=parameters.track_agent_interaction_layers[i], device=device)
            for i in range(self.num_layers)
        ]
        self.map_interaction_layers = [
            TtMapInteraction(parameters=parameters.map_interaction_layers[i], device=device)
            for i in range(self.num_layers)
        ]
        self.bev_interaction_layers = [
            TtMotionTransformerAttentionLayer(
                parameters=parameters.bev_interaction_layers[i],
                device=device,
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

        self.static_dynamic_fuser = [
            ttnn.linear,
            ttnn.relu,
            ttnn.linear,
        ]
        self.dynamic_embed_fuser = [
            ttnn.linear,
            ttnn.relu,
            ttnn.linear,
        ]
        self.in_query_fuser = [
            ttnn.linear,
            ttnn.relu,
            ttnn.linear,
        ]
        self.out_query_fuser = [
            ttnn.linear,
            ttnn.relu,
            ttnn.linear,
        ]

    def __call__(
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
        if kwargs["spatial_shapes"].dtype != ttnn.bfloat16:
            kwargs["spatial_shapes"] = ttnn.from_device(kwargs["spatial_shapes"])
            kwargs["spatial_shapes"] = ttnn.to_dtype(kwargs["spatial_shapes"], dtype=ttnn.bfloat16)
            kwargs["spatial_shapes"] = ttnn.to_device(kwargs["spatial_shapes"], device=self.device)

        intermediate = []
        intermediate_reference_trajs = []

        B, _, P, D = agent_level_embedding.shape
        track_query_bc = ttnn.expand(ttnn.unsqueeze(track_query, 2), (-1, -1, P, -1))
        track_query_pos_bc = ttnn.expand(ttnn.unsqueeze(track_query_pos, 2), (-1, -1, P, -1))

        # static intention embedding, which is imutable throughout all layers
        agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)
        static_intention_embed = agent_level_embedding + scene_level_offset_embedding + learnable_embed
        reference_trajs_input = ttnn.unsqueeze(reference_trajs, 4)

        query_embed = ttnn.zeros(
            static_intention_embed.shape,
            dtype=static_intention_embed.dtype,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
        )
        for lid in range(self.num_layers):
            # fuse static and dynamic intention embedding
            # the dynamic intention embedding is the output of the previous layer, which is initialized with anchor embedding

            for index, layer in enumerate(self.dynamic_embed_fuser):
                if index == 0:
                    dynamic_query_embed = layer(
                        ttnn.concat(
                            [agent_level_embedding, scene_level_offset_embedding, scene_level_ego_embedding], dim=-1
                        ),
                        self.parameters.dynamic_embed_fuser[index].weight,
                        bias=self.parameters.dynamic_embed_fuser[index].bias,
                        dtype=ttnn.bfloat16,
                    )
                else:
                    if layer == ttnn.relu:
                        dynamic_query_embed = layer(dynamic_query_embed)
                    else:
                        dynamic_query_embed = layer(
                            dynamic_query_embed,
                            self.parameters.dynamic_embed_fuser[index].weight,
                            bias=self.parameters.dynamic_embed_fuser[index].bias,
                            dtype=ttnn.bfloat16,
                        )

            for index, layer in enumerate(self.static_dynamic_fuser):
                if index == 0:
                    query_embed_intention = layer(
                        ttnn.concat([static_intention_embed, dynamic_query_embed], dim=-1),
                        self.parameters.static_dynamic_fuser[index].weight,
                        bias=self.parameters.static_dynamic_fuser[index].bias,
                        dtype=ttnn.bfloat16,
                    )
                else:
                    if layer == ttnn.relu:
                        query_embed_intention = layer(query_embed_intention)
                    else:
                        query_embed_intention = layer(
                            query_embed_intention,
                            self.parameters.static_dynamic_fuser[index].weight,
                            bias=self.parameters.static_dynamic_fuser[index].bias,
                            dtype=ttnn.bfloat16,
                        )

            for index, layer in enumerate(self.in_query_fuser):
                if index == 0:
                    query_embed = layer(
                        ttnn.concat([query_embed, query_embed_intention], dim=-1),
                        self.parameters.in_query_fuser[index].weight,
                        bias=self.parameters.in_query_fuser[index].bias,
                        dtype=ttnn.bfloat16,
                    )
                else:
                    if layer == ttnn.relu:
                        query_embed = layer(query_embed)
                    else:
                        query_embed = layer(
                            query_embed,
                            self.parameters.in_query_fuser[index].weight,
                            bias=self.parameters.in_query_fuser[index].bias,
                            dtype=ttnn.bfloat16,
                        )

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
            query_embed = ttnn.concat(query_embed, dim=-1)

            for index, layer in enumerate(self.out_query_fuser):
                if layer == ttnn.relu:
                    query_embed = layer(query_embed)
                else:
                    query_embed = layer(
                        query_embed,
                        self.parameters.out_query_fuser[index].weight,
                        bias=self.parameters.out_query_fuser[index].bias,
                        dtype=ttnn.bfloat16,
                    )

            if traj_reg_branches is not None:
                tmp = ttnn.clone(query_embed)
                for index in range(len(traj_reg_branches[lid])):
                    if index % 2 == 0:
                        tmp = ttnn.linear(
                            tmp,
                            traj_reg_branches[lid][index]["weight"],
                            bias=traj_reg_branches[lid][index]["bias"],
                            dtype=ttnn.bfloat16,
                        )
                    else:
                        tmp = ttnn.relu(tmp)

                bs, n_agent, n_modes, n_steps, _ = reference_trajs.shape
                tmp = ttnn.reshape(tmp, (bs, n_agent, n_modes, n_steps, -1))

                tmp_a = ttnn.clone(tmp[..., :2])
                tmp_b = ttnn.clone(tmp[..., 2:])
                tmp_a = ttnn.cumsum(tmp_a, dim=3)
                tmp = ttnn.concat([tmp_a, tmp_b], dim=-1)
                ttnn.deallocate(tmp_a)
                ttnn.deallocate(tmp_b)

                new_reference_trajs = ttnn.zeros(
                    reference_trajs.shape, dtype=reference_trajs.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
                )
                new_reference_trajs = tmp[..., :2]
                reference_trajs = new_reference_trajs
                reference_trajs_input = ttnn.unsqueeze(reference_trajs, 4)  # BS NUM_AGENT NUM_MODE 12 NUM_LEVEL  2

                ep_offset_embed = ttnn.clone(reference_trajs)
                ep_ego_embed = ttnn.squeeze(
                    trajectory_coordinate_transform(
                        ttnn.unsqueeze(reference_trajs, 2),
                        track_bbox_results,
                        with_translation_transform=True,
                        with_rotation_transform=False,
                    ),
                    2,
                )
                ep_agent_embed = ttnn.squeeze(
                    trajectory_coordinate_transform(
                        ttnn.unsqueeze(reference_trajs, 2),
                        track_bbox_results,
                        with_translation_transform=False,
                        with_rotation_transform=True,
                    ),
                    2,
                )

                agent_level_embedding = pos2posemb2d(norm_points(ep_agent_embed[..., -1, :], self.pc_range))
                for index in range(len(agent_level_embedding_layer)):
                    if index % 2 == 0:
                        agent_level_embedding = ttnn.linear(
                            agent_level_embedding,
                            agent_level_embedding_layer[index]["weight"],
                            bias=agent_level_embedding_layer[index]["bias"],
                            dtype=ttnn.bfloat16,
                        )
                    else:
                        agent_level_embedding = ttnn.relu(agent_level_embedding)

                scene_level_ego_embedding = pos2posemb2d(norm_points(ep_ego_embed[..., -1, :], self.pc_range))
                for index in range(len(scene_level_ego_embedding_layer)):
                    if index % 2 == 0:
                        scene_level_ego_embedding = ttnn.linear(
                            scene_level_ego_embedding,
                            scene_level_ego_embedding_layer[index]["weight"],
                            bias=scene_level_ego_embedding_layer[index]["bias"],
                            dtype=ttnn.bfloat16,
                        )
                    else:
                        scene_level_ego_embedding = ttnn.relu(scene_level_ego_embedding)

                scene_level_offset_embedding = pos2posemb2d(norm_points(ep_offset_embed[..., -1, :], self.pc_range))
                for index in range(len(scene_level_ego_embedding_layer)):
                    if index % 2 == 0:
                        scene_level_offset_embedding = ttnn.linear(
                            scene_level_offset_embedding,
                            scene_level_offset_embedding_layer[index]["weight"],
                            bias=scene_level_offset_embedding_layer[index]["bias"],
                            dtype=ttnn.bfloat16,
                        )
                    else:
                        scene_level_offset_embedding = ttnn.relu(scene_level_offset_embedding)

                intermediate.append(query_embed)
                intermediate_reference_trajs.append(reference_trajs)

        return ttnn.stack(intermediate, dim=0), ttnn.stack(intermediate_reference_trajs, dim=0)
