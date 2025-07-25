# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import copy
import warnings

import ttnn

from models.experimental.uniad.tt.ttnn_deformable_attention import multi_scale_deformable_attn_pytorch
from models.experimental.uniad.tt.ttnn_ffn import TtFFN


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
        dropout=0.1,
        bev_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size=[0.2, 0.2, 8],
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        self.device = device
        self.parameters = parameters
        # super().__init__()
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
        self.sampling_offsets = (
            ttnn.linear
        )  # nn.Linear(embed_dims, num_heads * num_steps * num_levels * num_points * 2)
        self.attention_weights = ttnn.linear  # nn.Linear(embed_dims, num_heads * num_steps * num_levels * num_points)
        self.value_proj = ttnn.linear  # nn.Linear(embed_dims, embed_dims)
        self.output_proj = [
            ttnn.linear,
            ttnn.layer_norm,  # nn.Linear(num_steps * embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True)
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
        query = ttnn.reshape(
            query, (query.shape[0], query.shape[1] * query.shape[2], query.shape[3])
        )  # torch.flatten(query, start_dim=1, end_dim=2)

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
        # bs, n_query, n_head, n_steps, N_level, N_points, 2
        # BS NUM_AGENT NUM_MODE 12 NUM_LEVEL  2
        ##check if we can do this in preprocess
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
            )  # permute([0,1,3,2,4,5,6])
            attention_weights = ttnn.permute(attention_weights, (0, 1, 3, 2, 4, 5))  # .permute([0,1,3,2,4,5])
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
        output = ttnn.reshape(
            output, (output.shape[0], output.shape[1], output.shape[2] * output.shape[3])
        )  # (start_dim=2, end_dim=3))
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

    def rot_2d(self, yaw):
        sy, cy = torch.sin(yaw), torch.cos(yaw)
        out = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).permute([2, 0, 1])
        return out


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
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        batch_first=False,
        **kwargs,
    ):
        deprecated_args = dict(
            feedforward_channels="feedforward_channels", ffn_dropout="ffn_drop", ffn_num_fcs="num_fcs"
        )
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
                )  # build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = []

        num_ffns = operation_order.count("ffn")
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ffn_cfgs  # ConfigDict(ffn_cfgs) added by me
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if "embed_dims" not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]["embed_dims"] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]["embed_dims"] == self.embed_dims
            self.ffns.append(
                TtFFN(params=parameters.ffns[ffn_index], device=device)
            )  # ,feedforward_channels=512,num_fcs=2,ffn_drop=0.1,act_cfg={'type': 'ReLU', 'inplace': True}))

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


class TtMapInteraction:
    """
    Modeling the interaction between the agent and the map
    """

    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, batch_first=True, norm_cfg=None, init_cfg=None):
        super().__init__()

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def __call__(self, query, key, query_pos=None, key_pos=None):
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
