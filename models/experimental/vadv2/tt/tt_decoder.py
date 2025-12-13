# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import copy
import math
import warnings

import ttnn
from models.experimental.vadv2.tt.tt_base_transformer_layer import TtBaseTransformerLayer
from models.experimental.vadv2.tt.tt_utils import inverse_sigmoid


def _safe_int(value):
    return int(value) if not isinstance(value, int) else value


def _extract_shape_3d(tensor):
    try:
        padded_shape = tensor.padded_shape
    except AttributeError:
        return None
    dims = tuple(_safe_int(dim) for dim in padded_shape)
    if len(dims) != 3:
        return None
    return dims


def _compute_sharding_meta(tensor, batch_first):
    dims = _extract_shape_3d(tensor)
    if dims is None:
        return None
    if batch_first:
        batch_size, seq_len, hidden_size = dims
    else:
        seq_len, batch_size, hidden_size = dims
    if batch_size == 0 or seq_len == 0 or hidden_size == 0:
        return None
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
    }


def _create_width_sharded_config(meta):
    tile = ttnn.TILE_SIZE
    hidden_size = meta["hidden_size"]
    num_cores = max(1, math.ceil(hidden_size / tile))
    core_grid_x = min(8, num_cores)
    core_grid_y = max(1, math.ceil(num_cores / core_grid_x))
    total_cores = core_grid_x * core_grid_y
    shard_width = math.ceil(hidden_size / total_cores)
    core_grid = ttnn.CoreGrid(y=core_grid_y, x=core_grid_x)
    meta["core_grid"] = core_grid
    meta["total_cores"] = total_cores
    meta["shard_width"] = shard_width
    return ttnn.create_sharded_memory_config(
        shape=[meta["batch_size"] * meta["seq_len"], shard_width],
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _create_layernorm_program_config(meta):
    tile = ttnn.TILE_SIZE
    core_grid = meta["core_grid"]
    block_h = max(1, math.ceil(meta["seq_len"] / tile))
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        subblock_w=1,
        block_h=block_h,
        block_w=1,
        inplace=False,
    )


def prepare_width_sharded_for_layernorm(tensor, batch_first):
    meta = _compute_sharding_meta(tensor, batch_first)
    if meta is None:
        return tensor, None, None
    if not batch_first:
        return tensor, None, None
    if meta["seq_len"] > 128:
        return tensor, None, None
    width_cfg = _create_width_sharded_config(meta)
    tensor_sharded = ttnn.to_memory_config(tensor, width_cfg)
    program_config = _create_layernorm_program_config(meta)
    return tensor_sharded, width_cfg, program_config


def prepare_width_sharded(tensor, batch_first):
    sharded, cfg, _ = prepare_width_sharded_for_layernorm(tensor, batch_first)
    return sharded, cfg


class TtWidthShardedTransformerLayer(TtBaseTransformerLayer):
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
        elif isinstance(attn_masks, ttnn.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in {self.__class__.__name__} ")
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
                sharded_query, width_cfg, program_cfg = prepare_width_sharded_for_layernorm(query, self.batch_first)
                if width_cfg is not None and program_cfg is not None:
                    query = ttnn.layer_norm(
                        sharded_query,
                        weight=self.params.norms[f"norm{norm_index}"].weight,
                        bias=self.params.norms[f"norm{norm_index}"].bias,
                        memory_config=width_cfg,
                        program_config=program_cfg,
                    )
                    query = ttnn.to_memory_config(query, ttnn.L1_MEMORY_CONFIG)
                else:
                    query = ttnn.layer_norm(
                        query,
                        weight=self.params.norms[f"norm{norm_index}"].weight,
                        bias=self.params.norms[f"norm{norm_index}"].bias,
                    )
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


class TtDetectionTransformerDecoder:
    def __init__(self, num_layers, embed_dim, num_heads, params, params_branches, device):
        self.return_intermediate = True
        self.device = device
        self.params = params
        self.params_branches = params_branches
        self.layers = [
            TtWidthShardedTransformerLayer(
                params.layers[f"layer{i}"],
                self.device,
                attn_cfgs=[
                    {
                        "type": "MultiheadAttention",
                        "embed_dims": embed_dim,
                        "num_heads": num_heads,
                    },
                    {
                        "type": "CustomMSDeformableAttention",
                        "embed_dims": embed_dim,
                        "num_levels": 1,
                    },
                ],
                ffn_cfgs={
                    "type": "FFN",
                    "embed_dims": embed_dim,
                    "feedforward_channels": 512,
                    "num_fcs": 2,
                    "ffn_drop": 0.0,
                    "act_cfg": {"type": "ReLU", "inplace": True},
                },
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                norm_cfg={"type": "LN"},
                init_cfg=None,
                batch_first=False,
                kwargs={
                    "feedforward_channels": 512,
                    "act_cfg": {"type": "ReLU", "inplace": True},
                    "ffn_num_fcs": 2,
                },
            )
            for i in range(num_layers)
        ]

    def __call__(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        reg_branches=None,
        cls_branches=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2]
            reference_points_input = ttnn.unsqueeze(reference_points_input, 2)
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
            )
            output = ttnn.permute(output, (1, 0, 2))
            ttnn.ReadDeviceProfiler(self.device)

            if reg_branches is not None:
                # Select reg_branch layers for current lid
                layers = self.params_branches.reg_branches[str(lid)]

                tmp, tmp_width_cfg = prepare_width_sharded(output, batch_first=True)
                linear_mem_cfg = tmp_width_cfg or ttnn.L1_MEMORY_CONFIG
                for i in range(3):
                    tmp = ttnn.linear(
                        tmp,
                        layers[str(i)].weight,
                        bias=layers[str(i)].bias,
                        memory_config=linear_mem_cfg,
                    )
                    if i < 2:
                        tmp = ttnn.relu(tmp)
                if tmp_width_cfg is not None:
                    tmp = ttnn.to_memory_config(tmp, ttnn.L1_MEMORY_CONFIG)
                assert reference_points.shape[-1] == 3

                new_reference_points = ttnn.zeros_like(reference_points, memory_config=ttnn.L1_MEMORY_CONFIG)
                updated_xy = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])  # shape (..., 2)
                updated_z = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])  # shape (..., 1)

                new_reference_points = ttnn.concat([updated_xy, updated_z], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(tmp)
                new_reference_points = ttnn.sigmoid(new_reference_points, memory_config=ttnn.L1_MEMORY_CONFIG)

                reference_points = new_reference_points

            output = ttnn.permute(output, (1, 0, 2))
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            a = ttnn.stack(intermediate, dim=0)
            b = ttnn.stack(intermediate_reference_points, dim=0)
            return a, b
        return output, reference_points


class TtMapDetectionTransformerDecoder:
    def __init__(self, num_layers, embed_dim, num_heads, params, params_branches, device):
        self.return_intermediate = True
        self.device = device
        self.params = params
        self.params_branches = params_branches
        self.layers = [
            TtWidthShardedTransformerLayer(
                params.layers[f"layer{i}"],
                self.device,
                attn_cfgs=[
                    {
                        "type": "MultiheadAttention",
                        "embed_dims": embed_dim,
                        "num_heads": num_heads,
                    },
                    {
                        "type": "CustomMSDeformableAttention",
                        "embed_dims": embed_dim,
                        "num_levels": 1,
                    },
                ],
                ffn_cfgs={
                    "type": "FFN",
                    "embed_dims": embed_dim,
                    "feedforward_channels": 512,
                    "num_fcs": 2,
                    "ffn_drop": 0.0,
                    "act_cfg": {"type": "ReLU", "inplace": True},
                },
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                norm_cfg={"type": "LN"},
                init_cfg=None,
                batch_first=False,
                kwargs={
                    "feedforward_channels": 512,
                    "act_cfg": {"type": "ReLU", "inplace": True},
                    "ffn_num_fcs": 2,
                },
            )
            for i in range(num_layers)
        ]

    def __call__(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        map_reg_branches=None,
        cls_branches=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2]
            reference_points_input = ttnn.unsqueeze(reference_points_input, 2)
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
            )
            ttnn.ReadDeviceProfiler(self.device)  # Clear device profiler buffer after layer

            output = ttnn.permute(output, (1, 0, 2))

            if map_reg_branches is not None:
                layers = self.params_branches.map_reg_branches[str(lid)]

                tmp, tmp_width_cfg = prepare_width_sharded(output, batch_first=True)
                linear_mem_cfg = tmp_width_cfg or ttnn.L1_MEMORY_CONFIG
                for i in range(3):
                    tmp = ttnn.linear(
                        tmp,
                        layers[str(i)].weight,
                        bias=layers[str(i)].bias,
                        memory_config=linear_mem_cfg,
                    )
                    if i < 2:  # Apply ReLU after the first two layers
                        tmp = ttnn.relu(tmp)
                if tmp_width_cfg is not None:
                    tmp = ttnn.to_memory_config(tmp, ttnn.L1_MEMORY_CONFIG)

                assert reference_points.shape[-1] == 2

                new_reference_points = ttnn.zeros_like(reference_points, memory_config=ttnn.L1_MEMORY_CONFIG)

                updated_xy = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])

                new_reference_points = ttnn.concat([updated_xy], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

                ttnn.deallocate(tmp)
                new_reference_points = ttnn.sigmoid(new_reference_points)

                reference_points = new_reference_points

            ttnn.ReadDeviceProfiler(self.device)  # Clear device profiler buffer
            output = ttnn.permute(output, (1, 0, 2))
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            a = ttnn.stack(intermediate, dim=0)
            b = ttnn.stack(intermediate_reference_points, dim=0)
            return a, b
        return output, reference_points


class TtCustomTransformerDecoder:
    def __init__(self, params, device, num_layers, return_intermediate=False, embed_dim=256, num_heads=8):
        super(TtCustomTransformerDecoder, self).__init__()
        self.device = device
        self.params = params
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        self.layers = [
            TtWidthShardedTransformerLayer(
                params.layers[f"layer{i}"],
                self.device,
                attn_cfgs=[
                    {
                        "type": "MultiheadAttention",
                        "embed_dims": 256,
                        "num_heads": 8,
                    }
                ],
                ffn_cfgs={
                    "type": "FFN",
                    "embed_dims": 256,
                    "feedforward_channels": 512,
                    "num_fcs": 2,
                    "act_cfg": {"type": "ReLU", "inplace": True},
                },
                operation_order=("cross_attn", "norm", "ffn", "norm"),
                norm_cfg={"type": "LN"},
                init_cfg=None,
                batch_first=False,
                kwargs={"feedforward_channels": 512},
            )
            for i in range(num_layers)
        ]

    def __call__(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        key_padding_mask=None,
        *args,
        **kwargs,
    ):
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs,
            )

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return ttnn.stack(intermediate, dim=0)

        return query
