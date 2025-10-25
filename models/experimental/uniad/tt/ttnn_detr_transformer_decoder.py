# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import warnings
from models.experimental.uniad.tt.ttnn_ffn import TtFFN
from models.experimental.uniad.tt.ttnn_mha import TtMultiheadAttention
from models.experimental.uniad.tt.ttnn_detr_transformer_encoder import TtMultiScaleDeformableAttention
import copy


def inverse_sigmoid(x, eps: float = 1e-5):
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.clamp(x, min=0, max=1)
    x1 = ttnn.clamp(x, min=eps)
    if len(x.shape) == 3:
        x_temp = ttnn.ones(shape=[x.shape[0], x.shape[1], x.shape[2]], layout=ttnn.TILE_LAYOUT, device=x.device())
    else:
        x_temp = ttnn.ones(
            shape=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]], layout=ttnn.TILE_LAYOUT, device=x.device()
        )
    x_temp = x_temp - x
    x2 = ttnn.clamp(x_temp, min=eps)
    return ttnn.log(ttnn.div(x1, x2))


class TtDetrTransformerDecoderLayer:
    def __init__(
        self,
        params,
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
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        self.params = params

        deprecated_args = dict(feedforward_channels="feedforward_channels", ffn_num_fcs="num_fcs")
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments "
                    f"to a dict named `ffn_cfgs`. "
                )
                ffn_cfgs[new_name] = kwargs[ori_name]

        super(TtDetrTransformerDecoderLayer, self).__init__()

        self.batch_first = batch_first
        self.device = device

        assert set(operation_order) & set(["self_attn", "norm", "ffn", "cross_attn"]) == set(operation_order), (
            f"The operation_order of"
            f" {self.__class__.__name__} should "
            f"contains all four operation type "
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"
        )

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
        self.attentions = []

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first
                if attn_cfgs[index]["type"] == "MultiheadAttention":
                    type = attn_cfgs[index].pop("type")
                    attention = TtMultiheadAttention(params.attentions[0], device, **attn_cfgs[index])  # Changed here
                    attn_cfgs[index]["type"] = "MultiheadAttention"
                elif attn_cfgs[index]["type"] == "MultiScaleDeformableAttention":
                    type = attn_cfgs[index].pop("type")
                    attention = TtMultiScaleDeformableAttention(params.attentions[1], device, **attn_cfgs[index])
                    attn_cfgs[index]["type"] = "MultiScaleDeformableAttention"

                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

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
            self.ffns.append(TtFFN(params.ffns[0].ffn.ffn0, self.device))

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
                query = ttnn.layer_norm(
                    query,
                    weight=self.params.norms[norm_index].weight,
                    bias=self.params.norms[norm_index].bias,
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


class TtDeformableDetrTransformerDecoder:
    def __init__(self, num_layers, embed_dim, num_heads, params, device, params_branches=None):
        self.return_intermediate = True
        self.device = device
        self.params = params
        self.params_branches = params_branches
        self.layers = [
            TtDetrTransformerDecoderLayer(
                params.layers[i],
                self.device,
                attn_cfgs=[
                    {
                        "type": "MultiheadAttention",
                        "embed_dims": embed_dim,
                        "num_heads": num_heads,
                    },
                    {
                        "type": "MultiScaleDeformableAttention",
                        "embed_dims": embed_dim,
                        "num_levels": 4,
                    },
                ],
                ffn_cfgs={
                    "type": "FFN",
                    "embed_dims": embed_dim,
                    "feedforward_channels": 512,
                    "num_fcs": 2,
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
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        reg_branches=None,
        cls_branches=None,
        valid_ratios=None,
        level_start_index=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                valid_ratios_concat = ttnn.concat([valid_ratios, valid_ratios], dim=-1)
                reference_points_unsq = ttnn.unsqueeze(reference_points, dim=2)
                valid_ratios_unsq = ttnn.unsqueeze(valid_ratios_concat, dim=1)
                reference_points_input = ttnn.mul(reference_points_unsq, valid_ratios_unsq)
            else:
                assert reference_points.shape[-1] == 2
                reference_points_unsq = ttnn.unsqueeze(reference_points, dim=2)
                valid_ratios_unsq = ttnn.unsqueeze(valid_ratios, dim=1)
                reference_points_input = ttnn.mul(reference_points_unsq, valid_ratios_unsq)

            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            output = ttnn.permute(output, (1, 0, 2))

            if reg_branches is not None:
                layers = self.params_branches[lid]
                tmp = output
                for i in range(0, 5, 2):
                    tmp = ttnn.linear(
                        tmp,
                        layers[i]["weight"],
                        bias=layers[i]["bias"],
                    )
                    if i <= 2:
                        tmp = ttnn.relu(tmp)

                if reference_points.shape[-1] == 4:
                    inv = inverse_sigmoid(reference_points)
                    new_reference_points = tmp + inv
                    new_reference_points = ttnn.sigmoid(new_reference_points)
                else:
                    assert reference_points.shape[-1] == 2
                    inv = inverse_sigmoid(reference_points)
                    new_reference_points = tmp

                    new_reference_points = ttnn.to_torch(new_reference_points)
                    tmp = ttnn.to_torch(tmp)
                    inv = ttnn.to_torch(inv)
                    new_reference_points[..., :2] = tmp[..., :2] + inv
                    new_reference_points = ttnn.from_torch(
                        new_reference_points, device=self.device, layout=ttnn.TILE_LAYOUT
                    )
                    new_reference_points = ttnn.sigmoid(new_reference_points)
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
