# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.vadv2.tt.tt_base_transformer_layer import TtBaseTransformerLayer


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class TtDetectionTransformerDecoder:
    def __init__(self, num_layers, embed_dim, num_heads, params, device):
        self.return_intermediate = True
        self.device = device
        self.params = params
        self.layers = [
            TtBaseTransformerLayer(
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
                    "ffn_dropout": 0.1,
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

            if reg_branches is not None:
                output = ttnn.to_torch(output).float()
                # ss
                tmp = reg_branches[lid](output)
                tmp = ttnn.from_torch(output, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
                assert reference_points.shape[-1] == 2

                new_reference_points = ttnn.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = ttnn.sigmoid(new_reference_points, memory_config=ttnn.L1_MEMORY_CONFIG)

                reference_points = new_reference_points.detach()

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
    def __init__(self, num_layers, embed_dim, num_heads, params, device):
        self.return_intermediate = True
        self.device = device
        self.params = params
        self.layers = [
            TtBaseTransformerLayer(
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
                    "ffn_dropout": 0.1,
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

            if reg_branches is not None:
                output = ttnn.to_torch(output).float()
                # ss
                tmp = reg_branches[lid](output)
                tmp = ttnn.from_torch(output, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
                assert reference_points.shape[-1] == 2

                new_reference_points = ttnn.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = ttnn.sigmoid(new_reference_points, memory_config=ttnn.L1_MEMORY_CONFIG)

                reference_points = new_reference_points.detach()

            output = ttnn.permute(output, (1, 0, 2))
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            a = ttnn.stack(intermediate, dim=0)
            b = ttnn.stack(intermediate_reference_points, dim=0)
            return a, b
        return output, reference_points
