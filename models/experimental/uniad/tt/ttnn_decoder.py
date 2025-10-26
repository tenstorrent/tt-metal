# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.uniad.tt.ttnn_detr_transformer_decoder_layer import TtDetrTransformerDecoderLayer
from models.experimental.uniad.tt.ttnn_utils import inverse_sigmoid


class TtDetectionTransformerDecoder:
    def __init__(self, num_layers, embed_dim, num_heads, params, device=None):
        self.num_layers = num_layers
        self.return_intermediate = True
        self.device = device
        self.params = params

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

            if reg_branches is not None:
                # Select reg_branch layers for current lid
                layers = reg_branches[lid]

                tmp = output
                for i in range(5):
                    if i in [1, 3]:
                        tmp = ttnn.relu(tmp)
                    else:
                        tmp = ttnn.linear(
                            tmp,
                            layers[i]["weight"],
                            bias=layers[i]["bias"],
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                        )

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
