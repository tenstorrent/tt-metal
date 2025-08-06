# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch

import ttnn

from models.experimental.panoptic_deeplab.tt.tt_conv2dWrapper import TtConv2d, TtConv2dParameters


def get_ttnn_activation(activation_name: str):
    """Returns a ttnn activation function."""
    if activation_name.lower() == "silu":
        return ttnn.silu
    elif activation_name.lower() == "relu":
        return ttnn.relu
    else:
        raise NotImplementedError(f"Activation '{activation_name}' not supported in ttnn.")


def get_ttnn_norm(norm_name: str, num_channels: int, device):
    """Returns a ttnn normalization function."""
    if norm_name.lower() == "gn":
        num_groups = 32
        weight = ttnn.ones((1, 1, 1, num_channels), device=device, layout=ttnn.TILE_LAYOUT)
        bias = ttnn.zeros((1, 1, 1, num_channels), device=device, layout=ttnn.TILE_LAYOUT)
        return lambda x: ttnn.group_norm(x, num_groups=num_groups, weight=weight, bias=bias)
    elif norm_name == "":
        return lambda x: x  # No-op
    else:
        raise NotImplementedError(f"Normalization '{norm_name}' not supported.")


class TtASPP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dilations,
        device: ttnn.MeshDevice,
        *,
        norm,
        activation,
        dropout: float = 0.0,
    ):
        super(TtASPP, self).__init__()
        assert len(dilations) == 3, "ASPP expects 3 dilations, got {}".format(len(dilations))
        self.dropout = dropout
        use_bias = norm == ""
        self.convs = nn.ModuleList()

        self.activation = get_ttnn_activation(activation)
        self.device = device

        # Shared Method to create TtConv2d objects
        def create_ttconv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=True):
            param_dict = {
                "weight": torch.empty(out_channels, in_channels, kernel_size, kernel_size),
                "dilation": (dilation, dilation),
            }
            if bias:
                param_dict["bias"] = torch.empty(out_channels)
            parameters = TtConv2dParameters.from_torch(param_dict, device=self.device)
            return TtConv2d(parameters, stride=stride, padding=padding)

        self.convs.append(
            nn.Sequential(  # Include activation and norm externally
                create_ttconv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    bias=use_bias,
                ),
                get_ttnn_norm(norm, out_channels, device=self.device),
                self.activation,
            )
        )
        # weight_init.c2_xavier_fill(self.convs[-1]) check if this is needed

        for dilation in dilations:
            self.convs.append(
                nn.Sequential(
                    create_ttconv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=dilation,
                        dilation=dilation,
                        bias=use_bias,
                    ),
                    get_ttnn_norm(norm, out_channels, device=self.device),
                    self.activation,
                )
            )
            # weight_init.c2_xavier_fill(self.convs[-1]) # check if this is needed

        image_pooling = nn.Sequential(
            ttnn.global_avg_pool2d(kernel_size=1, stride=1),
            create_ttconv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True,
            ),
            self.activation,
        )

        # weight_init.c2_xavier_fill(image_pooling[1]) check if this is needed

        self.convs.append(image_pooling)

        self.project = nn.Sequential(
            create_ttconv2d(
                in_channels=5 * out_channels,  # Concatenation results in 5 branches
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=use_bias,
            ),
            get_ttnn_norm(norm, out_channels, device=self.device),
            self.activation,
        )
        # weight_init.c2_xavier_fill(self.project) check if this is needed

    def forward(self, x):
        size = x.shape[-2:]
        H, W = size

        res = []
        for conv in self.convs:
            res.append(conv(x))

        scale_h = H // res[-1].shape[1]  # Integer division
        scale_w = W // res[-1].shape[2]  # Integer division

        res[-1] = ttnn.upsample(res[-1], scale_factor=(scale_h, scale_w), mode="bilinear")
        res = ttnn.concat(res, dim=3)  # Maybe dim = 3 for NHWC layout in TTNN, it was dim = 1
        res = self.project(res)
        res = ttnn.experimental.dropout(res, probability=self.dropout) if self.dropout > 0 else res
        return res
