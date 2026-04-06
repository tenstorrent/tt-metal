# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Modified from the Panoptic-DeepLab implementation in Detectron2 library
# https://github.com/facebookresearch/detectron2/tree/main/projects/Panoptic-DeepLab
# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from models.experimental.panoptic_deeplab.reference.pytorch_conv2d_wrapper import Conv2d


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": lambda channels: nn.BatchNorm2d(channels),
            "SyncBN": lambda channels: nn.SyncBatchNorm(channels),
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dilations,
        *,
        norm,
        activation,
        pool_kernel_size=None,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels (int): number of input channels for ASPP.
            out_channels (int): number of output channels.
            dilations (list): a list of 3 dilations in ASPP.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format. norm is
                applied to all conv layers except the conv following
                global average pooling.
            activation (callable): activation function.
            pool_kernel_size (tuple, list): the average pooling size (kh, kw)
                for image pooling layer in ASPP. If set to None, it always
                performs global average pooling. If not None, it must be
                divisible by the shape of inputs in forward(). It is recommended
                to use a fixed input feature size in training, and set this
                option to match this size, so that it performs global average
                pooling in training, and the size of the pooling window stays
                consistent in inference.
            dropout (float): apply dropout on the output of ASPP. It is used in
                the official DeepLab implementation with a rate of 0.1:
                https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/model.py#L532  # noqa
            use_depthwise_separable_conv (bool): use DepthwiseSeparableConv2d
                for 3x3 convs in ASPP, proposed in :paper:`DeepLabV3+`.
        """
        super(ASPP, self).__init__()
        assert len(dilations) == 3, "ASPP expects 3 dilations, got {}".format(len(dilations))
        self.pool_kernel_size = pool_kernel_size
        self.dropout = dropout
        use_bias = norm == ""

        self.convs = nn.ModuleList()

        # conv 1x1
        self.convs.append(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=use_bias,
                norm=get_norm(norm, out_channels),
                activation=deepcopy(activation),
            )
        )

        # Dilation convs
        for dilation in dilations:
            self.convs.append(
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=use_bias,
                    norm=get_norm(norm, out_channels),
                    activation=deepcopy(activation),
                )
            )

        # Image pooling branch
        pooling_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=pool_kernel_size, stride=1),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=True,  # PKL file expects bias=True for pooling branch
                norm=None,  # PKL file doesn't have normalization for pooling branch
                activation=deepcopy(activation),
            ),
        )
        self.convs.append(pooling_branch)

        # Project conv to concatenate all branches - use 'project' name to match pkl
        self.project = Conv2d(
            5 * out_channels,
            out_channels,
            kernel_size=1,
            bias=use_bias,
            norm=get_norm(norm, out_channels),
            activation=deepcopy(activation),
        )

    def forward(self, x):
        size = x.shape[-2:]
        if self.pool_kernel_size is not None:
            if size[0] % self.pool_kernel_size[0] or size[1] % self.pool_kernel_size[1]:
                raise ValueError(
                    "`pool_kernel_size` must be divisible by the shape of inputs. "
                    "Input size: {} `pool_kernel_size`: {}".format(size, self.pool_kernel_size)
                )
        res = []

        # Process first 4 convs (1x1 + 3 dilated convs)
        for i, conv in enumerate(self.convs[:-1]):
            res.append(conv(x))

        # Process pooling branch (convs[4]) with interpolation
        pool_out = self.convs[-1](x)
        pool_out = F.interpolate(pool_out, size=size, mode="bilinear", align_corners=False).to(x.dtype)
        res.append(pool_out)

        res = torch.cat(res, dim=1)

        res = self.project(res)

        res = F.dropout(res, self.dropout, training=self.training) if self.dropout > 0 else res

        return res
