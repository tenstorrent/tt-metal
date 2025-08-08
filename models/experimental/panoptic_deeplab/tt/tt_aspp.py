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
    elif norm_name.lower() == "ln":
        TILE_WIDTH = 32
        weight_shape = (1, 1, num_channels // TILE_WIDTH, TILE_WIDTH)

        weight = ttnn.ones(weight_shape, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        bias = ttnn.zeros(weight_shape, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        return lambda x: ttnn.layer_norm(x, weight=weight, bias=bias)
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
        pool_kernel_size,
    ):
        super(TtASPP, self).__init__()
        assert len(dilations) == 3, "ASPP expects 3 dilations, got {}".format(len(dilations))
        self.dropout = dropout
        use_bias = norm == ""
        self.conv_branches = []

        self.activation = get_ttnn_activation(activation)
        self.device = device
        self.pool_kernel_size = pool_kernel_size

        # Shared Method to create TtConv2d objects
        def create_ttconv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=True):
            param_dict = {
                "weight": torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16),
                "dilation": (dilation, dilation),
            }
            if bias:
                param_dict["bias"] = torch.empty(out_channels)
            parameters = TtConv2dParameters.from_torch(param_dict, device=self.device)

            return TtConv2d(parameters, stride=stride, padding=padding)

        # 1x1 conv
        conv = create_ttconv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            bias=use_bias,
        )
        norm_func = get_ttnn_norm(norm, out_channels, device=self.device)
        self.conv_branches.append((conv, norm_func))

        # weight_init.c2_xavier_fill(self.convs[-1]) check if this is needed

        for dilation in dilations:
            conv = create_ttconv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=(1, 1),
                padding=(dilation, dilation),
                dilation=dilation,
                bias=use_bias,
            )
            norm_func = get_ttnn_norm(norm, out_channels, device=self.device)
            self.conv_branches.append((conv, norm_func))
            # weight_init.c2_xavier_fill(self.convs[-1]) # check if this is needed

        self.pool_conv = create_ttconv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            bias=True,
        )

        # weight_init.c2_xavier_fill(image_pooling[1]) check if this is needed

        self.project_conv = create_ttconv2d(
            in_channels=5 * out_channels,  # Concatenation results in 5 branches
            out_channels=out_channels,
            kernel_size=1,
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            bias=use_bias,
        )

        self.project_norm = get_ttnn_norm(norm, out_channels, device=self.device)
        # weight_init.c2_xavier_fill(self.project) check if this is needed

    def forward(self, x):
        input_shape = x.shape
        N = x.shape[0]  # Batch size
        H = x.shape[1]
        W = x.shape[2]
        C = x.shape[3]  # Channels
        print("N", N, "H", H, "W", W, "C", C)

        if H % self.pool_kernel_size[0] or W % self.pool_kernel_size[1]:
            raise ValueError(
                "`pool_kernel_size` must be divisible by the shape of inputs. "
                "Input size: {} `pool_kernel_size`: {}".format(input_shape, self.pool_kernel_size)
            )

        res = []
        for conv, norm in self.conv_branches:
            branch_out = conv(x)
            print("Branch out tensor")
            print(branch_out)
            print()
            branch_out = norm(branch_out)
            branch_out = self.activation(branch_out)
            print(f"branch_out shape {branch_out.shape}")
            res.append(branch_out)

        input_shape = (1, 1, N * H * W, C)  # Reshape to (1, 1, N*H*W, C) for TTNN
        ttnn_perm = ttnn.permute(x, (0, 2, 3, 1))  # Convert to NHWC format for TTNN
        ttnn_reshape = ttnn_perm.reshape(input_shape)

        kernel_h, kernel_w = self.pool_kernel_size
        if kernel_h % 2 == 1 and kernel_w % 2 == 1:
            # Odd kernel - symmetric padding
            paddingSame = [(kernel_h - 1) // 2, (kernel_w - 1) // 2]
        else:
            # Even kernel - asymmetric padding
            pad_h = kernel_h - 1
            pad_w = kernel_w - 1
            paddingSame = [pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2]

        pooled = ttnn.avg_pool2d(
            input_tensor=ttnn_reshape,
            batch_size=N,
            input_h=H,
            input_w=W,
            channels=C,
            kernel_size=self.pool_kernel_size,  # Use full spatial dimensions for global pooling effect
            stride=[1, 1],
            padding=paddingSame,  # Adjust padding to ensure output size matches input size
            ceil_mode=False,
        )
        pooled = ttnn.reshape(pooled, (N, H, W, C))  # Reshape to (N, pooled_h, pooled_w, C)

        # height_sharded_config = ttnn.create_sharded_memory_config(
        #     pooled.shape,  # Use the tensor's current shape
        #     core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # Same grid as current
        #     strategy=ttnn.ShardStrategy.HEIGHT,
        #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #     use_height_and_width_as_shard_shape=True,
        # )

        for i in range(len(res)):
            res[i] = ttnn.to_memory_config(res[i], ttnn.DRAM_MEMORY_CONFIG)
            print(f"Memory config of res[{i}]: {res[i].memory_config()}")

        pooled = ttnn.to_memory_config(pooled, ttnn.DRAM_MEMORY_CONFIG)  # Convert to height-sharded memory config

        pooled = self.pool_conv(pooled)
        pooled = self.activation(pooled)

        # pooled = ttnn.to_memory_config(pooled, ttnn.DRAM_MEMORY_CONFIG)

        res.append(pooled)

        print()
        print("Starting memory config check!!!!!")
        print()

        for i, r in enumerate(res):
            print(f"Memory Config of res[{i}]: {r.memory_config()}")

        total_channels = sum(tensor.shape[-1] for tensor in res)

        # output_height_sharded_config = ttnn.create_sharded_memory_config(
        #     (res[0].shape[1] * res[0].shape[2], total_channels),  # (H*W, total_channels)
        #     core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        #     strategy=ttnn.ShardStrategy.HEIGHT,
        #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #     use_height_and_width_as_shard_shape=True,
        # )

        # res = ttnn.concat(res, dim=3, memory_config=output_height_sharded_config) #Maybe dim = 3 for NHWC layout in TTNN, it was dim = 1, getting OOM for now for HEIGHT_SHARDING
        res = ttnn.concat(
            res, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # Concatenate along the channel dimension (NHWC layout)

        print(res)

        res = self.project_conv(res)

        print("After project conv")
        print(res)

        res = self.project_norm(res)

        print("After project norm")
        print(res)

        res = self.activation(res)

        print("After activation")
        print(res)

        if self.dropout > 0:
            res = ttnn.experimental.dropout(res, probability=self.dropout, scale=1.0 / (1.0 - self.dropout), seed=42)

        print()
        print("Final res")
        print()

        print(res)

        return res
