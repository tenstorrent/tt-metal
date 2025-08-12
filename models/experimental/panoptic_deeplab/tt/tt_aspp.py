# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from math import floor
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
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
    ):
        super(TtASPP, self).__init__()
        assert len(dilations) == 3, "ASPP expects 3 dilations, got {}".format(len(dilations))
        self.dropout = dropout
        use_bias = False
        self.conv_branches = []

        self.activation = get_ttnn_activation(activation)
        self.device = device
        self.pool_kernel_size = pool_kernel_size

        self.shared_weight_tensor_kernel1 = shared_weight_tensor_kernel1
        self.shared_weight_tensor_kernel3 = shared_weight_tensor_kernel3
        self.shared_weight_tensor_kernel1_output5 = shared_weight_tensor_kernel1_output5

        # Shared Method to create TtConv2d objects
        def create_ttconv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False, isProjectConv=False
        ):
            if kernel_size == 1 and isProjectConv:
                weight = self.shared_weight_tensor_kernel1_output5
            elif kernel_size == 3:
                weight = self.shared_weight_tensor_kernel3
            else:
                weight = self.shared_weight_tensor_kernel1
            param_dict = {
                "weight": weight,
                "dilation": (dilation, dilation),
            }
            if bias:
                param_dict["bias"] = torch.empty(1, 1, 1, out_channels, dtype=torch.bfloat16)
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

        self.pool_conv = create_ttconv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            bias=False,
        )

        self.project_conv = create_ttconv2d(
            in_channels=5 * out_channels,  # Concatenation results in 5 branches
            out_channels=out_channels,
            kernel_size=1,
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            bias=use_bias,
            isProjectConv=True,
        )

        self.project_norm = get_ttnn_norm(norm, out_channels, device=self.device)

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
            # print("Branch out before conv")
            # print(x)
            branch_out = conv(x)
            print("Branch out after conv")
            print(branch_out)
            # print()
            branch_out = norm(branch_out)
            branch_out = self.activation(branch_out)
            print(f"branch_out shape {branch_out.shape}")
            # print(branch_out)
            res.append(branch_out)

        input_shape = (1, 1, N * H * W, C)  # Reshape to (1, 1, N*H*W, C) for TTNN
        # ttnn_perm = ttnn.permute(x, (0, 2, 3, 1))  # Convert to NHWC format for TTNN
        ttnn_reshape = x.reshape(input_shape)
        print("TTNN Reshape shape: ", ttnn_reshape.shape)
        # kernel_h, kernel_w = self.pool_kernel_size
        # if kernel_h % 2 == 1 and kernel_w % 2 == 1:
        #     # Odd kernel - symmetric padding
        #     paddingSame = [(kernel_h - 1) // 2, (kernel_w - 1) // 2]
        # else:
        #     # Even kernel - asymmetric padding
        #     pad_h = kernel_h - 1
        #     pad_w = kernel_w - 1
        #     paddingSame = [pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2]

        pooled = ttnn.avg_pool2d(
            input_tensor=ttnn_reshape,
            batch_size=N,
            input_h=H,
            input_w=W,
            channels=C,
            kernel_size=self.pool_kernel_size,  # Use full spatial dimensions for global pooling effect
            stride=[1, 1],
            padding=[0, 0],
            ceil_mode=False,
        )

        output_h = floor(H + 0 - self.pool_kernel_size[0]) + 1
        output_w = floor(W + 0 - self.pool_kernel_size[1]) + 1
        print("Output height:", output_h, "Output width:", output_w)
        print("pooled tensor shape", pooled.shape)

        pooled = ttnn.reshape(pooled, (N, output_h, output_w, C))  # Reshape to (N, pooled_h, pooled_w, C)
        # height_sharded_config = ttnn.create_sharded_memory_config(
        #     pooled.shape,  # Use the tensor's current shape
        #     core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # Same grid as current
        #     strategy=ttnn.ShardStrategy.HEIGHT,
        #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #     use_height_and_width_as_shard_shape=True,
        # )

        for i in range(len(res)):
            res[i] = ttnn.to_memory_config(res[i], ttnn.DRAM_MEMORY_CONFIG)
            # print(f"Memory config of res[{i}]: {res[i].memory_config()}")
        pooled = ttnn.to_memory_config(pooled, ttnn.DRAM_MEMORY_CONFIG)

        print("Pooled layout")
        print(pooled.get_layout())

        # pooled = ttnn.tilize_with_zero_padding(pooled)
        # pooled = ttnn.tilize_with_val_padding(
        #     pooled,
        #     output_tensor_shape = (N, H, W, C),
        #     pad_value=0.0,
        #     use_multicore=True,  # Use multicore tiling
        # )

        # pooled = ttnn.pad(
        #     pooled,
        #     padding=((0,0), (1,2), (1,2), (0,0)),  # No padding needed for global pooling
        #     value=0.0,
        # )

        print("Pooled tensor after padding")
        print(f"{pooled.shape=} {pooled.padded_shape=}")

        pooled = self.pool_conv(pooled)

        # unpad here
        # pooled = ttnn.unpad(pooled, (0, 0, 0, 0))  # Unpad the tensor to remove padding

        pooled = self.activation(pooled)

        # unpadded_shape_end = [N-1, output_h-1, output_w-1, C-1]  # Unpadded shape end indices
        # pooled = ttnn.untilize_with_unpadding(
        #     pooled,
        #     output_tensor_end=unpadded_shape_end,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG
        # )

        current_h, current_w = pooled.shape[1], pooled.shape[2]
        print(f"Current height: {current_h}, Current width: {current_w}")
        print("Pooled shape before upsample:", pooled.shape)
        print(pooled.memory_config())
        scale_factor = [H // current_h, W // current_w]
        print(f"Scale factor: {scale_factor}")

        # numTiles = C // 32

        # pooled = pooled.reshape(N, current_h, 32, numTiles)  # Reshape to (N, H, W, C) for upsampling

        pooled = ttnn.upsample(pooled, scale_factor=scale_factor, mode="bilinear")

        # pooled = ttnn.upsample(pooled, scale_factor=scale_factor, mode="bilinear")
        # print(pooled.shape)

        print("Pooled tensor")
        print(pooled)
        return pooled

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
