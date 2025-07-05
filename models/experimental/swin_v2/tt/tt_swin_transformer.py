# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.experimental.swin_v2.tt.tt_swin_transformer_block_v2 import TtSwinTransformerBlockV2
from models.experimental.swin_v2.tt.tt_patchmerging_v2 import TtPatchMergingV2
import ttnn
from models.experimental.swin_v2.tt.common import Conv


class TtSwinTransformer:
    def __init__(
        self,
        device,
        parameters,
        patch_size,
        embed_dim,
        depths,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        num_classes=1000,
        block=None,
        norm_layer=None,
        attn_mask_tuple=None,
    ):
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.device = device
        self.parameters = parameters
        self.depths = depths
        self.embed_dim = embed_dim
        self.blocks = block
        self.norm_layer = norm_layer

        self.conv2d = Conv([4, 4, 0, 0], parameters=parameters["features"][0][0], reshard=True, device=self.device)
        if block is None:
            self.block = TtSwinTransformerBlockV2

        self.downsample_layer = TtPatchMergingV2
        self.layers = []
        index = 0
        for i_stage in range(1, 8, 2):
            stage = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[index]):
                stage.append(
                    self.block(
                        device=device,
                        parameters=parameters["features"][i_stage][i_layer],
                        dim=dim,
                        num_heads=num_heads[index],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        attn_mask=attn_mask_tuple[index],
                    )
                )
            self.layers.append(stage)

            if i_stage != 7:
                self.layers.append(self.downsample_layer(device, parameters["features"][i_stage + 1], dim))
            index += 1

    def __call__(self, x):
        N, C, H, W = x.shape
        min_channels = 16
        if C < min_channels:
            channel_padding_needed = min_channels - C
            nchw = ttnn.pad(x, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
        else:
            nchw = x
        nhwc = ttnn.permute(nchw, (0, 2, 3, 1))  # , memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(nchw)
        ttnn.deallocate(x)
        nhwc = ttnn.reallocate(nhwc)
        x = self.conv2d(self.device, nhwc)
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        if self.norm_layer is None:
            x = ttnn.layer_norm(
                x,
                weight=self.parameters.features[0][2].weight,
                bias=self.parameters.features[0][2].bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            pass

        for i_stage in range(len(self.layers)):
            if i_stage % 2 == 0:
                for j in range(len(self.layers[i_stage])):
                    x = self.layers[i_stage][j](x)
            else:
                x = self.layers[i_stage](x)

        if self.norm_layer is None:
            x = ttnn.layer_norm(
                x,
                weight=self.parameters.norm.weight,
                bias=self.parameters.norm.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            pass

        x = ttnn.permute(x, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.global_avg_pool2d(x, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.reshape(x, (x.shape[0], -1), memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.linear(
            x,
            self.parameters.head.weight,
            bias=self.parameters.head.bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return x
