# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from torch import nn
from models.experimental.swin_s.tt.tt_swin_transformer_block import TtSwinTransformerBlock
from models.experimental.swin_s.tt.tt_patchmerging import TtPatchMerging
import ttnn
from models.experimental.swin_s.tt.common import Conv


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

        self.conv2d = Conv([4, 4, 0, 0], parameters=parameters["features"][0][0], reshard=True)
        if block is None:
            self.block = TtSwinTransformerBlock

        self.downsample_layer = TtPatchMerging
        self.layers = []
        index = 0
        for i_stage in range(1, 8, 2):
            stage = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[index]):
                stage.append(
                    self.block(
                        device,
                        parameters["features"][i_stage][i_layer],
                        dim,
                        num_heads[index],
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

        self.flatten = nn.Flatten(1)

    def __call__(self, input_tensor):
        # conv starts
        input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = self.conv2d(self.device, input_tensor)
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        # conv ends

        if self.norm_layer is None:
            output_tensor = ttnn.layer_norm(
                output_tensor,
                weight=self.parameters.features[0][2].weight,
                bias=self.parameters.features[0][2].bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            pass

        for i_stage in range(len(self.layers)):
            if i_stage % 2 == 0:
                for j in range(len(self.layers[i_stage])):
                    output_tensor = self.layers[i_stage][j](output_tensor)
                ttnn.DumpDeviceProfiler(self.device)
            else:
                output_tensor = self.layers[i_stage](output_tensor)

        ttnn.DumpDeviceProfiler(self.device)
        if self.norm_layer is None:
            output_tensor = ttnn.layer_norm(
                output_tensor,
                weight=self.parameters.norm.weight,
                bias=self.parameters.norm.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            pass
        output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        # AdaptiveAvgPool2d starts
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.global_avg_pool2d(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        # AdaptiveAvgPool2d  ends
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.reshape(
            output_tensor, (output_tensor.shape[0], -1)
        )  # Replace for flatten, self.flatten(output_tensor)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.linear(
            output_tensor,
            self.parameters.head.weight,
            bias=self.parameters.head.bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return output_tensor
