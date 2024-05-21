# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn, Tensor
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from models.experimental.functional_swin_s.tt.tt_swin_transformer_block import TtSwinTransformerBlock
from models.experimental.functional_swin_s.tt.tt_patchmerging import TtPatchMerging
import ttnn
import tt_lib


class TtSwinTransformer:
    def __init__(
        self,
        parameters,
        device,
        patch_size,
        embed_dim,
        depths,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=1000,
        block=None,
        norm_layer=None,
    ):
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.device = device
        self.parameters = parameters
        self.depths = depths
        self.embed_dim = embed_dim
        self.blocks = block
        self.norm_layer = norm_layer

        self.conv2d = self.parameters.conv2d
        if block is None:
            self.block = TtSwinTransformerBlock

        self.downsample_layer = TtPatchMerging
        self.layers = []

        for i_stage in range(len(depths)):
            stage = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        device=device,
                        parameters=parameters,
                    )
                )
            self.layers.append(stage)
            if i_stage < (len(depths) - 1):
                self.layers.append(self.downsample_layer(dim, device, parameters))

        self.avgpool = tt_lib.fallback_ops.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def __call__(self, x):
        x = x.to(self.device)
        x = self.conv2d(x)
        x = ttnn.permute(x, (0, 2, 3, 1))
        if self.norm_layer is None:
            x = ttnn.layer_norm(x, weight=self.parameters.norm[""], bias=self.parameters.norm[""])
        else:
            pass

        layer_block_id = 0
        for i_stage in range(len(self.depths)):
            for i_layer in range(self.depths[i_stage]):
                x = self.layers[layer_block_id][i_layer](x)

        layer_block_id += 1
        if i_stage < (len(self.depths) - 1):
            x = self.layers[layer_block_id](x)
            layer_block_id += 1

        if self.norm_layer is None:
            x = ttnn.layer_norm(x, weight=self.parameters.norm[""], bias=self.parameters.norm[""])
        else:
            pass

        x = ttnn.permute(x, (0, 3, 1, 2))
        x = ttnn.to_torch(x)
        x = torch_to_tt_tensor_rm(x, self.device, put_on_device=True)
        x = self.avgpool(x)
        x = tt_to_torch_tensor(x)
        x = self.flatten(x)
        x = ttnn.from_torch(x, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        x = ttnn.linear(x, self.parameters[""], bias=self.parameters[""])
        return ttnn.from_device(x)
