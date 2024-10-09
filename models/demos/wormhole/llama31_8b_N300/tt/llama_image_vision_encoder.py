# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch

import ttnn
from models.utility_functions import (
    nearest_32,
)
from models.common.lightweightmodule import LightweightModule
from models.demos.wormhole.llama31_8b_N300.tt.llama_conv2d_patch import TtLlamaConv2dPatch
from models.demos.wormhole.llama31_8b_N300.tt.llama_tile_position_embeddings import TtLlamaTilePositionEmbedding
from models.demos.wormhole.llama31_8b_N300.tt.llama_layernorm import TtLlamaLayerNorm
from models.demos.wormhole.llama31_8b_N300.tt.llama_image_transformer import TtLlamaImageTransformer


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class TtLlamaVisionEncoder(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        return_intermediate=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device

        self.max_num_tiles = configuration.vision_max_num_tiles
        self.image_size = to_2tuple(configuration.vision_chunk_size)
        self.patch_size = to_2tuple(configuration.vision_patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )

        self.width = configuration.vision_dim
        self.layers = configuration.vision_n_layers
        self.heads = configuration.vision_attn_n_heads
        self.mlp_ratio = configuration.vision_mlp_ratio
        self.act_layer = configuration.vision_act_layer
        self.in_channels = configuration.vision_in_channels
        self.n_global_layers = configuration.vision_global_n_layers
        self.global_model = True
        self.return_intermediate = return_intermediate

        self.conv1 = TtLlamaConv2dPatch(
            mesh_device,
            state_dict,
            f"{state_dict_prefix}conv1.",
            dtype,
            in_channels=self.in_channels,
            out_channels=self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.class_embedding = ttnn.as_tensor(
            state_dict[f"{state_dict_prefix}class_embedding"],
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # TODO: finish the rest of the model :)
        # self.positional_embedding = ttnn.as_tensor(

    def forward(self, x, return_intermediate=None, mask=None):
        """
        Different from reference impl in that if return_intermediates, it returns
        a list of intermediate tensors rather than a stack of intermediates.
        Outer code will have to be aware and handle this correctly.
        """
        seq_len = x.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        out = []
        for idx, r in enumerate(self.resblocks):
            if return_intermediate is not None and idx in return_intermediate:
                out.append(x)
            x = r(x, mask=mask)
        if return_intermediate is not None:
            return x, out
        return x
