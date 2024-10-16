# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import collections
import torch

import ttnn
from models.utility_functions import (
    nearest_32,
)
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3.tt.multimodal.llama_image_vision_encoder import TtLlamaVisionEncoder

from models.demos.falcon7b_common.tests.test_utils import (
    synchronize_devices,
)


class TtLlamaCrossAttentionTransformerVision(LightweightModule):
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
        self.model_config = configuration.get_model_config()

        self.dim = configuration.dim
        self.vision_dim = configuration.vision_dim
        self.image_res = configuration.vision_chunk_size
        self.patch_size = configuration.vision_patch_size
        self.configuration = configuration

        self.vision_encoder = TtLlamaVisionEncoder(
            mesh_device,
            state_dict,
            f"{state_dict_prefix}vision_encoder.",
            weight_cache_path=configuration.weight_cache_path(dtype),
            dtype=dtype,
            configuration=configuration,
            return_intermediate=return_intermediate,
        )

        torch_weight = lambda name, suffix: torch.transpose(
            self.state_dict[f"{state_dict_prefix}{name}.{suffix}"], -2, -1
        )
        torch_bias = lambda name, suffix: self.state_dict[f"{state_dict_prefix}{name}.{suffix}"]

        cache_name = lambda name, suffix: weight_cache_path / (state_dict_prefix + f".{name}.{suffix}")

        def shuffle_weight(weight):
            orig_shape = weight.shape
            weight = weight.transpose(-1, -2)
            w = torch.zeros_like(weight)
            w[..., : self.vision_dim] = weight[..., : self.vision_dim]
            w[..., self.vision_dim :] = (
                weight[..., self.vision_dim :]
                .view(-1, self.vision_dim, 5)
                .transpose(-1, -2)
                .reshape(-1, self.vision_dim * 5)
            )
            return w.transpose(-1, -2).view(orig_shape)

        as_interleaved_tensor = lambda name, suffix, type, dim: ttnn.as_tensor(
            shuffle_weight(torch_weight(name, suffix))
            if suffix == "weight"
            else torch_bias(name, suffix),  # Grab only the wX part of the name
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=dim)
            if dim is not None
            else ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name, suffix),
        )

        # Sharded weights
        self.vision_projection_weight = as_interleaved_tensor("vision_projection", "weight", dtype, dim=-1)
        self.vision_projection_bias = as_interleaved_tensor("vision_projection", "bias", ttnn.bfloat16, dim=-1)

    def forward(self, images, ar):
        vision_tokens = self.vision_encoder(images, ar)

        seq_len = vision_tokens.shape[-2]

        pc = self.model_config["VISION_PROJ_PROGCFG"](seq_len)

        vision_tokens = ttnn.linear(
            vision_tokens,
            self.vision_projection_weight,
            bias=self.vision_projection_bias,
            compute_kernel_config=self.configuration.compute_kernel_config_hifi4,
            core_grid=None,
            dtype=ttnn.bfloat16,
            program_config=pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        vision_tokens = ttnn.all_gather(vision_tokens, dim=3, num_links=1, topology=ttnn.Topology.Linear)

        return vision_tokens
