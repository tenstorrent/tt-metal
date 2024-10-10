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
from models.demos.llama3.tt.llama_conv2d_patch import TtLlamaConv2dPatch
from models.demos.llama3.tt.llama_tile_position_embedding import TtLlamaTilePositionEmbedding
from models.demos.llama3.tt.llama_layernorm import TtLayerNorm
from models.demos.llama3.tt.llama_image_transformer import TtLlamaImageTransformer

import importlib

encoder_utils = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.multimodal.encoder_utils"
)


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
        self.n_global_layers = configuration.vision_n_global_layers
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

        # TODO: Add back in all of the embeddings after Ammar is done :)
        # self.class_embedding = ttnn.as_tensor(
        #     state_dict[f"{state_dict_prefix}class_embedding"],
        #     dtype=dtype,
        #     device=mesh_device,
        #     layout=ttnn.TILE_LAYOUT,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        # )

        self.ln_post = TtLayerNorm(
            device=mesh_device,
            dim=self.width,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_post.",
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

        self.ln_pre = TtLayerNorm(
            device=mesh_device,
            dim=self.width,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_pre.",
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

        self.transformer = TtLlamaImageTransformer(
            mesh_device,
            state_dict,
            f"{state_dict_prefix}transformer.",
            weight_cache_path,
            dtype,
            configuration,
            self.layers,
            gated=False,
        )

        self.global_transformer = TtLlamaImageTransformer(
            mesh_device,
            state_dict,
            f"{state_dict_prefix}global_transformer.",
            weight_cache_path,
            dtype,
            configuration,
            self.n_global_layers,
            gated=True,
        )

        self.pre_tile_pos_embed = TtLlamaTilePositionEmbedding(
            mesh_device,
            state_dict,
            f"{state_dict_prefix}pre_tile_pos_embed.",
            dtype,
            num_tiles=self.max_num_tiles,
            width=self.width,
            gated=True,
        )

        self.post_tile_pos_embed = TtLlamaTilePositionEmbedding(
            mesh_device,
            state_dict,
            f"{state_dict_prefix}post_tile_pos_embed.",
            dtype,
            num_tiles=self.max_num_tiles,
            width=self.width,
            gated=True,
        )

    def forward(self, images, ar):
        assert isinstance(
            images, torch.Tensor
        ), "VisionEncoder input must be a torch tensor because of unfold in self.conv1"
        SKIP_EMBED = True
        if images.ndim == 5:
            num_concurrent_media = 1
            bsz, num_chunks, nch, w, h = images.shape
        else:
            bsz, num_concurrent_media, num_chunks, nch, w, h = images.shape

        images = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        ar = ar.reshape(bsz * num_concurrent_media, 2)

        # patch embedding
        x = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        x = self.conv1(x)  # shape = [*, width, grid ** 2]
        # NOTE: at this point, x is a ttnn Tensor!
        # _, ntok, dim = x.shape
        ntok, dim = x.shape[2], x.shape[3]
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)

        # tile embeddings
        x = self.pre_tile_pos_embed(x, ar)
        x = x.reshape(1, bsz * num_concurrent_media * num_chunks, ntok, dim)

        # apply cls token
        if not SKIP_EMBED:
            x = self.apply_class_embedding(x)
            ntok += 1

        # apply position embeddings
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)
        if not SKIP_EMBED:
            x = self.apply_positional_embedding(x, ar)

        # BUG: layernorm takes 4d tensor -> 3d??
        x = self.ln_pre(x)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)  # BUG: required for above note
        npad, attn_mask = 0, None
        # x, npad = expand_num_tokens_to_mult8(x)
        # TODO: See if we need to do this padding (expand/contract)
        npad = 0
        fake_x = torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        attn_mask = encoder_utils.build_encoder_attention_mask(fake_x, ar, ntok, num_chunks, 1)
        attn_mask = ttnn.as_tensor(
            attn_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # TODO: Should it replicate or shard?
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        x = x.reshape(1, bsz * num_concurrent_media, -1, dim)
        # breakpoint()
        x, int_x = self.transformer(x, return_intermediate=self.return_intermediate, mask=attn_mask)
        # TODO: int_x is a list of tensors, what do we do about that on TTNN side?
        # breakpoint()

        x = self.ln_post(x)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        x = self.post_tile_pos_embed(x, ar)
        x = x.reshape(1, bsz * num_concurrent_media, num_chunks * (ntok + npad), dim)
        x = self.global_transformer(x, mask=attn_mask)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        # x = contract_num_tokens_from_mult8(x, npad)

        # adding back intermediate layer outputs
        # NOTE: We cannot do 5-dim tensors. It should be find to send back 4-dim as long as calling code knows.
        # x = x.reshape(bsz, num_concurrent_media, num_chunks, ntok, dim)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)
        int_x = int_x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, -1)
        # int_x = contract_num_tokens_from_mult8(int_x, npad)
        # int_x = int_x.reshape(bsz, num_concurrent_media, num_chunks, ntok, -1)
        x = ttnn.concat([x, int_x], dim=-1)
        breakpoint()
        return x
