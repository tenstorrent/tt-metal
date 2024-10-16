# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import collections
import torch
from loguru import logger

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


def pad_seq_one_tile(x, mesh_device):
    num_pad_tokens = 32

    pad_tensor = ttnn.as_tensor(
        torch.zeros(x.shape[0], x.shape[1], num_pad_tokens, x.shape[-1]),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    return (
        ttnn.concat([x, pad_tensor], dim=2),
        num_pad_tokens,
    )


def mask_tile_padding(attn_mask, ntok, npad, num_chunks):
    npad8 = 8 - (ntok % 8)
    n_extra_pad = npad - npad8
    for i in range(1, num_chunks + 1):
        attn_mask[:, :, i * (ntok + npad) - n_extra_pad : i * (ntok + npad), :] = torch.finfo(attn_mask.dtype).min
        attn_mask[:, :, :, i * (ntok + npad) - n_extra_pad : i * (ntok + npad)] = torch.finfo(attn_mask.dtype).min

    return attn_mask


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
        x = ttnn.reshape(x, (bsz * num_concurrent_media, num_chunks, ntok, dim))

        # tile embeddings
        x = self.pre_tile_pos_embed(x, ar)
        x = ttnn.reshape(x, (1, bsz * num_concurrent_media * num_chunks, ntok, dim))

        # apply cls token
        if not SKIP_EMBED:
            x = self.apply_class_embedding(x)
            ntok += 1

        # apply position embeddings
        x = ttnn.reshape(x, (bsz * num_concurrent_media, num_chunks, ntok, dim))
        if not SKIP_EMBED:
            x = self.apply_positional_embedding(x, ar)

        # BUG: layernorm takes 4d tensor -> 3d??
        x = self.ln_pre(x)
        x = ttnn.reshape(x, (bsz * num_concurrent_media, num_chunks, ntok, dim))  # BUG: required for above note
        npad, attn_mask = 0, None

        # NOTE: We need to do this padding because it creates a funky attention mask
        # return x
        logger.info(f"TT VisionEncoder: before tile padding x.shape: {x.shape}")
        x, npad = pad_seq_one_tile(x, self.mesh_device)
        logger.info(f"TT VisionEncoder: after tile padding x.shape: {x.shape}")
        fake_x = torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        attn_mask = encoder_utils.build_encoder_attention_mask(fake_x, ar, ntok, num_chunks, 1)
        attn_mask = mask_tile_padding(attn_mask, ntok, 32, num_chunks)
        logger.info(f"TT VisionEncoder: attn_mask.shape: {attn_mask.shape}")
        torch.save(attn_mask, "vision_encoder_attn_mask.pt")
        attn_mask = ttnn.as_tensor(
            attn_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        x = ttnn.reshape(x, (1, bsz * num_concurrent_media, -1, dim))
        logger.info(f"TT VisionEncoder: input to self.transformer x.shape: {x.shape}")
        x, int_x = self.transformer(x, return_intermediate=self.return_intermediate, mask=attn_mask)

        # # DEBUG transformer output
        # x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + 32, dim)
        # x = ttnn.slice(x, (0, 0, 0, 0), (bsz * num_concurrent_media, num_chunks, ntok, dim))

        x = self.ln_post(x)
        x = ttnn.reshape(x, (bsz * num_concurrent_media, num_chunks, ntok + npad, dim))
        x = self.post_tile_pos_embed(x, ar)
        x = ttnn.reshape(x, (1, bsz * num_concurrent_media, num_chunks * (ntok + npad), dim))
        x = self.global_transformer(x, mask=attn_mask)
        x = ttnn.reshape(x, (bsz * num_concurrent_media, num_chunks, ntok + npad, dim))
        x = ttnn.slice(x, (0, 0, 0, 0), (bsz * num_concurrent_media, num_chunks, ntok, dim))

        # adding back intermediate layer outputs
        # NOTE: We cannot do 5-dim tensors. It should be find to send back 4-dim as long as calling code knows.
        x = ttnn.reshape(x, (bsz * num_concurrent_media, num_chunks, ntok, dim))

        # NOTE: I can't correctly stack and reshape int_x because of ttnn page size limitations.
        # NOTE: this means I will have to modify calling code to know that int_x is not shuffled

        int_x = [ttnn.reshape(ix, (bsz * num_concurrent_media, num_chunks, ntok + npad, dim)) for ix in int_x]
        int_x = [ttnn.slice(ix, (0, 0, 0, 0), (bsz * num_concurrent_media, num_chunks, ntok, dim)) for ix in int_x]
        x = ttnn.concat(
            [
                x,
            ]
            + int_x,
            dim=-1,
        )
        return x
