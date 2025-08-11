# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_24b.tt.vision_conv2d import TtMistralConv2dPatch
from models.common.rmsnorm import RMSNorm as RMSNorm
from models.tt_transformers.tt.distributed_norm import DistributedNorm

from models.tt_transformers.tt.common import position_ids_in_meshgrid_tt, generate_block_attention_mask_tt
from models.experimental.mistral_24b.tt.vision_rope import VisionRotarySetup as RotarySetup

from models.experimental.mistral_24b.tt.vision_pixtral_transformer import TtPixtralTransformer
import torch


class MistralVisionTower(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        configuration,
        weight_cache_path=None,
        return_intermediate=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.config = configuration

        self.image_size = configuration.vision_chunk_size
        self.patch_size = configuration.vision_patch_size
        self.width = configuration.vision_dim
        self.layers = configuration.vision_n_layers
        self.heads = configuration.vision_attn_n_heads
        self.vision_head_dim = configuration.vision_head_dim
        self.mlp_ratio = configuration.vision_mlp_ratio
        self.act_layer = configuration.vision_act_layer
        self.in_channels = configuration.vision_in_channels
        self.n_global_layers = configuration.vision_n_global_layers
        self.max_seq_len = configuration.max_seq_len
        self.return_intermediate = return_intermediate
        self.n_layers = configuration.vision_n_layers

        in_channels, out_channels, kernel_size, stride, bias = (
            3,
            configuration.vision_dim,
            configuration.vision_patch_size,
            configuration.vision_patch_size,
            False,
        )

        self.patch_conv = TtMistralConv2dPatch(
            mesh_device=self.mesh_device,
            state_dict=self.state_dict,
            state_dict_prefix=f"{state_dict_prefix}patch_conv.",
            dtype=self.dtype,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

        layer_norm = RMSNorm(
            device=mesh_device,
            dim=self.width,
            state_dict=self.state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_dtype=dtype,
            weight_key="ln_pre",
            is_distributed=configuration.is_distributed_norm,
        )

        self.ln_pre = DistributedNorm(
            layer_norm,
            configuration,
            TG=configuration.is_galaxy,
        )

        image_size = configuration.vision_image_size
        patch_size = configuration.vision_patch_size
        dim = configuration.vision_head_dim
        num_patches_per_dim = image_size // patch_size
        num_patches = num_patches_per_dim * num_patches_per_dim
        self.num_patches = num_patches
        print("MistralVisionTower RotarySetup initialized with:")
        print("self.dim:", configuration.head_dim)
        print("image_size:", image_size)
        print("patch_size:", patch_size)
        print("dim:", dim)
        print("num_patches_per_dim:", num_patches_per_dim)
        print("num_patches:", num_patches)
        print("self.n_layers:", self.n_layers)
        print("configuration.n_layers:", configuration.vision_n_layers)

        self.patch_positional_embedding = RotarySetup(
            self.mesh_device,
            1,
            dim,
            image_size,
            patch_size,
            num_patches,
            configuration.vision_rope_theta,
            scale_factor=None,
            orig_context_len=num_patches,
            datatype=dtype,
        )

        self.transformer = TtPixtralTransformer(
            mesh_device=self.mesh_device,
            state_dict=self.state_dict,
            state_dict_prefix=f"{state_dict_prefix}transformer.",
            weight_cache_path=configuration.weight_cache_path(dtype),
            dtype=self.dtype,
            configuration=configuration,
            layers=self.n_layers,
        )

    def forward(self, input_tensor):
        """
        input_tensor shape: (B, C, H, W)
        """
        print("MistralVisionTower forward called with input_tensor shape:", input_tensor.shape)

        patch_embeds = self.patch_conv(input_tensor)
        patch_embeds = ttnn.transpose(patch_embeds, 1, 2)
        patch_embeds = ttnn.reshape(
            patch_embeds, (1, self.width, self.image_size // self.patch_size, self.image_size // self.patch_size)
        )
        image_sizes = [(self.image_size, self.image_size)]

        patch_embeds_list = [
            ttnn.slice(
                patch_embeds,
                [0, 0, 0, 0],
                [1, self.width, size[0] // self.patch_size, size[1] // self.patch_size],
            )
            for size in image_sizes
        ]

        reshaped_patches = []
        for p in patch_embeds_list:
            p = ttnn.reshape(p, (1, self.width, -1))
            p = ttnn.transpose(p, 1, 2)
            reshaped_patches.append(p)

        patch_embeds = ttnn.concat(reshaped_patches, dim=0)

        # ln_pre RMS Norm
        mode = "decode"  # if self.max_seq_len <= 32 else "prefill"
        patch_embeds = self.ln_pre(patch_embeds, mode=mode)

        # # positional embeddings
        position_ids = position_ids_in_meshgrid_tt(
            patch_embeds_list,
            max_width=self.config.vision_image_size // self.config.vision_patch_size,
            device=self.mesh_device,
        )

        position_ids = ttnn.to_torch(position_ids).to(torch.long)
        position_embeddings = self.patch_positional_embedding.get_rot_mats(position_ids)

        attention_mask = generate_block_attention_mask_tt(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds, tt_device=self.mesh_device
        )

        patch_embeds = ttnn.unsqueeze(patch_embeds, 0)
        out = self.transformer(
            patch_embeds,
            mask=attention_mask,
        )
        return out
