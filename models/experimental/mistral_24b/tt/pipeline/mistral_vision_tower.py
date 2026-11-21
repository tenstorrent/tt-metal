# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
This file implements the Vision Tower submodule specific for the Mistral-Small-3.1-24B-Instruct-2503 model.
This pipeline constructs the vision tower from vision model architecture.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_24b.tt.vision_conv2d import TtMistralConv2dPatch
from models.experimental.mistral_24b.tt.rmsnorm import RMSNorm

from models.tt_transformers.tt.common import position_ids_in_meshgrid_tt
from models.experimental.mistral_24b.tt.vision_rope import VisionRotarySetup as RotarySetup

from models.experimental.mistral_24b.tt.vision_pixtral_transformer import TtPixtralTransformer
from ttnn import ConcatMeshToTensor


class MistralVisionTower(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        state_dict_prefix,
        dtype,
        configuration,
        return_intermediate=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
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

        self.ln_pre = RMSNorm(
            device=mesh_device,
            dim=self.width,
            state_dict=self.state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_dtype=dtype,
            weight_key="ln_pre",
            is_distributed=False,
            simplified_rms=True,
        )

        image_size = configuration.vision_image_size
        patch_size = configuration.vision_patch_size
        dim = configuration.vision_head_dim
        num_patches_per_dim = image_size // patch_size
        num_patches = num_patches_per_dim * num_patches_per_dim
        self.num_patches = num_patches

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
            tt_ccl=tt_ccl,
            state_dict=self.state_dict,
            state_dict_prefix=f"{state_dict_prefix}transformer.",
            weight_cache_path=configuration.weight_cache_path(dtype),
            dtype=self.dtype,
            configuration=configuration,
            layers=self.n_layers,
        )

    def forward(self, input_tensor, image_sizes):
        """
        input_tensor shape: (B, C, H, W)
        image_sizes: List of tuples [(height, width), ...] for each image in the batch
        """
        if image_sizes is None or len(image_sizes) == 0:
            raise ValueError("image_sizes must be provided and non-empty")
        patch_embeds = self.patch_conv(input_tensor)
        patch_embeds = ttnn.transpose(patch_embeds, 1, 2)
        height, width = input_tensor.shape[-2], input_tensor.shape[-1]
        patch_embeds = ttnn.reshape(
            patch_embeds,
            [patch_embeds.shape[0], self.width, height // self.patch_size, width // self.patch_size],
        )

        patch_embeds_list = [
            ttnn.slice(
                patch_embeds,
                [i, 0, 0, 0],
                [i + 1, self.width, size[0] // self.patch_size, size[1] // self.patch_size],
            )
            for i, size in enumerate(image_sizes)
        ]

        mode = "prefill"
        max_width = self.config.vision_image_size // self.config.vision_patch_size
        per_image_outputs = []

        # Process each image independently so non-causal SDPA stays isolated per image
        # (matching Pixtral's block-diagonal image mask) and rotary embeddings stay bounded by self.num_patches.
        for p_2d in patch_embeds_list:
            p = ttnn.reshape(p_2d, (1, self.width, -1))
            p = ttnn.transpose(p, 1, 2)
            p = self.ln_pre(p, mode=mode)

            position_ids = position_ids_in_meshgrid_tt(
                [p_2d],
                max_width=max_width,
                device=self.mesh_device,
            )
            torch_position_ids = ttnn.to_torch(
                position_ids, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=0)
            )[: position_ids.shape[-1]]
            ttnn.deallocate(position_ids)

            position_embeddings = self.patch_positional_embedding.get_rot_mats(torch_position_ids)
            p = ttnn.unsqueeze(p, 0)
            img_out = self.transformer(p, position_embeddings=position_embeddings)
            ttnn.deallocate(position_embeddings[0])
            ttnn.deallocate(position_embeddings[1])
            per_image_outputs.append(img_out)

        if len(per_image_outputs) == 1:
            return per_image_outputs[0]
        return ttnn.concat(per_image_outputs, dim=2)
