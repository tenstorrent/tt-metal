# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_24b.tt.vision_conv2d import TtMistralConv2dPatch
from models.common.rmsnorm import RMSNorm as RMSNorm
from models.tt_transformers.tt.distributed_norm import DistributedNorm


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

        self.image_size = configuration.vision_chunk_size
        self.patch_size = configuration.vision_patch_size
        self.width = configuration.vision_dim
        self.layers = configuration.vision_n_layers
        self.heads = configuration.vision_attn_n_heads
        self.mlp_ratio = configuration.vision_mlp_ratio
        self.act_layer = configuration.vision_act_layer
        self.in_channels = configuration.vision_in_channels
        self.n_global_layers = configuration.vision_n_global_layers
        self.max_seq_len = configuration.max_seq_len
        self.return_intermediate = return_intermediate

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

        return patch_embeds
