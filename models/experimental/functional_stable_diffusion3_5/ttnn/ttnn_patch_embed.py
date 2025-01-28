# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.functional_stable_diffusion3_5.ttnn.common import Conv


class ttnn_PatchEmbed:
    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        pos_embed_max_size=None,
        parameters=None,
    ):
        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = Conv([patch_size, patch_size, 0, 0], parameters=parameters.proj)
        if layer_norm:
            # This is not invoked in our call
            self.norm = ttnn.layer_norm
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

        # Calculate positional embeddings based on max size or default
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None
        elif pos_embed_type == "sincos":
            # pos_embed = get_2d_sincos_pos_embed(
            #     embed_dim, grid_size, base_size=self.base_size, interpolation_scale=self.interpolation_scale
            # )
            # persistent = True if pos_embed_max_size else False
            # self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=persistent)

            self.pos_embed = parameters["pos_embed"]  # we have stored in the parameters and loading it
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

    def cropped_pos_embed(self, height, width):
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}.")

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = ttnn.reshape(self.pos_embed, (1, self.pos_embed_max_size, self.pos_embed_max_size, -1))
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = ttnn.reshape(spatial_pos_embed, (1, -1, spatial_pos_embed.shape[-1]))
        spatial_pos_embed = ttnn.to_layout(spatial_pos_embed, layout=ttnn.TILE_LAYOUT)
        return spatial_pos_embed

    def __call__(self, device, latent):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2], latent.shape[-1]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        latent = ttnn.permute(latent, (0, 2, 3, 1))  # NCHW to NHWC
        latent = ttnn.to_layout(latent, layout=ttnn.ROW_MAJOR_LAYOUT)
        latent = self.proj(device, latent)
        latent = ttnn.permute(latent, (0, 3, 1, 2))

        if self.flatten:
            latent = ttnn.to_layout(latent, layout=ttnn.ROW_MAJOR_LAYOUT)
            latent = ttnn.reshape(
                latent, (latent.shape[0], latent.shape[1], latent.shape[2] * latent.shape[3])
            )  # latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
            latent = ttnn.to_layout(latent, layout=ttnn.TILE_LAYOUT)
            latent = ttnn.permute(latent, (0, 2, 1))
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(latent.dtype)

        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            # This is not invoked in our call
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed

        return latent + pos_embed
