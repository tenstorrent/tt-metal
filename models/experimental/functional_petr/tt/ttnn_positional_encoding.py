# # SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# # SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import math


class ttnn_SinePositionalEncoding3D:
    def __init__(
        self, num_feats, temperature=10000, normalize=False, scale=2 * math.pi, eps=1e-6, offset=0.0, init_cfg=None
    ):
        if normalize:
            assert isinstance(scale, (float, int))
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset
        self.torch_fallback = True

        # Pre-compute dim_t
        self.dim_t_torch = torch.arange(num_feats, dtype=torch.bfloat16)
        self.dim_t_torch = temperature ** (2 * (self.dim_t_torch // 2) / num_feats)

    def __call__(self, mask):
        device = mask.device()
        B, N, H, W = mask.shape

        # TTNN operations
        ones = ttnn.ones_like(mask)
        not_mask = ttnn.subtract(ones, mask)

        n_embed = ttnn.cumsum(not_mask, dim=1)
        y_embed = ttnn.cumsum(not_mask, dim=2)
        x_embed = ttnn.cumsum(not_mask, dim=3)

        if self.torch_fallback:
            # Convert to torch for remaining operations
            n_embed = ttnn.to_torch(n_embed).to(torch.bfloat16)
            y_embed = ttnn.to_torch(y_embed).to(torch.bfloat16)
            x_embed = ttnn.to_torch(x_embed).to(torch.bfloat16)

            if self.normalize:
                n_embed = (n_embed + self.offset) / (n_embed[:, -1:, :, :] + self.eps) * self.scale
                y_embed = (y_embed + self.offset) / (y_embed[:, :, -1:, :] + self.eps) * self.scale
                x_embed = (x_embed + self.offset) / (x_embed[:, :, :, -1:] + self.eps) * self.scale

            # Move dim_t to same device
            dim_t = self.dim_t_torch.to(n_embed.device)

            # Positional encoding computation
            pos_n = n_embed[:, :, :, :, None] / dim_t
            pos_x = x_embed[:, :, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, :, None] / dim_t

            # Sin/cos encoding
            pos_n = torch.stack((pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()), dim=4).view(
                B, N, H, W, -1
            )

            pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=4).view(
                B, N, H, W, -1
            )

            pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=4).view(
                B, N, H, W, -1
            )

            pos = torch.cat((pos_n, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)

            # Convert back to ttnn
            pos = ttnn.from_torch(pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            return pos

        else:
            # Normalization (if needed)
            if self.normalize:
                # For n_embed: normalize by last value in dim 1
                n_embed = n_embed + self.offset
                norm_factor = n_embed[:, -1:, :, :] + self.eps
                n_embed = ttnn.div(n_embed, norm_factor)
                n_embed = n_embed * self.scale

                # For y_embed: normalize by last value in dim 2
                y_embed = y_embed + self.offset
                norm_factor = y_embed[:, :, -1:, :] + self.eps
                y_embed = ttnn.div(y_embed, norm_factor)
                y_embed = y_embed * self.scale

                # For x_embed: normalize by last value in dim 3
                x_embed = x_embed + self.offset
                norm_factor = x_embed[:, :, :, -1:] + self.eps
                x_embed = ttnn.div(x_embed, norm_factor)
                x_embed = x_embed * self.scale

            dim_t = ttnn.arange(end=self.num_feats, dtype=ttnn.bfloat16, device=mask.device())
            dim_t = ttnn.to_layout(dim_t, layout=ttnn.TILE_LAYOUT)
            dim_t = ttnn.reshape(dim_t, (1, -1))  # This is becuase ttnn.arange creates 4d tensor

            dim_t = ttnn.div(dim_t, 2)
            dim_t = ttnn.floor(dim_t)
            dim_t = 2 * ttnn.div(dim_t, self.num_feats)

            dim_t = ttnn.pow(float(self.temperature), dim_t)

            # Add dimension for broadcasting: [B, N, H, W] -> [B, N, H, W, 1]
            n_embed = ttnn.unsqueeze(n_embed, dim=-1)
            y_embed = ttnn.unsqueeze(y_embed, dim=-1)
            x_embed = ttnn.unsqueeze(x_embed, dim=-1)

            # Divide by dim_t (broadcasting)
            pos_n = ttnn.div(n_embed, dim_t)
            pos_y = ttnn.div(y_embed, dim_t)
            pos_x = ttnn.div(x_embed, dim_t)

            # Apply sin/cos to alternating dimensions

            # For pos_n
            sin_part_n = ttnn.sin(pos_n[:, :, :, :, 0::2])
            cos_part_n = ttnn.cos(pos_n[:, :, :, :, 1::2])
            stacked_n = ttnn.stack((sin_part_n, cos_part_n), dim=4)
            pos_n = ttnn.reshape(stacked_n, (B, N, H, W, -1))

            # For pos_y
            sin_part_y = ttnn.sin(pos_y[:, :, :, :, 0::2])
            cos_part_y = ttnn.cos(pos_y[:, :, :, :, 1::2])
            stacked_y = ttnn.stack((sin_part_y, cos_part_y), dim=4)
            pos_y = ttnn.reshape(stacked_y, (B, N, H, W, -1))

            # For pos_x
            sin_part_x = ttnn.sin(pos_x[:, :, :, :, 0::2])
            cos_part_x = ttnn.cos(pos_x[:, :, :, :, 1::2])
            stacked_x = ttnn.stack((sin_part_x, cos_part_x), dim=4)
            pos_x = ttnn.reshape(stacked_x, (B, N, H, W, -1))

            # Concatenate all embeddings
            pos = ttnn.concat([pos_n, pos_y, pos_x], dim=4)

            # Permute to [B, N, num_feats*3, H, W]
            pos = ttnn.permute(pos, (0, 1, 4, 2, 3))

            return pos
