# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import math
import torch


class ttnn_SinePositionalEncoding3D:
    def __init__(
        self, num_feats, temperature=10000, normalize=False, scale=2 * math.pi, eps=1e-6, offset=0.0, init_cfg=None
    ):
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set," "scale should be provided and in float or int type, " f"found {type(scale)}"
            )
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def __call__(self, mask):
        # In this class many issue are there so, we are not trying to complete this model as there are many convert operation happening.
        not_mask = ttnn.ones(mask.shape, layout=ttnn.TILE_LAYOUT, device=mask.device()) - mask
        not_mask = ttnn.to_torch(not_mask)
        n_embed = not_mask.cumsum(1, dtype=torch.float32)  # issue 15293
        y_embed = not_mask.cumsum(2, dtype=torch.float32)  # issue 15293
        x_embed = not_mask.cumsum(3, dtype=torch.float32)  # issue 15293
        if self.normalize:
            n_embed = (n_embed + self.offset) / (n_embed[:, -1:, :, :] + self.eps) * self.scale
            y_embed = (y_embed + self.offset) / (y_embed[:, :, -1:, :] + self.eps) * self.scale  # issue 15216
            x_embed = (x_embed + self.offset) / (x_embed[:, :, :, -1:] + self.eps) * self.scale  # issue 15216

        dim_t = ttnn.arange(end=self.num_feats, dtype=ttnn.bfloat16, device=mask.device())
        dim_t = ttnn.to_layout(dim_t, layout=ttnn.TILE_LAYOUT)
        dim_t = ttnn.reshape(dim_t, (1, -1))  # This is becuase ttnn.arange creates 4d tensor

        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats) Replaced this with the following steps
        dim_t = ttnn.div(dim_t, 2)
        dim_t = ttnn.floor(dim_t)
        dim_t = 2 * ttnn.div(dim_t, self.num_feats)
        dim_t = self.temperature ** ttnn.to_torch(dim_t)  # issue 15212
        # dim_t=ttnn.from_torch(dim_t,layout=ttnn.TILE_LAYOUT,device=mask.device)
        # return dim_t

        # n_embed=ttnn.from_torch(n_embed,layout=ttnn.TILE_LAYOUT,device=mask.device())
        # y_embed=ttnn.from_torch(y_embed,layout=ttnn.TILE_LAYOUT,device=mask.device())
        # x_embed=ttnn.from_torch(x_embed,layout=ttnn.TILE_LAYOUT,device=mask.device())

        pos_n = n_embed[:, :, :, :, None] / dim_t  # issue 15216
        pos_x = x_embed[:, :, :, :, None] / dim_t  # issue 15216
        pos_y = y_embed[:, :, :, :, None] / dim_t  # issue 15216

        # pos_n=ttnn.from_torch(pos_n,layout=ttnn.TILE_LAYOUT,dtype=ttnn.bfloat16,device=mask.device())

        B, N, H, W = mask.shape

        # will use concat instead of stack.
        pos_n = torch.stack((pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()), dim=4).view(B, N, H, W, -1)
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=4).view(B, N, H, W, -1)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=4).view(B, N, H, W, -1)
        pos = torch.cat((pos_n, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)
        pos = ttnn.from_torch(pos, device=mask.device())
        return pos
