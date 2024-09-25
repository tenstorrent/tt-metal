# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

import ttnn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)


import ttnn
from models.experimental.swin.tt.swin_patch_embedding import TtSwinPatchEmbeddings
from tt_lib.fallback_ops import fallback_ops


class TtSwinEmbeddings(nn.Module):
    def __init__(self, config, state_dict, base_address, device, use_mask_token=False):
        super().__init__()
        self.config = config
        self.device = device
        self.patch_embeddings = TtSwinPatchEmbeddings(
            config=self.config,
            state_dict=state_dict,
            base_address=f"{base_address}.patch_embeddings",
            device=self.device,
        )
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None

        gamma = torch_to_tt_tensor_rm(state_dict[f"{base_address}.norm.weight"], self.device)
        beta = torch_to_tt_tensor_rm(state_dict[f"{base_address}.norm.bias"], self.device)
        self.norm = fallback_ops.LayerNorm(gamma, beta, normalized_shape=config.embed_dim, eps=config.layer_norm_eps)

    def const_tensor(self, shape, value):
        return ttnn.full(shape, value)

    def forward(
        self,
        pixel_values: Optional[ttnn.Tensor],
        bool_masked_pos: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        _, batch_size, seq_len, _ = embeddings.shape.with_tile_padding()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            bool_masked_pos = tt_to_torch_tensor(bool_masked_pos)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)

            mask_tokens = torch_to_tt_tensor_rm(mask_tokens, self.device)
            mask = torch_to_tt_tensor_rm(mask, self.device)

            mask_tokens = ttnn.mul(mask_tokens, mask)

            unit_tensor = self.const_tensor(mask.shape.with_tile_padding(), 1)
            mask = ttnn.sub(unit_tensor, mask)

            embeddings = ttnn.mul(embeddings, mask)
            embeddings = ttnn.add(embeddings, mask_tokens)

        if self.position_embeddings is not None:
            self.position_embeddings = torch_to_tt_tensor_rm(self.position_embeddings, self.device)
        if self.position_embeddings is not None:
            embeddings = ttnn.add(embeddings, self.position_embeddings)

        return embeddings, output_dimensions
