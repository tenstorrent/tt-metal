# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)


import tt_lib
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
        return tt_lib.tensor.full(shape, value)

    def forward(
        self,
        pixel_values: Optional[tt_lib.tensor.Tensor],
        bool_masked_pos: Optional[tt_lib.tensor.Tensor] = None,
    ) -> Tuple[tt_lib.tensor.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        _, batch_size, seq_len, _ = embeddings.get_legacy_shape()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            bool_masked_pos = tt_to_torch_tensor(bool_masked_pos)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)

            mask_tokens = torch_to_tt_tensor_rm(mask_tokens, self.device)
            mask = torch_to_tt_tensor_rm(mask, self.device)

            mask_tokens = tt_lib.tensor.mul(mask_tokens, mask)

            unit_tensor = self.const_tensor(mask.get_legacy_shape(), 1)
            mask = tt_lib.tensor.sub(unit_tensor, mask)

            embeddings = tt_lib.tensor.mul(embeddings, mask)
            embeddings = tt_lib.tensor.add(embeddings, mask_tokens)

        if self.position_embeddings is not None:
            self.position_embeddings = torch_to_tt_tensor_rm(self.position_embeddings, self.device)
        if self.position_embeddings is not None:
            embeddings = tt_lib.tensor.add(embeddings, self.position_embeddings)

        return embeddings, output_dimensions
