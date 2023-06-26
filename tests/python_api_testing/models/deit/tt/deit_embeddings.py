from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from typing import Optional, Set, Tuple, Union

import torch
from torch import nn
from deit_config import DeiTConfig

import tt_lib
from tt_lib.fallback_ops import fallback_ops

from deit_patch_embeddings import DeiTPatchEmbeddings

class DeiTEmbeddings(nn.Module):

    def __init__(self, config: DeiTConfig(), base_address: str, state_dict: dict, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(state_dict[f"{base_address}.cls_token"])
        self.distillation_token = nn.Parameter(state_dict[f"{base_address}.distillation_token"])
        self.mask_token = nn.Parameter(state_dict[f"{base_address}.mask_token"]) if use_mask_token else None

        self.patch_embeddings = DeiTPatchEmbeddings(config,
                                                    state_dict=state_dict,
                                                    base_address=f"{base_address}.patch_embeddings")

        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(state_dict[f"{base_address}.position_embeddings"])

    def forward(self,
                pixel_values: torch.Tensor,
                bool_masked_pos: Optional[torch.BoolTensor] = None) -> torch.Tensor:

        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_length, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings

        return embeddings
