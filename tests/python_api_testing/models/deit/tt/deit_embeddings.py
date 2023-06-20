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
from deit_helper_funcs import make_linear
from tt_lib.fallback_ops import fallback_ops

from deit_patch_embeddings import TtDeiTPatchEmbeddings


class TtDeiTEmbeddings(nn.Module):
    """
    Construct the CLS token, distillation token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: DeiTConfig() , host, device, state_dict=None, base_address="", use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = fallback_ops.full([1, 1, 1, config.hidden_size], 0)
        self.distillation_token = fallback_ops.full([1, 1, 1, config.hidden_size], 0)
        self.patch_embeddings = TtDeiTPatchEmbeddings(config, host, device, state_dict, f"{base_address}.patch_embeddings.projection")
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = fallback_ops.full([1, 1, num_patches + 2, config.hidden_size], 0)

    def forward(self, pixel_values: tt_lib.tensor.Tensor, bool_masked_pos = None) -> tt_lib.tensor.Tensor:
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, _ , seq_length, _ = embeddings.shape()

        # if bool_masked_pos is not None:
        #     mask_tokens = fallback_ops.expand() self.mask_token.expand(batch_size, seq_length, -1)
        #     # replace the masked visual tokens by mask_tokens
        #     mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
        #     embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        cls_tokens = self.cls_token
        distillation_tokens = self.distillation_token
        embeddings = fallback_ops.concat([cls_tokens, distillation_tokens, embeddings], dim=2)
        embeddings = tt_lib.tensor.add(embeddings , self.position_embeddings)
        # embeddings = self.dropout(embeddings)
        return embeddings
