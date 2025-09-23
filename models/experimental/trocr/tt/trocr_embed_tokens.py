# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm
from models.experimental.trocr.tt.trocr_configuration import TtTrOCRConfig
import ttnn


class TtTrOCREmbedTokens(nn.Module):
    def __init__(
        self,
        config: TtTrOCRConfig,
        device=None,
        state_dict=None,
        base_address="",
    ):
        super().__init__()
        self.device = device
        weights = state_dict[f"{base_address}.weight"]
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=1, _weight=weights)

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        input_ids = tt_to_torch_tensor(input_ids).squeeze(0)
        position = self.embed_tokens(input_ids.long())
        position = torch_to_tt_tensor_rm(position, self.device, put_on_device=False)
        return position
