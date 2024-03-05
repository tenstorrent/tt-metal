# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib

from typing import List, Optional, Tuple, Union
from models.experimental.llama_old.tt.llama_model import TtLlamaShared

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import torch_to_tt_tensor_rm


class TtLlamaForCausalLM(TtLlamaShared):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        config,
        num_decoders,
    ):
        super().__init__(device, state_dict, base_url, max_position_embeddings, config, num_decoders)

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.state_dict = state_dict  # hugging_face_reference_model.state_dict()

        self.weight = torch_to_tt_tensor_rm(self.state_dict["lm_head.weight"], self.device)
        self.bias = None

        self.linear = TTLinear(
            self.weight.get_legacy_shape()[-1], self.weight.get_legacy_shape()[-2], self.weight, self.bias
        )

    def forward(self, x):
        encoder_output = super().forward(x)
        return self.linear(x)
