# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
import tt_lib as ttl

from typing import List
from models.utility_functions import torch2tt_tensor


class TtFalconSoftmax:
    def __init__(
        self,
        device,
        model_config,
        head_dim: int = 64,
        seqlen: int = 32,
    ):
        super().__init__()

        self.model_config = model_config
        self.seqlen = seqlen
        self.scalar = 1 / math.sqrt(head_dim)

    def __call__(self, x: ttl.tensor.Tensor, attention_mask: ttl.tensor.Tensor) -> ttl.tensor.Tensor:
        softmax_progcfg = self.model_config["SOFTMAX_PROGCFG"]
        softmax_progcfg.block_w = (
            self.seqlen // 32
        )  # TODO: We can directly set this for prefill model config, but it might be confusing...

        # Subtract max value from activation before softmax
        out = ttl.operations.primary.transformers.scale_mask_softmax_in_place(
            x,
            self.scalar,
            attention_mask,
            # program_config=softmax_progcfg,
            # output_mem_config=self.model_config["DEFAULT_MEMCFG"],
            program_config=ttl.operations.primary.transformers.SoftmaxDefaultProgramConfig(),
            is_causal_mask=True,
        )

        return out
