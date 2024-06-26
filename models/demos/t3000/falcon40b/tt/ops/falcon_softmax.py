# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn

from typing import List
from models.utility_functions import torch2tt_tensor


class TtFalconSoftmax:
    def __init__(self, device, model_config, head_dim: int = 64, seqlen: int = 32, is_sharded=False):
        super().__init__()

        self.is_sharded = is_sharded
        self.model_config = model_config
        self.seqlen = seqlen
        self.scalar = 1 / math.sqrt(head_dim)

    def __call__(
        self, x: ttnn.experimental.tensor.Tensor, attention_mask: ttnn.experimental.tensor.Tensor
    ) -> ttnn.experimental.tensor.Tensor:
        softmax_progcfg = self.model_config["SOFTMAX_PROGCFG"]
        softmax_progcfg.block_w = (
            self.seqlen // 32
        )  # TODO: We can directly set this for prefill model config, but it might be confusing...

        if self.is_sharded:
            # Subtract max value from activation before softmax
            out = ttnn.scale_mask_softmax_in_place(
                x,
                self.scalar,
                attention_mask,
                program_config=softmax_progcfg,
                # output_mem_config=self.model_config["DEFAULT_MEMCFG"],
                # program_config=ttnn.SoftmaxDefaultProgramConfig(),
                is_causal_mask=True,
            )
        else:
            # Subtract max value from activation before softmax
            out = ttnn.scale_mask_softmax_in_place(
                x,
                self.scalar,
                attention_mask,
                # program_config=softmax_progcfg,
                # output_mem_config=self.model_config["DEFAULT_MEMCFG"],
                program_config=ttnn.SoftmaxDefaultProgramConfig(),
                is_causal_mask=True,
            )

        return out
