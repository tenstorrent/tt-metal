# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Language-model head -- hidden state to vocabulary logits.

One matmul, but the widest one in the model, and the reason prefill slices down
to a single token before calling it.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtQwen36LMHead(LightweightModule):
    """
    Output projection to vocab logits.

    Dimensions:
        D = 5120      hidden size
        V = 248320    vocab size

    Weight:
        lm_head  [248320, 5120]   untied from the embedding table

    Shapes:
        x       [B, T, D]
        logits  [B, T, V]

    Watch the output size: at T = 4096 the logits are 4096 * 248320 * 2 bytes
    = ~2 GB. Prefill only ever needs the LAST token's logits, so the caller
    should slice x to [B, 1, D] first. Decode passes T = 1 anyway.

    To implement:
        1. load the weight transposed to [5120, 248320]
        2. ttnn.linear(x, weight)
    """

    def __init__(self, device):
        self.device = device

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[B, T, D] -> [B, T, V]."""
        return x
