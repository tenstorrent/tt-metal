# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Top-level Qwen3.6-27B text model.

    token_ids -> embedding -> [block x 16] -> norm -> lm_head -> logits

The 64 decoder layers are grouped into 16 identical blocks of 4 layers each
(3 Gated DeltaNet + 1 Gated Attention). See tt_block.py.

SKELETON: forwards are pass-throughs. Shapes in the docstrings are the contract
each body has to end up satisfying.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.qwen_3_27b.tt.tt_block import TtQwen36Block
from models.experimental.qwen_3_27b.tt.tt_embedding import TtQwen36Embedding
from models.experimental.qwen_3_27b.tt.tt_lm_head import TtQwen36LMHead
from models.experimental.qwen_3_27b.tt.tt_rms_norm import TtQwen36RmsNorm

NUM_BLOCKS = 16  # 64 layers / 4 layers per block


class TtQwen36Model(LightweightModule):
    """
    Qwen3.6-27B decoder stack.

    Dimensions (hardcoded for now; these move into a config later):
        D = 5120        hidden size
        V = 248320      vocab size
        64 layers       = 16 blocks x (3 DeltaNet + 1 Gated Attention)

    Shapes:
        token_ids   [B, T]        uint32
        hidden      [B, T, D]     bfloat16   <- canonical activation format
        logits      [B, T, V]     bfloat16

    To implement:
        1. embedding lookup
        2. run the blocks in order, threading the hidden state
        3. final norm
        4. lm_head
    """

    def __init__(self, device):
        self.device = device

        self.embedding = TtQwen36Embedding(device)
        self.blocks = [TtQwen36Block(device, block_idx) for block_idx in range(NUM_BLOCKS)]
        self.norm = TtQwen36RmsNorm(device)
        self.lm_head = TtQwen36LMHead(device)

    def forward(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        token_ids [B, T] -> logits [B, T, V].

        NOTE for later: prefill only needs the LAST position's logits, and both
        the norm and the lm_head are position-wise -- so slicing x to [B, 1, D]
        right after the blocks is bit-identical and much cheaper (V = 248320, so
        full-sequence logits at T=4096 are ~2 GB in bf16). The blocks themselves
        always need the full sequence: attention mixes positions.
        """
        x = self.embedding(token_ids)  # [B, T, D]

        for block in self.blocks:
            x = block(x)  # [B, T, D]

        x = self.norm(x)  # [B, T, D]
        logits = self.lm_head(x)  # [B, T, V]

        return logits
