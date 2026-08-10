# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttml

from ttml.models.llama.transformer import LlamaBlock
from ttml.models.llama import Llama


class LlamaBlockCompositeKV(LlamaBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Composite SDPA accepts non-broadcast masks like (B, 1, S, S).
        self.attention.sdpa = ttml.ops.attention.scaled_dot_product_attention_composite


class LlamaCompositeKV(Llama):
    def __init__(self, config):
        super().__init__(config)
        self.create_name("Llama")

        # Composite SDPA accepts non-broadcast masks like (B, 1, S, S).
        for block in self.blocks:
            block.attention.sdpa = ttml.ops.attention.scaled_dot_product_attention_composite
