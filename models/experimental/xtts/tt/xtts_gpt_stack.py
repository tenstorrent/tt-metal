# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the full XTTS-v2 GPT decoder stack.

Mirrors ``reference/xtts_gpt_stack.py``: 30 identical GPT-2 decoder blocks
(each a :class:`~models.experimental.xtts.tt.xtts_gpt_block.TtXttsGptBlock`)
followed by a final LayerNorm (``ln_f``). Output is the decoder's final
layer-normed hidden states, before the text/mel heads.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

from models.experimental.xtts.reference.xtts_gpt_block import LAYER_NORM_EPS, NUM_LAYERS
from models.experimental.xtts.tt.xtts_gpt_block import TtXttsGptBlock, _to_device


class TtXttsGptStack(LightweightModule):
    def __init__(self, state_dict, device, num_layers=NUM_LAYERS):
        super().__init__()
        self.device = device
        self.num_layers = num_layers

        # One TtXttsGptBlock per repeating layer, each loading its own weights.
        self.blocks = [TtXttsGptBlock(state_dict, device, layer_idx=i) for i in range(num_layers)]

        # Final LayerNorm (ln_f) applied after the last block.
        self.ln_f_weight = _to_device(state_dict["gpt.gpt.ln_f.weight"], device)
        self.ln_f_bias = _to_device(state_dict["gpt.gpt.ln_f.bias"], device)

    def forward(self, x):
        """Run the full stack. ``x`` is ``[batch, seq, hidden]`` on device."""
        for block in self.blocks:
            x = block(x)
        return ttnn.layer_norm(x, weight=self.ln_f_weight, bias=self.ln_f_bias, epsilon=LAYER_NORM_EPS)
