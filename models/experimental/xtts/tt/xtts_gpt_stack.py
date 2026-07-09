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
        """Run the full stack. ``x`` is ``[batch, seq, hidden]`` on device.

        Each block frees its own input internally; here we free the final block
        output once ``ln_f`` has consumed it."""
        for block in self.blocks:
            x = block(x)
        y = ttnn.layer_norm(x, weight=self.ln_f_weight, bias=self.ln_f_bias, epsilon=LAYER_NORM_EPS)
        ttnn.deallocate(x)
        return y

    def forward_prefill(self, x):
        """Prefill the prompt. Returns ``(ln_f(hidden), kv)`` where ``kv`` is the
        per-layer ``[(k, v), ...]`` list that seeds the decode KV cache."""
        kv = []
        for block in self.blocks:
            x, k, v = block.forward_prefill(x)
            kv.append((k, v))
        y = ttnn.layer_norm(x, weight=self.ln_f_weight, bias=self.ln_f_bias, epsilon=LAYER_NORM_EPS)
        ttnn.deallocate(x)
        return y, kv

    def forward_decode(self, x, kv):
        """Decode one token. ``kv`` is the per-layer cache list; returns the
        ``ln_f`` hidden state and the grown cache."""
        new_kv = []
        for block, (k, v) in zip(self.blocks, kv):
            x, k, v = block.forward_decode(x, k, v)
            new_kv.append((k, v))
        y = ttnn.layer_norm(x, weight=self.ln_f_weight, bias=self.ln_f_bias, epsilon=LAYER_NORM_EPS)
        ttnn.deallocate(x)
        return y, new_kv
