# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the full XTTS-v2 GPT decoder stack.

Mirrors ``reference/xtts_gpt_stack.py``: 30 identical GPT-2 decoder blocks
(each a :class:`~models.experimental.xtts.tt.xtts_gpt_block.TtXttsGptBlock`)
followed by a final LayerNorm (``ln_f``). Output is the decoder's final
layer-normed hidden states, before the text/mel heads.

KV caching mirrors the per-block API:
    * ``forward``          — stateless full-sequence pass (unchanged).
    * ``forward_prefill``  — full prompt pass that seeds every block's KV cache.
    * ``forward_decode``   — one new token, attending over the cached history.
An autoregressive caller runs ``forward_prefill`` once on the prompt, then
``forward_decode`` per generated token. Feeding the whole sequence through
prefill + per-token decode reproduces the stateless ``forward`` output — this is
what the stack PCC test checks.
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

    def _ln_f(self, x):
        return ttnn.layer_norm(x, weight=self.ln_f_weight, bias=self.ln_f_bias, epsilon=LAYER_NORM_EPS)

    def forward(self, x):
        """Run the full stack, stateless. ``x`` is ``[batch, seq, hidden]`` on device."""
        for block in self.blocks:
            x = block(x)
        return self._ln_f(x)

    def forward_prefill(self, x):
        """Run the full prompt through every block, seeding each block's KV cache.

        ``x`` is ``[batch, seq, hidden]``. Same output as ``forward``; call once
        before decoding.
        """
        for block in self.blocks:
            x = block.forward_prefill(x)
        return self._ln_f(x)

    def forward_decode(self, x):
        """Run one new token (``x`` is ``[batch, 1, hidden]``) through the cached stack.

        Requires ``forward_prefill`` first. Returns ``[batch, 1, hidden]``.
        """
        for block in self.blocks:
            x = block.forward_decode(x)
        return self._ln_f(x)
