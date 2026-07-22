# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the full XTTS-v2 GPT decoder stack.

Mirrors ``reference/xtts_gpt_stack.py``: 30 identical GPT-2 decoder blocks
(each a :class:`~models.experimental.xtts.tt.xtts_gpt_block.TtXttsGptBlock`)
followed by a final LayerNorm (``ln_f``). Output is the decoder's final
layer-normed hidden states, before the text/mel heads.
"""

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

from models.experimental.xtts.reference.xtts_gpt_block import LAYER_NORM_EPS, NUM_LAYERS
from models.experimental.xtts.tt.xtts_gpt_block import NEG_INF, TtXttsGptBlock, _to_device


class TtXttsGptStack(LightweightModule):
    def __init__(self, state_dict, device, num_layers=NUM_LAYERS, max_seq=0):
        super().__init__()
        self.device = device
        self.num_layers = num_layers

        # One TtXttsGptBlock per repeating layer, each loading its own weights.
        self.blocks = [TtXttsGptBlock(state_dict, device, layer_idx=i) for i in range(num_layers)]

        # Final LayerNorm (ln_f) applied after the last block.
        self.ln_f_weight = _to_device(state_dict["gpt.gpt.ln_f.weight"], device)
        self.ln_f_bias = _to_device(state_dict["gpt.gpt.ln_f.bias"], device)

        # Static-KV decode support (trace-compatible; see forward_decode_static). Sized only
        # when max_seq is given so the concat-based path is unaffected.
        self.max_seq = 0
        if max_seq:
            self.init_static(max_seq)

    def init_static(self, max_seq):
        """Build the persistent column-index arange used by forward_decode_static (idempotent)."""
        self.max_seq = max_seq
        self.arange = ttnn.from_torch(
            torch.arange(max_seq, dtype=torch.float32).reshape(1, 1, 1, max_seq),
            device=self.device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )

    def forward_decode(self, x, kv, pos):
        """DECODE — one-token decode over FIXED-size caches (the stack's decode forward).

        ``kv`` is the per-layer ``[(k_cache, v_cache), ...]`` list of ``[1, heads, max_seq,
        head_dim]`` caches; ``pos`` is a ``[1, 1, 1, max_seq]`` tensor filled with the current
        absolute cache position. Builds the one-hot cache-write mask and the additive
        attention mask ONCE (shared across layers) from ``pos`` + the persistent arange, then
        runs every block. Returns the ``ln_f`` hidden (caches updated in place); all shapes static."""
        onehot_row = ttnn.typecast(ttnn.eq(self.arange, pos), ttnn.bfloat16)  # [1,1,1,MAX] 1 at col=pos
        onehot = ttnn.reshape(onehot_row, (1, 1, self.max_seq, 1))  # [1,1,MAX,1] cache-write selector
        keep = ttnn.add(ttnn.multiply(onehot, -1.0), 1.0)  # 1 - onehot
        le = ttnn.typecast(ttnn.le(self.arange, pos), ttnn.bfloat16)  # [1,1,1,MAX] 1 for cached positions
        add_mask = ttnn.multiply(
            ttnn.add(ttnn.multiply(le, -1.0), 1.0), NEG_INF
        )  # (1-le)*(-1e30): 0 cached, -inf ahead
        for block, (k, v) in zip(self.blocks, kv):
            x = block.forward_decode(x, k, v, onehot, keep, add_mask)  # k, v updated in place
        y = ttnn.layer_norm(x, weight=self.ln_f_weight, bias=self.ln_f_bias, epsilon=LAYER_NORM_EPS)
        ttnn.deallocate(x)
        return y

    def forward(self, x):
        """Full teacher-forced pass (no cache kept). Routes through each block's PREFILL
        forward and discards the returned K/V — the block has no separate full-forward."""
        for block in self.blocks:
            x, _, _ = block.forward_prefill(x)  # full causal block; drop the prompt K/V
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
