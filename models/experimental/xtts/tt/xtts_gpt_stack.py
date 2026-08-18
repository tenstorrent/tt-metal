# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

from models.experimental.xtts.reference.xtts_gpt_block import NUM_LAYERS
from models.experimental.xtts.tt.xtts_gpt_block import (
    NEG_INF,
    TtXttsGptBlock,
    _to_device,
    sharded_decode_ln,
    sharded_prefill_ln,
)


class TtXttsGptStack(LightweightModule):
    def __init__(self, state_dict, device, num_layers=NUM_LAYERS, max_seq=0):
        """Build the GPT block stack and final layer-norm weights."""
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.blocks = [TtXttsGptBlock(state_dict, device, layer_idx=i) for i in range(num_layers)]
        self.ln_f_weight = _to_device(state_dict["gpt.gpt.ln_f.weight"], device)
        self.ln_f_bias = _to_device(state_dict["gpt.gpt.ln_f.bias"], device)
        self.max_seq = 0
        if max_seq:
            self.init_static(max_seq)

    def init_static(self, max_seq):
        """Allocate static arange and key-pad buffers for decode masking."""
        self.max_seq = max_seq
        self.arange = ttnn.from_torch(
            torch.arange(max_seq, dtype=torch.float32).reshape(1, 1, 1, max_seq),
            device=self.device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )
        # Zero = nothing hidden, so decode is unchanged until set_key_pad says otherwise.
        self.key_pad = ttnn.zeros([1, 1, 1, max_seq], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def set_key_pad(self, lo, hi):
        # Written in place: the decode trace bound this address at capture. Derived on device
        # from self.arange, so no host tensor enters the per-chunk path.
        """Hide cache positions [lo, hi) from decode attention."""
        ge = ttnn.typecast(ttnn.ge(self.arange, float(lo)), ttnn.bfloat16)
        lt = ttnn.typecast(ttnn.lt(self.arange, float(hi)), ttnn.bfloat16)
        band = ttnn.multiply(ge, lt)  # 1.0 inside [lo, hi), else 0.0 — empty when lo == hi
        ttnn.multiply(band, NEG_INF, output_tensor=self.key_pad)
        for t in (ge, lt, band):
            ttnn.deallocate(t)

    def forward_decode(self, x, kv, pos, write_idx=None):
        # add_mask must stay DRAM (SDPA asserts DRAM mask).
        """Run one decode step through all blocks and final LN."""
        gt = ttnn.typecast(ttnn.gt(self.arange, pos), ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        causal = ttnn.multiply(gt, NEG_INF, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        add_mask = ttnn.add(causal, self.key_pad, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(causal)
        onehot = None
        if write_idx is None:
            onehot_row = ttnn.typecast(ttnn.eq(self.arange, pos), ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            onehot = ttnn.reshape(onehot_row, (1, 1, self.max_seq, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        for block, (k, v) in zip(self.blocks, kv):
            x = block.forward_decode(x, k, v, onehot, add_mask, write_idx)
        return sharded_decode_ln(x, self.ln_f_weight, self.ln_f_bias, self.device)

    def forward(self, x):
        """Run prefill through all blocks without returning KV."""
        for block in self.blocks:
            x, _, _ = block.forward_prefill(x)
        return sharded_prefill_ln(x, self.ln_f_weight, self.ln_f_bias, self.device)

    def forward_prefill(self, x):
        """Run prefill through all blocks and collect KV caches."""
        kv = []
        for block in self.blocks:
            x, k, v = block.forward_prefill(x)
            kv.append((k, v))
        y = sharded_prefill_ln(x, self.ln_f_weight, self.ln_f_bias, self.device)
        return y, kv
