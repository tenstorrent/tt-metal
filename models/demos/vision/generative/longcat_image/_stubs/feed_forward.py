# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `feed_forward` (diffusers ``FeedForward``) for the
meituan-longcat/LongCat-Image transformer block.

``FeedForward.forward(hidden_states)`` with ``net = [GELU, Dropout, Linear]``::

    hidden_states = net[0].proj(hidden_states)          # Linear dim -> inner
    hidden_states = gelu(hidden_states, approximate="tanh")
    # net[1] Dropout — identity in eval
    hidden_states = net[2](hidden_states)               # Linear inner -> dim_out
    return hidden_states

Runs fully fp32 (fp32 activations + fp32 output, HiFi4 / fp32-accumulate). The
tanh GELU is matched exactly by ttnn.GeluVariant.Tanh. Input/output are the
transformer's (B, N, C) sequence.
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
TILE = ttnn.TILE_LAYOUT


class _FeedForward:
    def __init__(self, device, ff):
        self.device = device
        self.ff = ff.eval() if hasattr(ff, "eval") else ff
        # GELU approximate mode ("tanh" for this model) -> matching ttnn variant.
        approx = getattr(self.ff.net[0], "approximate", "none")
        self.gelu_variant = ttnn.GeluVariant.Tanh if approx == "tanh" else ttnn.GeluVariant.Accurate
        self._lin = {}
        self._compute = None

    def _ck(self):
        if self._compute is None:
            self._compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
            )
        return self._compute

    def _linear(self, x, tm):
        key = id(tm)
        if key not in self._lin:
            b = tm.bias
            self._lin[key] = (
                ttnn.from_torch(
                    tm.weight.detach().to(torch.float32), dtype=F32, layout=TILE, device=self.device, memory_config=DRAM
                ),
                ttnn.from_torch(
                    b.detach().reshape(1, 1, -1).to(torch.float32),
                    dtype=F32,
                    layout=TILE,
                    device=self.device,
                    memory_config=DRAM,
                )
                if b is not None
                else None,
            )
        wt, bt = self._lin[key]
        return ttnn.linear(
            x, wt, bias=bt, transpose_b=True, memory_config=DRAM, dtype=F32, compute_kernel_config=self._ck()
        )

    def __call__(self, hidden_states, *args, **_ignored):
        x = hidden_states
        if isinstance(x, ttnn.Tensor):
            if x.layout != TILE:
                x = ttnn.to_layout(x, TILE)
            if x.dtype != F32:
                x = ttnn.typecast(x, F32)
        else:
            x = ttnn.from_torch(x.to(torch.float32), dtype=F32, layout=TILE, device=self.device, memory_config=DRAM)

        net = self.ff.net
        x = self._linear(x, net[0].proj)  # dim -> inner
        x = ttnn.gelu(x, variant=self.gelu_variant)  # tanh gelu, fp32
        for mod in net[1:]:  # skip Dropout, apply trailing Linear(s)
            if isinstance(mod, torch.nn.Linear):
                x = self._linear(x, mod)
        return x


def build(device, torch_module):
    """PCC-harness entry point: native TTNN FeedForward from the torch module."""
    return _FeedForward(device, torch_module)
