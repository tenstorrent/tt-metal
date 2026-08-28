# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native, tensor-parallel TTNN port of `Flux2Modulation`
(`double_stream_modulation_img`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

    mod = self.linear(self.act_fn(temb))     # SiLU, then 4096 -> 4096*3*2 = 24576

One projection of the timestep embedding into the six modulation vectors
(shift / scale / gate, for the attention and the feed-forward sub-block) that
every dual-stream block consumes. `Flux2Modulation.split` slices them apart
inside the blocks; this component produces the packed 24576-wide tensor.

TENSOR-PARALLEL SCHEME (TP=8, validated by gathered PCC)
--------------------------------------------------------
COLUMN-parallel + `all_gather`.

The weight is 4096 x 24576, so the OUTPUT axis is the large one and splitting it
gives each chip 3072 columns (96 whole tiles). That is also what the general
principle selects: this projection's output feeds purely per-element work (the
blocks multiply and add these vectors into their normalised activations), so
nothing forces a reduction.

Unlike the feed-forward's fused `linear_in`, the six modulation vectors are NOT
combined with each other by any per-element op -- each is sliced out and used on
its own -- so the columns need no regrouping: a plain contiguous split is
correct, and `all_gather(dim=-1)` puts them back in exactly the original order.

The gather is what the model wants anyway: every block needs all six vectors at
full width, replicated. A row-parallel alternative (split the 4096 inputs,
all_reduce) would move the same numbers but split the SMALLER axis and leave the
weight shards 8x wider.

The forward is pure ttnn: no torch math, no device->host readback. torch appears
only in `__init__`, to transpose the checkpoint weight into ttnn's `[in, out]`
layout before staging it on the mesh.
"""

from __future__ import annotations

import ttnn
from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import (
    TtLinear,
    all_gather,
    as_rank4,
    num_devices,
    restore_rank,
    to_device,
)


class TtFlux2Modulation:
    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("flux2_modulation needs the torch reference module for weights")
        self.device = device
        self.tp = num_devices(device)
        self.linear = TtLinear(device, torch_module.linear, scheme="column")
        self.out_features = self.linear.out_features

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, temb, **kwargs):
        x4, shape = as_rank4(to_device(self.device, temb))
        act = ttnn.silu(x4)
        local = self.linear(act)
        ttnn.deallocate(act)
        out = all_gather(self.device, local, 3) if self.tp > 1 else local
        if out is not local:
            ttnn.deallocate(local)
        return restore_rank(out, shape, self.out_features)


def build(device, torch_module=None):
    return TtFlux2Modulation.build(device, torch_module)


def flux2_modulation(device, torch_module=None):
    return TtFlux2Modulation.build(device, torch_module)
