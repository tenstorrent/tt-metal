# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `Flux2PosEmbed` (`pos_embed`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

Builds the 4-axis rotary table. For each axis `i` of `axes_dim = (32, 32, 32,
32)` the reference calls `get_1d_rotary_pos_embed(dim_i, ids[..., i],
theta=2000, repeat_interleave_real=True)`, i.e.

    inv[f]        = theta ** (-2f / dim_i)          f = 0 .. dim_i/2 - 1
    freqs[s, f]   = ids[s, i] * inv[f]
    cos_i, sin_i  = cos/sin(freqs) each repeat_interleaved by 2   -> [S, dim_i]

and concatenates the four axes into `(cos, sin)` of `[S, 128]`.

This is a pure table build with no trainable parameters -- there is nothing to
shard, so it stays REPLICATED on every chip (the general TP principle "split
large matmul weights, not lookup tables"). `inv` repeat-interleaved is folded
into a constant per-axis frequency row at construction time, so the device work
is one broadcast multiply per axis plus one `cos` and one `sin`.

PRECISION: this runs in **float32**, not bfloat16. The phase `ids * inv` reaches
the position count in radians (~63 for a 1024px latent grid), and bfloat16's
~0.4% relative step is then ~0.25 rad of phase -- measured on device,
`ttnn.cos` of a bfloat16 argument is off by up to 0.9 absolute at that range,
versus 0.0 in float32. The outer product is done as `repeat` + `mul` rather than
a K=1 matmul for the same reason: the matmul path rounds through bfloat16 and
was measured at ~2e-3 relative error, the broadcast multiply at exactly 0.

The forward is pure ttnn: no torch math and no device->host readback. torch is
used only in `__init__` to build the constant frequency rows.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import replicate_mapper, to_device


class TtFlux2PosEmbed:
    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("flux2_pos_embed needs the torch reference module for its config")
        self.device = device
        self.theta = float(torch_module.theta)
        self.axes_dim = [int(d) for d in torch_module.axes_dim]
        self.total_dim = sum(self.axes_dim)

        self.inv_freqs = []
        for dim in self.axes_dim:
            # float64 here mirrors the reference's `freqs_dtype`; the row is then
            # stored as float32, which is what the device multiply consumes.
            inv = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
            row = inv.repeat_interleave(2).to(torch.float32).reshape(1, 1, 1, dim)
            self.inv_freqs.append(
                ttnn.from_torch(
                    row,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=replicate_mapper(device),
                )
            )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, ids, **kwargs):
        ids = to_device(self.device, ids)
        shape = list(ids.shape)
        seq = int(shape[-2])
        ids4 = ttnn.reshape(ids, [1, 1, seq, int(shape[-1])])
        if ids4.dtype != ttnn.float32:
            ids4 = ttnn.typecast(ids4, ttnn.float32)

        columns = []
        for axis, dim in enumerate(self.axes_dim):
            pos = ttnn.slice(ids4, [0, 0, 0, axis], [1, 1, seq, axis + 1])
            spread = ttnn.repeat(pos, [1, 1, 1, dim])
            columns.append(ttnn.mul(spread, self.inv_freqs[axis]))
            ttnn.deallocate(pos)
            ttnn.deallocate(spread)

        freqs = ttnn.concat(columns, dim=3) if len(columns) > 1 else columns[0]
        for column in columns:
            if column is not freqs:
                ttnn.deallocate(column)

        cos = ttnn.reshape(ttnn.cos(freqs), [seq, self.total_dim])
        sin = ttnn.reshape(ttnn.sin(freqs), [seq, self.total_dim])
        ttnn.deallocate(freqs)
        return cos, sin


def build(device, torch_module=None):
    return TtFlux2PosEmbed.build(device, torch_module)


def flux2_pos_embed(device, torch_module=None):
    return TtFlux2PosEmbed.build(device, torch_module)
