# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native, tensor-parallel TTNN port of `Flux2SwiGLU`
(`transformer_blocks.0.ff.act_fn`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

    x1, x2 = x.chunk(2, dim=-1)
    x      = silu(x1) * x2

A gated activation with NO trainable parameters: the "linear projection layer is
fused into the first linear layer of the FF sub-block", as the reference's own
docstring puts it, so this module is pure elementwise work on 24576 features
producing 12288.

TENSOR-PARALLEL SCHEME (TP=8, validated by gathered PCC)
--------------------------------------------------------
There is no weight to shard -- the general principle "split large matmul
weights, NOT elementwise ops" means nothing here is a candidate for a
column/row split. What CAN be split is the feature axis of the work itself, and
that is exactly what this module does in situ: inside `Flux2FeedForward` it
consumes the output of the column-parallel `linear_in`, so each chip already
holds its own 1536-wide slice of both halves and the activation runs locally
with NO collective at all (see `_stubs/flux2_feed_forward.py`).

Exercised standalone, its input arrives replicated, so this port reproduces that
placement explicitly: `mesh_partition` each half across the mesh, apply the gate
locally, `all_gather` the result. The critical detail is that the two halves are
partitioned SEPARATELY. Partitioning the packed 24576-wide tensor in one go
would give chips 0-3 nothing but `x1` and chips 4-7 nothing but `x2`, and the
per-element `silu(x1) * x2` would pair features that live on different chips --
the same hazard the fused `linear_in` weight is regrouped to avoid.

The forward is pure ttnn: no torch math, no device->host readback.
"""

from __future__ import annotations

import ttnn
from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import (
    all_gather,
    as_rank4,
    mesh_partition,
    num_devices,
    restore_rank,
    split_last,
    to_device,
)


class TtFlux2SwiGLU:
    def __init__(self, device, torch_module=None) -> None:
        self.device = device
        self.tp = num_devices(device)

    @classmethod
    def build(cls, device, torch_module=None):
        return cls(device, torch_module)

    def __call__(self, x, **kwargs):
        x4, shape = as_rank4(to_device(self.device, x))
        half = int(x4.shape[-1]) // 2
        x1, x2 = split_last(x4, [half, half])

        if self.tp > 1:
            # Partition each half on its own: a single partition of the packed
            # tensor would split it at the chunk boundary, not across it.
            local1 = mesh_partition(self.device, x1, 3)
            local2 = mesh_partition(self.device, x2, 3)
            ttnn.deallocate(x1)
            ttnn.deallocate(x2)
        else:
            local1, local2 = x1, x2

        gate = ttnn.silu(local1)
        local = ttnn.mul(gate, local2)
        ttnn.deallocate(gate)
        ttnn.deallocate(local1)
        ttnn.deallocate(local2)

        out = all_gather(self.device, local, 3) if self.tp > 1 else local
        if out is not local:
            ttnn.deallocate(local)
        return restore_rank(out, shape, half)


def build(device, torch_module=None):
    return TtFlux2SwiGLU.build(device, torch_module)


def flux2_swi_g_l_u(device, torch_module=None):
    return TtFlux2SwiGLU.build(device, torch_module)
