# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native, tensor-parallel TTNN port of `Flux2FeedForward`
(`transformer_blocks.0.ff`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

    x = linear_in(x)          # 4096 -> 24576
    x = Flux2SwiGLU(x)        # silu(x[:12288]) * x[12288:]  -> 12288
    x = linear_out(x)         # 12288 -> 4096

TENSOR-PARALLEL SCHEME (TP=8, validated by gathered PCC)
--------------------------------------------------------
Textbook column-then-row, with one twist this model forces.

* `linear_in` is COLUMN-parallel: its output feeds a per-element gate, so no
  collective is needed after it. But the naive contiguous split is WRONG here:
  `linear_in` is FUSED -- its 24576 columns are SwiGLU's two 12288-wide halves
  concatenated, and `silu(x1) * x2` pairs column `i` of the first half with
  column `i` of the second. A contiguous 8-way split would put all of `x1` on
  chips 0-3 and all of `x2` on chips 4-7, so the gate would multiply features
  that live on different chips. The weight columns are therefore REORDERED at
  load time (`_regroup` in `_flux2_ttnn.py`) into
  `[x1_chip0, x2_chip0, x1_chip1, x2_chip1, ...]`, after which a plain
  `ShardTensorToMesh(dim=-1)` hands each chip 1536 matching columns of BOTH
  halves and the local SwiGLU is exactly the global one restricted to those
  features.
* `linear_out` is ROW-parallel: it reduces the 12288-wide activation back to the
  model dim, so its INPUT features are split (`ShardTensorToMesh(dim=2)`,
  12288/8 = 1536 rows -- the same 1536 features each chip just produced, since
  the `x1` blocks are contiguous in the original order). Each chip then holds a
  partial sum over the full output and one `all_reduce` (SUM) completes it.

Math is unchanged: the same products, summed in a different order. The
implementation lives in `_flux2_ttnn.py`, shared with the blocks that contain
this layer. The forward is pure ttnn: no torch math, no device->host readback.
"""

from __future__ import annotations

from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import (
    TtFlux2FeedForward,
    as_rank4,
    restore_rank,
    to_device,
)


class TtFlux2FeedForwardStub:
    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("flux2_feed_forward needs the torch reference module for weights")
        self.device = device
        self.ff = TtFlux2FeedForward(device, torch_module)

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, x, **kwargs):
        x4, shape = as_rank4(to_device(self.device, x))
        out = self.ff(x4)
        return restore_rank(out, shape, self.ff.linear_out.out_features)


def build(device, torch_module=None):
    return TtFlux2FeedForwardStub.build(device, torch_module)


def flux2_feed_forward(device, torch_module=None):
    return TtFlux2FeedForwardStub.build(device, torch_module)
