# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native, tensor-parallel TTNN port of `mlp` (`transformer_blocks.0.ff_context`)
of `black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

    x = linear_in(x)          # 4096 -> 24576
    x = Flux2SwiGLU(x)        # silu(x[:12288]) * x[12288:]  -> 12288
    x = linear_out(x)         # 12288 -> 4096

This is the TEXT stream's feed-forward of a dual-stream block -- the same
`Flux2FeedForward` class as `transformer_blocks.0.ff` (the `flux2_feed_forward`
component), instantiated on the other stream, so it shares that component's
already-validated implementation and TP scheme rather than duplicating it.

Why the canonical wrapper was replaced
--------------------------------------
This component was auto-seeded to delegate to `models/tt_transformers/tt/mlp.py`
via `ModelArgs(mesh_device=device, instruct=True)`. That path cannot work here,
for two independent reasons:

1. `ModelArgs` reads a `transformers` config and raised
   `ValueError: Unrecognized model in ... Should have a `model_type` key in its
   config.json`. This is a diffusers Flux2 checkpoint, not a transformers LLM,
   so there is no `ModelArgs` to adjust into place.
2. Even given a config, the canonical `MLP` is the Llama shape -- three separate
   projections, `w2(silu(w1(x)) * w3(x))`. `Flux2FeedForward` is one FUSED
   `linear_in` emitting both SwiGLU halves as a single 24576-wide output, then
   `linear_out`. The weights do not correspond, so no constructor argument makes
   the canonical class compute this layer.

The native implementation is `TtFlux2FeedForward` in `_flux2_ttnn.py`, which
reads its dimensions off the reference module and so fits `ff_context` exactly
as it fits `ff`.

TENSOR-PARALLEL SCHEME (TP=8, validated by gathered PCC)
--------------------------------------------------------
Textbook column-then-row, with the one twist this model's fused `linear_in`
forces.

* `linear_in` is COLUMN-parallel: its output feeds a per-element gate, so no
  collective is needed after it. But the naive contiguous split is WRONG here --
  the 24576 columns are SwiGLU's two 12288-wide halves concatenated, and
  `silu(x1) * x2` pairs column `i` of the first half with column `i` of the
  second. A contiguous 8-way split would put all of `x1` on chips 0-3 and all of
  `x2` on chips 4-7, so the gate would multiply features living on different
  chips. The weight columns are therefore REORDERED at load time (`_regroup` in
  `_flux2_ttnn.py`, driven by `groups=[inner, inner]`) into
  `[x1_chip0, x2_chip0, x1_chip1, x2_chip1, ...]`, after which a plain
  `ShardTensorToMesh(dim=-1)` hands each chip 1536 MATCHING columns of both
  halves and the local SwiGLU is exactly the global one restricted to those
  features.
* `linear_out` is ROW-parallel: it reduces the 12288-wide activation back to the
  model dim, so its INPUT features are split (12288/8 = 1536 rows -- the same
  1536 features each chip just produced). Each chip holds a partial sum over the
  full output and one `all_reduce` (SUM) completes it.
* `linear_out`'s bias, if present, is added once after the reduce, not per chip.

Math is unchanged: the same products, summed in a different order, so the
gathered output equals the single-device golden. The forward is pure ttnn: no
torch math, no device->host readback.
"""

from __future__ import annotations

from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import (
    TtFlux2FeedForward,
    as_rank4,
    restore_rank,
    to_device,
)


class TtMlp:
    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("mlp needs the torch reference module for weights")
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
    return TtMlp.build(device, torch_module)


def mlp(device, torch_module=None):
    return TtMlp.build(device, torch_module)
