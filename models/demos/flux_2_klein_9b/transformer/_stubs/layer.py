# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native, tensor-parallel TTNN port of `layer` (`transformer_blocks.0.norm1`)
of `black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

    out = layer_norm(x, eps) * weight + bias      # affine terms optional

The pre-attention LayerNorm of a dual-stream block. In this checkpoint the block
norms carry no affine parameters (`elementwise_affine=False`) -- the per-channel
scale and shift are supplied instead by the block's AdaLN modulation, which is
applied outside this module. The port reads `weight`/`bias` off the reference
module and passes whatever is there (`None` when absent) straight to
`ttnn.layer_norm`, so it is correct either way.

This component was auto-seeded as a REUSE copy of
`models/tt_transformers/tt/multimodal/llama_layernorm.py`. That copy does not
fit this model: its constructor takes `(device, dim, state_dict,
state_dict_prefix, ...)` rather than the harness's `(device, torch_module)`, and
it unconditionally indexes `state_dict[prefix + "weight"]` and `[... + "bias"]`,
which an affine-free LayerNorm does not have. It is replaced here by the
`TtLayerNorm` in `_flux2_ttnn.py` -- the same normalisation the graduated Flux2
blocks already run, so this component and its callers stay one implementation.

TENSOR-PARALLEL SCHEME (TP=8, validated by gathered PCC)
--------------------------------------------------------
REPLICATED -- this layer shards in no scheme, and that is the scheme.

LayerNorm reduces the mean and variance over the FULL model dim (4096). Splitting
that axis 8 ways would make each chip normalise by the statistics of its own 512
features, which is a different function, not a redistribution of the same one --
so no `ShardTensorToMesh` here, at any axis. Sharding instead over the SEQUENCE
axis would be arithmetically sound (the norm is per-token), but the residual
stream this norm sits on is full-width and replicated on every chip -- the
attention and feed-forward around it each end in an `all_reduce` -- so a
sequence split would have to be undone by a gather before the very next op and
would buy nothing.

The `weight` and `bias`, when present, are elementwise over the unsplit channel
axis and are staged with `ReplicateTensorToMesh`, per the general principle that
norms, biases and lookup tables stay replicated. Every chip therefore computes
the identical full-width result, and the gathered output equals the
single-device golden exactly -- no collective needed, because there is no
partial sum to complete.

The forward is pure ttnn: no torch math, no device->host readback.
"""

from __future__ import annotations

from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import (
    TtLayerNorm,
    as_rank4,
    restore_rank,
    to_device,
)


class TtLayerStub:
    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("layer needs the torch reference module for weights")
        self.device = device
        self.norm = TtLayerNorm(device, torch_module)

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, x, **kwargs):
        x4, shape = as_rank4(to_device(self.device, x))
        out = self.norm(x4)
        return restore_rank(out, shape)


def build(device, torch_module=None):
    return TtLayerStub.build(device, torch_module)


def layer(device, torch_module=None):
    return TtLayerStub.build(device, torch_module)
