# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `patch_embed` (`x_embedder`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

    hidden_states = x_embedder(latents)      # 128 -> 4096, no bias

The transformer's input projection: it lifts each already-patchified latent
token from `in_channels=128` to the model dim `inner_dim = 32 * 128 = 4096`, and
its output IS the start of the residual stream the blocks then refine.

Note this is a plain `nn.Linear`, NOT a convolution. The config sets
`patch_size: 1`, so patchifying is a reshape done by the pipeline before the
transformer is entered, and `x_embedder` has a single `x_embedder.weight` in the
checkpoint with no `bias` and no kernel/stride.

That is why the auto-seeded REUSE copy of
`models/tt_transformers/tt/multimodal/llama_conv2d_patch.py` was replaced. It
did not even expose the entry point the harness calls (`AttributeError: module
... has no attribute 'patch_embed'`), and fixing that alone would not have made
it correct: `TtLlamaConv2dPatch` indexes `state_dict[prefix + "_linear.weight"]`
and `[... + "_linear.bias"]`, takes `kernel_size`/`stride` constructor args, and
runs `torch.nn.Unfold` on host in its forward -- a conv-patching layer this
model does not have, computed off device. The port below uses the shared
`TtLinear` from `_flux2_ttnn.py`, the same primitive the graduated blocks use.

TENSOR-PARALLEL SCHEME (TP=8)
-----------------------------
COLUMN-parallel with a closing `all_gather`.

`x_embedder` widens 128 -> 4096, so the axis worth splitting is the OUTPUT: each
chip holds 4096/8 = 512 of the output features (16 tiles) and computes them from
the full 128-wide input, which is replicated. The input axis is only 128 wide --
splitting it row-parallel would give each chip 16 features, below a tile, and
would cost an `all_reduce` instead of a gather.

The consumer decides the collective: this projection's output feeds the first
block's LayerNorm, which reduces over the full model dim, so the column shards
must be reassembled before it. One `all_gather` along the feature axis does
that, restoring the full-width replicated residual stream the blocks expect.
Since the split is a single contiguous group, the gather returns the features in
their original order, so the result is exactly the single-device answer. There
is no bias to worry about double-counting.

At TP=1 `TtLinear` degrades to `scheme='replicate'` and the gather is skipped.

The forward is pure ttnn: no torch math, no device->host readback.
"""

from __future__ import annotations

from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import (
    TtLinear,
    all_gather,
    as_rank4,
    num_devices,
    restore_rank,
    to_device,
)


class TtPatchEmbed:
    def __init__(self, device, torch_module, *, tensor_parallel=True) -> None:
        if torch_module is None:
            raise RuntimeError("patch_embed needs the torch reference module for weights")
        self.device = device
        self.tp = num_devices(device) if tensor_parallel else 1
        scheme = "column" if tensor_parallel else "replicate"
        self.proj = TtLinear(device, torch_module, scheme=scheme)
        self.out_features = self.proj.out_features

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, x, **kwargs):
        x4, shape = as_rank4(to_device(self.device, x))
        out = self.proj(x4)
        if self.proj.scheme == "column" and self.tp > 1:
            out = all_gather(self.device, out, 3)
        return restore_rank(out, shape, self.out_features)


def build(device, torch_module=None):
    return TtPatchEmbed.build(device, torch_module)


def patch_embed(device, torch_module=None):
    return TtPatchEmbed.build(device, torch_module)
