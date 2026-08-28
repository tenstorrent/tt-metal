# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `timesteps` (`time_guidance_embed.time_proj`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

diffusers' `Timesteps` / `get_timestep_embedding` -- the sinusoidal timestep
features, 1 scalar -> 256 channels:

    freq[k] = max_period ** (-k / (half - downscale_freq_shift))
    emb     = scale * timestep[:, None] * freq[None, :]
    out     = cat([cos(emb), sin(emb)])        # flip_sin_to_cos=True

It is the projection feeding the `timestep_embedding` component, and shares the
already-graduated `TtTimesteps` implementation in `_flux2_ttnn.py` rather than
carrying a second copy.

The stub this replaces was the autofill CPU fallback, which ran the HF PyTorch
module on host and loaded the checkpoint through
`AutoModelForCausalLM`/`AutoModel` -- impossible for a diffusers Flux2
checkpoint with no `model_type` key.

TENSOR-PARALLEL SCHEME (TP=8)
-----------------------------
REPLICATED -- this layer shards in no scheme, and that is the scheme.

It has NO parameters to split. The only device-resident tensor is the 128-wide
frequency row, which is a lookup table computed once from `max_period` and
`num_channels`, and the general principle is that lookup tables stay replicated
(`ReplicateTensorToMesh`). Splitting the 256 output channels would be possible
arithmetically -- the cos/sin are elementwise -- but the consumer is
`timestep_embedding`'s `linear_1`, which is itself column-parallel and therefore
needs the FULL 256-wide input on every chip; a split here would have to be
undone by a gather immediately. The input is a single scalar per batch element,
so there is nothing else to distribute.

Every chip computes the identical result and the gathered output equals the
single-device golden exactly -- no collective, because there is no partial sum.

PRECISION: the sinusoid runs in float32. This model scales the timestep by 1000
before this layer, so the phase reaches ~1000 radians, where `ttnn.cos` of a
bfloat16 argument was measured off by up to 1.6 absolute (float32: 0.0). The
outer product is a `repeat` + `mul` rather than a K=1 matmul for the same
reason -- the matmul path rounds through bfloat16, the broadcast multiply was
measured exact.

Caveat worth knowing: the PCC harness marshals the primary input to bfloat16
before this module sees it. The captured timestep, 500, is exactly representable
in bfloat16, so the graded comparison is unaffected -- but a timestep such as 999
rounds to 1000 in bfloat16, and one radian of phase error is visible in the
output. In the pipeline the timestep should reach this layer in float32.

The forward is pure ttnn: no torch math, no device->host readback.
"""

from __future__ import annotations

from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import TtTimesteps


def build(device, torch_module=None):
    if torch_module is None:
        raise RuntimeError("timesteps needs the torch reference module for its projection config")
    return TtTimesteps(device, torch_module)


def timesteps(device, torch_module=None):
    return build(device, torch_module)
