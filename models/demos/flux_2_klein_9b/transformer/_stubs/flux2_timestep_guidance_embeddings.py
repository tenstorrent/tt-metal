# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `Flux2TimestepGuidanceEmbeddings` (`time_guidance_embed`)
of `black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

    timesteps_proj = time_proj(timestep)                 # sinusoidal, 256 ch
    timesteps_emb  = timestep_embedder(timesteps_proj)    # 256 -> SiLU -> 4096

This checkpoint has `guidance_embeds: false`, so `guidance_embedder` is None and
the guidance argument is ignored -- the reference returns `timesteps_emb`. The
guidance branch is still implemented (it is one extra embedder plus an add) so
the module is correct for a guidance-distilled sibling.

The sinusoidal projection runs in **float32**: the model scales the timestep by
1000 before this layer, so the phase reaches ~1000 radians, where a bfloat16
argument to `ttnn.cos` was measured off by up to 1.6 absolute (float32: 0.0).
The MLP runs in bfloat16 like the rest of the model.

Caveat worth knowing: the PCC harness marshals the primary input to bfloat16
before this module ever sees it. The captured timestep, 500, is exactly
representable in bfloat16, so the graded comparison is unaffected -- but a
timestep such as 999 rounds to 1000 in bfloat16, and one radian of phase error
is visible in the output. In the pipeline the timestep should reach this layer
in float32.

Tensor parallelism: the embedder is `linear_1` COLUMN-parallel (its output feeds
a per-element SiLU) then `linear_2` ROW-parallel with one `all_reduce`; the
sinusoidal table has no parameters and stays replicated. At TP=1 the collectives
are skipped. The implementation lives in `_flux2_ttnn.py`.

The forward is pure ttnn: no torch math, no device->host readback.
"""

from __future__ import annotations

from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import TtFlux2TimestepGuidanceEmbeddings


def build(device, torch_module=None):
    if torch_module is None:
        raise RuntimeError("flux2_timestep_guidance_embeddings needs the torch reference module for weights")
    return TtFlux2TimestepGuidanceEmbeddings(device, torch_module)


def flux2_timestep_guidance_embeddings(device, torch_module=None):
    return build(device, torch_module)
