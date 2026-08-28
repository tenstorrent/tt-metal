# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `timestep_embedding`
(`time_guidance_embed.timestep_embedder`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

    x = linear_1(sample)      # 256 -> 4096
    x = SiLU(x)
    x = linear_2(x)           # 4096 -> 4096

diffusers' `TimestepEmbedding`: the small MLP that turns the 256-channel
sinusoidal timestep features into a model-dim conditioning vector. This
checkpoint has no `cond_proj` and no `post_act`, and `guidance_embeds: false`,
so this single embedder is the whole of the timestep conditioning.

It is the inner MLP of the `flux2_timestep_guidance_embeddings` component and
shares that component's already-graduated implementation
(`TtTimestepEmbedding` in `_flux2_ttnn.py`) rather than carrying a second copy.

The stub this replaces was the autofill CPU fallback: it ran the HF PyTorch
module on host, and reached the device only to marshal tensors back off it. It
also loaded the checkpoint itself via `AutoModelForCausalLM`/`AutoModel`, which
raised `ValueError: Unrecognized model ... Should have a `model_type` key in its
config.json` -- this is a diffusers Flux2 checkpoint, not a transformers model.
That failure was in the stub's own loader, not the test's; the test resolves its
reference through the shared `tests/pcc/_reference_loader.py` and is fine.

Tensor parallelism: textbook column-then-row. `linear_1`'s output feeds a
per-element SiLU, so it is COLUMN-parallel and needs no collective; `linear_2`
reduces back to the model dim, so it is ROW-parallel over the matching input
slice and ends in one `all_reduce` (SUM). At TP=1 `TtLinear` degrades to
`replicate` and both collectives are skipped.

PRECISION NOTE: the input here is already the sinusoidal projection (values in
[-1, 1]), so bfloat16 is fine for this MLP. The float32 requirement documented on
`TtTimesteps` applies to the sinusoid itself, upstream of this module, where the
phase reaches ~1000 radians.

The forward is pure ttnn: no torch math, no device->host readback.
"""

from __future__ import annotations

from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import TtTimestepEmbedding


def build(device, torch_module=None):
    if torch_module is None:
        raise RuntimeError("timestep_embedding needs the torch reference module for weights")
    return TtTimestepEmbedding(device, torch_module)


def timestep_embedding(device, torch_module=None):
    return build(device, torch_module)
