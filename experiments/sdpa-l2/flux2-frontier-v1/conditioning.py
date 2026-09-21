# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in FLUX.2 conditioning repair, held identical across attention variants."""

import functools
import math

import torch
import ttnn

from models.tt_dit.utils.tensor import from_torch


def install(transformer):
    embedding = transformer.time_guidance_embed
    factors = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128)
    embedding.time_proj_factor = ttnn.unsqueeze_to_4D(
        from_torch(factors, device=embedding.mesh_device, dtype=ttnn.float32)
    )
    original = embedding.forward

    @functools.wraps(original)
    def forward(*, timestep, guidance=None, pooled_projection=None):
        if guidance is not None:
            guidance = ttnn.typecast(guidance, ttnn.float32) * 1000.0
        return original(timestep=timestep, guidance=guidance, pooled_projection=pooled_projection)

    embedding.forward = forward
