# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic FlowMatch Euler latent update on TTNN."""

from __future__ import annotations

import torch
import ttnn


def flow_euler_step(
    latents: ttnn.Tensor, velocity: ttnn.Tensor, sigma: float, next_sigma: float, device
) -> ttnn.Tensor:
    """Match Diffusers' BF16 product followed by an fp32 latent addition."""
    sample32 = ttnn.typecast(latents, ttnn.float32)
    delta = ttnn.from_torch(
        torch.tensor([[[next_sigma - sigma]]], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    product_bf16 = ttnn.multiply(velocity, delta)
    updated32 = ttnn.add(sample32, ttnn.typecast(product_bf16, ttnn.float32))
    return ttnn.typecast(updated32, ttnn.bfloat16)
