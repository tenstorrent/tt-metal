# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Seeded standard-normal initial image latents, generated entirely on TT."""

import math

import ttnn


def gaussian_from_uniforms(radial, angular):
    """Box–Muller transform of FP32 device uniforms; no host tensor compute."""
    # Avoid log(0). The floor also bounds an exceedingly rare extreme tail.
    radial = ttnn.maximum(radial, 2.0**-24)
    radius = ttnn.sqrt(ttnn.multiply(ttnn.log(radial), -2.0))
    phase = ttnn.multiply(angular, 2.0 * math.pi)
    return ttnn.multiply(radius, ttnn.cos(phase))


def gaussian_uniforms(shape, device, seed: int):
    """Two reproducible device RNG streams, exposed for paired validation."""
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    # TTNN rand treats seed=0 as nondeterministic. Map all user seeds to
    # nonzero stream seeds; use distant offsets rather than adjacent core seeds.
    stream_seeds = [((seed ^ offset) % (2**32 - 1)) + 1 for offset in (0x243F6A88, 0x85A308D3)]
    return tuple(
        ttnn.rand(
            list(shape),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            seed=stream_seed,
        )
        for stream_seed in stream_seeds
    )


def initial_image_latents(device, height: int, width: int, seed: int):
    """Return packed BF16 N(0,1) image latents with shape [1,H/16*W/16,64].

    This RNG is independent of PyTorch's generator. An equal numeric seed is
    reproducible on this TT implementation, not a CUDA/TT same-image guarantee.
    """
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError("image height and width must be positive multiples of 32")
    shape = (1, (height // 16) * (width // 16), 64)
    radial, angular = gaussian_uniforms(shape, device, seed)
    return ttnn.typecast(gaussian_from_uniforms(radial, angular), ttnn.bfloat16)
