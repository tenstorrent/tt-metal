# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementations of the first Qwen Image 2.1 DiT primitives.

Inputs and weights use the upstream checkpoint's logical layout. All tensors
stay on the specified device until the caller explicitly collects a result.
"""

from __future__ import annotations

import torch
import ttnn


def to_device(tensor: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None,
    )


def to_host(tensor: ttnn.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    shards = ttnn.get_device_tensors(tensor)
    result = ttnn.to_torch(shards[0]) if len(shards) > 1 else ttnn.to_torch(tensor)
    if result.shape != shape:
        result = result[tuple(slice(0, dimension) for dimension in shape)]
    return result.contiguous()


def linear(hidden: ttnn.Tensor, weight: torch.Tensor, device, compute_kernel_config) -> ttnn.Tensor:
    # PyTorch Linear stores [out, in]; TTNN matmul consumes [in, out].
    rhs = to_device(weight.T.contiguous(), device)
    return ttnn.matmul(
        hidden,
        rhs,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )


def layer_norm(hidden: ttnn.Tensor, eps: float = 1e-6) -> ttnn.Tensor:
    return ttnn.layer_norm(hidden, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def rms_norm(hidden: ttnn.Tensor, weight: torch.Tensor, device, eps: float = 1e-6) -> ttnn.Tensor:
    scale = to_device(weight.reshape(1, 1, 1, -1), device)
    return ttnn.rms_norm(hidden, weight=scale, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def split_half_indices(head_dim: int) -> torch.Tensor:
    """Reorder adjacent complex pairs for TT's half-rotation RoPE kernel.

    Both Q and K use this permutation, so their dot product is unchanged. A
    production weight loader can permute projection rows and norm scales once.
    """
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    return torch.cat((torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)))


def rotary_caches(rotary_complex: torch.Tensor, device) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    cos = torch.cat((rotary_complex.real, rotary_complex.real), dim=-1)[None, None]
    sin = torch.cat((rotary_complex.imag, rotary_complex.imag), dim=-1)[None, None]
    return to_device(cos, device), to_device(sin, device)


def rotary_split_half(hidden: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.experimental.rotary_embedding_hf(
        hidden, cos, sin, is_decode_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def silu(hidden: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.silu(hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def multiply(left: ttnn.Tensor, right: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.multiply(left, right, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def select_rows_by_mask(rows: ttnn.Tensor, target_mask: torch.Tensor, width: int, device) -> ttnn.Tensor:
    """Select timestep or t=0 row for each token without reading TT data back."""
    condition = to_device(target_mask.reshape(1, -1, 1).to(torch.bfloat16), device)
    real = ttnn.reshape(ttnn.slice(rows, (0, 0), (1, width)), (1, 1, width))
    zero = ttnn.reshape(ttnn.slice(rows, (1, 0), (2, width)), (1, 1, width))
    return ttnn.where(condition, real, zero, memory_config=ttnn.DRAM_MEMORY_CONFIG)
