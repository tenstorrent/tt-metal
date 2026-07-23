# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# On-device init-noise for the DiT denoise loop (no host ``torch.randn``).

from __future__ import annotations

import ttnn


def randn_init_latent_tt(
    device,
    shape: tuple[int, int, int, int],
    *,
    seed: int = 0,
    dtype=None,
    layout=None,
) -> ttnn.Tensor:
    """Sample N(0, 1) init noise on device as flat NHWC ``[1, 1, B*H*W, C]``.

    ``shape`` is torch-style ``(B, C, H, W)``. Default layout is **TILE** so the
    tensor matches the resident denoise / Euler path (``HY_LATENT_RESIDENT=1``).

    On a multi-device mesh, ``ttnn.randn`` draws independently per chip — we gather
    device-0 and re-upload with ``ReplicateTensorToMesh`` so every device shares
    identical noise (same contract as host ``torch.randn`` + replicate). That is
    one transport round-trip for consistency only; RNG itself stays on device.
    """
    if dtype is None:
        dtype = ttnn.bfloat16
    if layout is None:
        layout = ttnn.TILE_LAYOUT
    batch, channels, height, width = (int(x) for x in shape)
    flat = (1, 1, batch * height * width, channels)
    noise = ttnn.randn(
        flat,
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        seed=int(seed),
    )
    n_dev = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    if n_dev > 1:
        # Align all mesh chips to device-0's draw (replicate semantics).
        host = ttnn.to_torch(noise, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[:batch]
        # host is still flat [1,1,B*H*W,C] after taking first mesh shard's leading rows
        # ConcatMeshToTensor(dim=0) stacks [1,1,...] → [n_dev,1,...]; take [:1] then reshape.
        if host.ndim == 4 and int(host.shape[0]) >= 1:
            host = host[:1].contiguous()
        ttnn.deallocate(noise)
        noise = ttnn.from_torch(
            host,
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return noise


def resolve_latent_nchw(
    init_latent,
    *,
    token_h: int,
    token_w: int,
) -> tuple[int, int, int, int]:
    """Return ``(B, C, h, w)`` for a torch NCHW or device flat NHWC init latent."""
    if isinstance(init_latent, ttnn.Tensor):
        shape = list(init_latent.shape)
        if len(shape) != 4 or int(shape[0]) != 1 or int(shape[1]) != 1:
            raise ValueError(f"device init_latent must be flat NHWC [1,1,B*H*W,C], got shape={tuple(shape)}")
        n, channels = int(shape[2]), int(shape[3])
        hw = int(token_h) * int(token_w)
        if hw <= 0 or n % hw != 0:
            raise ValueError(f"device init_latent length {n} is not divisible by token grid " f"{token_h}x{token_w}")
        return n // hw, channels, int(token_h), int(token_w)

    import torch

    if isinstance(init_latent, torch.Tensor):
        if init_latent.ndim != 4:
            raise ValueError(f"torch init_latent must be NCHW, got shape={tuple(init_latent.shape)}")
        return tuple(int(x) for x in init_latent.shape)

    raise TypeError(f"init_latent must be torch NCHW or ttnn flat NHWC, got {type(init_latent)}")
