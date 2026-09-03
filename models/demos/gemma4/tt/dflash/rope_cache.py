# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device RoPE cos/sin for the DFlash drafter, replacing generate.py's former host-torch
per-iteration trig computation (build_noise_inputs). Mirrors the target model's own
``create_rope_caches`` 2D/embedding-gather pattern exactly (models/demos/gemma4/tt/model.py) --
a one-time host computation at setup (the same standard practice the target's own cache
uses, not a per-iteration cost), uploaded as a ROW_MAJOR [max_seq_len, head_dim] table,
gathered per call via ``ttnn.embedding``.

The drafter needs its OWN cache, not the target's: its rope_theta (from the DFlash
checkpoint's own config) is independent of whatever the target model's layers use.
"""

from __future__ import annotations

import torch

import ttnn


def build_dflash_rope_cache_2d(mesh_device, head_dim: int, rope_theta: float, max_seq_len: int):
    """Returns (cos_2d, sin_2d): ROW_MAJOR bf16 [max_seq_len, head_dim] ttnn tensors,
    replicated across the mesh. Formula matches generate.py's (now-removed)
    ``build_noise_inputs`` exactly -- confirmed bit-identical (bf16) against the real
    torch reference's own ``Qwen3RotaryEmbedding`` output for known positions."""
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim//2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq_len, head_dim]
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    cos_2d = ttnn.from_torch(
        cos, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
    )
    sin_2d = ttnn.from_torch(
        sin, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
    )
    return cos_2d, sin_2d


def gather_rope_on_device(mesh_device, positions: list[int], cos_2d, sin_2d, head_dim: int):
    """Gather cos/sin rows for ``positions`` (a contiguous absolute-position range) via
    ttnn.embedding, entirely on device -- no host trig, no per-call from_torch of the
    trig VALUES (only the small integer index tensor is built on host, same as every
    other position/index tensor already built this way elsewhere in this port).

    Returns (cos_tt, sin_tt): [1,1,len(positions),head_dim] bf16, matching what
    dflash_attention_forward expects for cos_full/sin_full."""
    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    idx = torch.tensor(positions, dtype=torch.int32).reshape(1, -1)
    idx_tt = ttnn.from_torch(
        idx, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=replicate
    )

    cos = ttnn.embedding(idx_tt, cos_2d, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(idx_tt, sin_2d, layout=ttnn.TILE_LAYOUT)
    ttnn.deallocate(idx_tt)
    return ttnn.unsqueeze_to_4D(cos), ttnn.unsqueeze_to_4D(sin)
