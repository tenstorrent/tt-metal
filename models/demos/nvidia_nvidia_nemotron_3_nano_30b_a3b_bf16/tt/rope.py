# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RoPE — bringup on QB (device 0).

Uses ttnn.experimental.rotary_embedding_hf which expects:
  input: [B, nH, S, head_dim] (or [B, 1, S, head_dim] for decode)
  cos/sin: [1, 1, S, head_dim]

For bringup we do the cos/sin computation on host and let the
rotary_embedding kernel handle the rotation.
"""

import torch

import ttnn
from ttnn import MeshDevice

HEAD_DIM = 128
ROPE_THETA = 10000.0

_R = ttnn.ReplicateTensorToMesh
_C = ttnn.ConcatMeshToTensor


def rope_forward(
    mesh_device: MeshDevice,
    query: torch.Tensor,  # [B, nH, S, head_dim]
    key: torch.Tensor,  # [B, nKV, S, head_dim]
    position_ids: torch.Tensor,  # [B, S]
    head_dim: int = HEAD_DIM,
    rope_theta: float = ROPE_THETA,
    **kwargs,
) -> tuple:
    """Apply RoPE using ttnn.experimental.rotary_embedding_hf.

    Returns (query_rotated, key_rotated) on CPU.
    """
    B, nH, S, D = query.shape

    # Build cos/sin for the given positions
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    pos = position_ids[0].float()  # [S]
    freqs = torch.outer(pos, inv_freq)  # [S, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [S, D]
    cos = emb.cos().bfloat16()
    sin = emb.sin().bfloat16()

    # ttnn.experimental.rotary_embedding_hf expects cos/sin as [1, 1, S, D]
    cos_tt = ttnn.from_torch(
        cos.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    sin_tt = ttnn.from_torch(
        sin.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )

    q_tt = ttnn.from_torch(
        query.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=_R(mesh_device)
    )
    q_rot_tt = ttnn.experimental.rotary_embedding_hf(q_tt, cos_tt, sin_tt)
    q_rot = ttnn.to_torch(q_rot_tt, mesh_composer=_C(mesh_device, dim=0)).bfloat16()

    k_tt = ttnn.from_torch(
        key.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=_R(mesh_device)
    )
    k_rot_tt = ttnn.experimental.rotary_embedding_hf(k_tt, cos_tt, sin_tt)
    k_rot = ttnn.to_torch(k_rot_tt, mesh_composer=_C(mesh_device, dim=0)).bfloat16()

    return q_rot, k_rot
