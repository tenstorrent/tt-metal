# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LMHead — final RMSNorm + vocab projection, bringup on QB (device 0).

Config: hidden_size=2688, vocab_size=131072, tie_word_embeddings=False.
"""

import torch

import ttnn
from ttnn import MeshDevice

NORM_EPS = 1e-5

_R = ttnn.ReplicateTensorToMesh
_C = ttnn.ConcatMeshToTensor


def lm_head_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [B, S, 2688] bf16 CPU
    norm_f_weight: torch.Tensor,  # [2688] bf16 CPU
    lm_head_weight: torch.Tensor,  # [131072, 2688] bf16 CPU
    norm_eps: float = NORM_EPS,
) -> torch.Tensor:
    """Returns [B, S, 131072] bfloat16 (CPU)."""
    h_tt = ttnn.from_torch(
        hidden_states.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    w_tt = ttnn.from_torch(
        norm_f_weight.bfloat16().unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    normed_tt = ttnn.rms_norm(h_tt, epsilon=norm_eps, weight=w_tt)

    wl_tt = ttnn.from_torch(
        lm_head_weight.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    logits_tt = ttnn.linear(normed_tt, wl_tt, transpose_b=True)
    return ttnn.to_torch(logits_tt, mesh_composer=_C(mesh_device, dim=0)).bfloat16()
