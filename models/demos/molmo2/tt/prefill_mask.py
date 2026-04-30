# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 prefill attention mask: causal + image-bidirectional override.

Image tokens (token_type_ids == 1) attend to ALL other image tokens regardless
of causal order. Text tokens remain strictly causal.

This matches the reference `build_prefill_mask()` in functional.py but produces
a TTNN tensor replicated across all devices.
"""

import torch

import ttnn


def build_molmo2_prefill_mask(
    seq_len: int,
    token_type_ids: torch.Tensor,
    mesh_device,
    dtype=ttnn.bfloat8_b,
) -> ttnn.Tensor:
    """Build the combined causal + image-bidirectional 4D attention mask.

    Args:
        seq_len: sequence length S
        token_type_ids: [B, S] CPU tensor (1=image token, 0=text token)
        mesh_device: TTNN mesh device
        dtype: TTNN dtype for the mask (use bfloat8_b to save memory at large ISL)

    Returns:
        mask [B, 1, S, S] TTNN tensor replicated across devices.
        0.0 where attention is allowed; -inf where blocked.
    """
    B = token_type_ids.shape[0]
    device = token_type_ids.device

    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)  # [S, 1]
    kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S]

    causal_block = kv_idx > q_idx  # [S, S]

    is_image = token_type_ids == 1  # [B, S]
    is_image_q = is_image.unsqueeze(2)  # [B, S, 1]
    is_image_kv = is_image.unsqueeze(1)  # [B, 1, S]
    image_override = is_image_q & is_image_kv  # [B, S, S]

    block = causal_block.unsqueeze(0) & ~image_override  # [B, S, S]
    mask = torch.where(
        block,
        torch.tensor(float("-inf"), dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
    )
    mask = mask.unsqueeze(1)  # [B, 1, S, S]

    return ttnn.from_torch(
        mask.to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
