# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 prefill attention mask: causal + image-bidirectional override.

Image tokens (token_type_ids == 1) attend to ALL other image tokens regardless
of causal order. Text tokens remain strictly causal.

Built entirely on device using TTNN ops. The causal lower-triangular mask is
static (depends only on S, never on the input) and can be pre-built once during
model init via build_causal_mask_cache(). At inference time, passing the cached
causal tensor skips the 32 MB H2D upload and ttnn.tril call.
"""

import torch

import ttnn


def build_causal_mask(seq_len: int, mesh_device) -> ttnn.Tensor:
    """Build a causal lower-triangular 1s mask [1, 1, S, S] on device.

    Called during model init for each bucket size. Result should be cached
    and passed to build_molmo2_prefill_mask as causal_cache.
    """
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    ones = ttnn.from_torch(
        torch.ones(1, 1, seq_len, seq_len, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    causal = ttnn.tril(ones)
    ttnn.deallocate(ones)
    return causal


def build_molmo2_prefill_mask(
    seq_len: int,
    token_type_ids: torch.Tensor,
    mesh_device,
    dtype=ttnn.bfloat16,
    causal_cache: ttnn.Tensor = None,
) -> ttnn.Tensor:
    """Build the combined causal + image-bidirectional 4D attention mask.

    Built on CPU with PyTorch (bool arithmetic, no DRAM constraint), then
    uploaded to device. This avoids mixed-dtype TTNN ops and lets dtype control
    device memory: bfloat16 (2× bytes) for small S, bfloat4_b (0.5× bytes) for
    large S where bfloat16 would OOM.

    Args:
        seq_len: sequence length S (must equal token_type_ids.shape[1])
        token_type_ids: [B, S] CPU tensor (non-zero = image/multimodal token)
        mesh_device: TTNN mesh device
        dtype: upload dtype (bfloat16 or bfloat4_b)
        causal_cache: unused (kept for API compatibility)

    Returns:
        mask [B, 1, S, S] TTNN tensor replicated across devices.
        0.0 where attention is allowed; -inf where blocked.
    """
    B, S = token_type_ids.shape

    # Build on CPU — bool ops are cheap and CPU RAM is not a constraint.
    is_mm = token_type_ids != 0  # [B, S] bool
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool))  # [S, S] bool
    img_mm = is_mm[:, :, None] & is_mm[:, None, :]  # [B, S, S] bool
    allowed = causal.unsqueeze(0) | img_mm  # [B, S, S] bool
    bias = torch.where(
        allowed.unsqueeze(1),  # [B, 1, S, S]
        torch.zeros(1, dtype=torch.bfloat16),
        torch.full((1,), float("-inf"), dtype=torch.bfloat16),
    )  # [B, 1, S, S] bfloat16

    return ttnn.from_torch(
        bias,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
