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
    """Build the combined causal + image-bidirectional 4D attention mask on device.

    Args:
        seq_len: sequence length S (must equal token_type_ids.shape[1])
        token_type_ids: [B, S] CPU tensor (non-zero = image/multimodal token)
        mesh_device: TTNN mesh device
        dtype: dtype for ALL tensors (intermediates + output). Use bfloat4_b to
            reduce memory 4× vs bfloat16 — critical for large S (e.g. S=32768
            needs 512 MB vs 2 GB). bfloat4_b is exact for 0.0 and -inf values.
        causal_cache: optional pre-built [1, 1, S, S] lower-triangular mask from
            build_causal_mask(). When provided, skips the 32 MB H2D ones upload
            and ttnn.tril call. Must match seq_len.

    Returns:
        mask [B, 1, S, S] TTNN tensor replicated across devices.
        0.0 where attention is allowed; -inf where blocked.
    """
    B, S = token_type_ids.shape
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    def _upload(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    # Upload is_mm as two tiny broadcastable tensors [B,1,S,1] and [B,1,1,S]
    is_mm = (token_type_ids != 0).float()
    is_mm_q = _upload(is_mm.reshape(B, 1, S, 1))
    is_mm_k = _upload(is_mm.reshape(B, 1, 1, S))

    # img_mm[b, q, k] = 1 iff both positions are image tokens (outer product)
    img_mm = ttnn.mul(is_mm_q, is_mm_k, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(is_mm_q)
    ttnn.deallocate(is_mm_k)

    # causal = lower-triangular 1s [1, 1, S, S]
    # Use pre-built cache when available — avoids 32 MB H2D + ttnn.tril per call
    if causal_cache is not None:
        causal = causal_cache
        owns_causal = False
    else:
        ones = _upload(torch.ones(1, 1, S, S))
        causal = ttnn.tril(ones)
        ttnn.deallocate(ones)
        owns_causal = True

    # allowed = causal OR img_mm  (max of 0/1 floats)
    allowed = ttnn.maximum(causal, img_mm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if owns_causal:
        ttnn.deallocate(causal)
    ttnn.deallocate(img_mm)

    # Convert to additive bias: 0.0 where allowed, -inf where blocked
    zeros = ttnn.zeros_like(allowed)
    neg_inf = _upload(torch.full((1, 1, 1, 1), float("-inf")))
    threshold = ttnn.full_like(allowed, 0.5)
    condition = ttnn.gt(allowed, threshold, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(threshold)
    ttnn.deallocate(allowed)

    bias = ttnn.where(condition, zeros, neg_inf, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(condition)
    ttnn.deallocate(zeros)
    ttnn.deallocate(neg_inf)

    return bias
