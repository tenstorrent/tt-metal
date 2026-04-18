# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
HF-aligned prefill attention bias for Molmo2 text self-attention.

Hugging Face combines ``create_causal_mask`` with ``token_type_ids_mask_function``
(modeling_molmo2): image / multimodal tokens (token_type_ids != 0) use bidirectional
attention among themselves; all other pairs follow causality.
"""

from typing import Optional

import torch

import ttnn


def build_molmo2_prefill_attention_bias(
    token_type_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Additive attention bias for scaled_dot_product_attention: 0 = allowed, -inf = disallowed.

    Shape: ``[B, 1, S, S]`` (broadcasts to query heads).

    Args:
        token_type_ids: ``[B, S]`` — non-zero marks HF multimodal / image token positions
            (from Molmo2Processor outputs).
        attention_mask: Optional ``[B, S]`` with 1 = valid, 0 = pad (HF padding mask).
        dtype: Compute dtype (use float32 for mask build; convert to bfloat16 for TTNN).

    Returns:
        Bias tensor on the same device as ``token_type_ids``.
    """
    B, S = token_type_ids.shape
    device = token_type_ids.device
    is_mm = token_type_ids != 0
    if is_mm.dtype != torch.bool:
        is_mm = is_mm.bool()

    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
    img_mm = is_mm[:, :, None] & is_mm[:, None, :]
    allowed = causal.unsqueeze(0) | img_mm

    if attention_mask is not None:
        pad = attention_mask.bool()
        allowed = allowed & pad[:, None, :] & pad[:, :, None]

    neg_inf = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)
    zero = torch.zeros((), dtype=dtype, device=device)
    bias = torch.where(allowed, zero, neg_inf)
    return bias.unsqueeze(1)


def build_molmo2_prefill_attention_bias_ttnn(
    token_type_ids: torch.Tensor,
    mesh_device,
    mesh_mapper,
    attention_mask: Optional[torch.Tensor] = None,
) -> ttnn.Tensor:
    """
    Build prefill attention bias ON DEVICE using ttnn ops.

    Same logic as build_molmo2_prefill_attention_bias but runs on device.

    Args:
        token_type_ids: ``[B, S]`` — non-zero marks image/multimodal positions
        mesh_device: TTNN mesh device
        mesh_mapper: Mesh mapper for replication
        attention_mask: Optional ``[B, S]`` with 1 = valid, 0 = pad

    Returns:
        Bias tensor on device: ``[B, 1, S, S]`` bfloat16
    """
    B, S = token_type_ids.shape

    # Create is_mm mask: [B, S] -> 1.0 where multimodal, 0.0 elsewhere
    is_mm_cpu = (token_type_ids != 0).float()

    # Transfer to device
    is_mm_ttnn = ttnn.from_torch(
        is_mm_cpu.reshape(B, 1, S, 1),  # [B, 1, S, 1] for broadcasting
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    is_mm_t_ttnn = ttnn.from_torch(
        is_mm_cpu.reshape(B, 1, 1, S),  # [B, 1, 1, S] for broadcasting
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # img_mm = is_mm[:, :, None] & is_mm[:, None, :] -> outer product
    # Using multiplication for logical AND on 0/1 floats
    img_mm = ttnn.mul(is_mm_ttnn, is_mm_t_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(is_mm_ttnn)
    ttnn.deallocate(is_mm_t_ttnn)

    # Create causal mask: [1, 1, S, S] lower triangular
    ones_cpu = torch.ones(1, 1, S, S, dtype=torch.bfloat16)
    ones_ttnn = ttnn.from_torch(
        ones_cpu,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    causal = ttnn.tril(ones_ttnn, diagonal=0)
    ttnn.deallocate(ones_ttnn)

    # allowed = causal | img_mm -> max(causal, img_mm) for 0/1 floats
    allowed = ttnn.maximum(causal, img_mm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(causal)
    ttnn.deallocate(img_mm)

    # Apply padding mask if provided
    if attention_mask is not None:
        pad_cpu = attention_mask.float()
        pad_row = ttnn.from_torch(
            pad_cpu.reshape(B, 1, S, 1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        pad_col = ttnn.from_torch(
            pad_cpu.reshape(B, 1, 1, S),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        # allowed = allowed & pad_row & pad_col
        allowed = ttnn.mul(allowed, pad_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        allowed = ttnn.mul(allowed, pad_col, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(pad_row)
        ttnn.deallocate(pad_col)

    # Convert to additive mask: 0 where allowed, -inf where not
    # bias = where(allowed > 0.5, 0, -inf)
    zeros = ttnn.zeros_like(allowed)
    neg_inf_val = torch.finfo(torch.bfloat16).min
    neg_inf_cpu = torch.full((1, 1, 1, 1), neg_inf_val, dtype=torch.bfloat16)
    neg_inf_ttnn = ttnn.from_torch(
        neg_inf_cpu,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # where(allowed > 0.5, 0, -inf)
    threshold = ttnn.full_like(allowed, 0.5)
    condition = ttnn.gt(allowed, threshold, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(threshold)
    ttnn.deallocate(allowed)

    bias = ttnn.where(condition, zeros, neg_inf_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(condition)
    ttnn.deallocate(zeros)
    ttnn.deallocate(neg_inf_ttnn)

    return bias
