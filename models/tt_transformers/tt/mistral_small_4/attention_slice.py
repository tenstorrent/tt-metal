# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mistral4 attention — narrow slices for incremental bring-up (no SDPA yet)."""

from __future__ import annotations

import torch

from models.tt_transformers.tt.mistral_small_4.linear import linear_bf16_no_bias
from models.tt_transformers.tt.mistral_small_4.rms_norm import rms_norm_bf16


def attention_q_after_q_bottleneck_bf16(
    mesh_device,
    hidden_states_bsh: torch.Tensor,
    q_a_weight_out_in: torch.Tensor,
    q_a_layernorm_weight_1d: torch.Tensor,
    q_a_layernorm_eps: float,
    q_b_weight_out_in: torch.Tensor,
) -> torch.Tensor:
    """
    HF ``Mistral4Attention`` Q path **before** reshape / RoPE / attention:

    ``q_b_proj( q_a_layernorm( q_a_proj( hidden_states ) ) )``

    Args:
        hidden_states_bsh: ``[B, S, hidden_size]``.
        q_a_weight_out_in: ``q_a_proj.weight`` ``[q_lora_rank, hidden_size]``.
        q_a_layernorm_weight_1d: ``q_a_layernorm.weight`` ``[q_lora_rank]``.
        q_b_weight_out_in: ``q_b_proj.weight`` ``[num_heads * qk_head_dim, q_lora_rank]``.

    Returns:
        Host bf16 ``[B, S, num_heads * qk_head_dim]``.

    ``hidden_size``, ``q_lora_rank``, and ``num_heads * qk_head_dim`` must be multiples of 32
    for current RMSNorm / linear TILE paths.
    """
    q_a = linear_bf16_no_bias(mesh_device, hidden_states_bsh, q_a_weight_out_in)
    q_a_normed = rms_norm_bf16(mesh_device, q_a, q_a_layernorm_weight_1d, epsilon=q_a_layernorm_eps)
    return linear_bf16_no_bias(mesh_device, q_a_normed, q_b_weight_out_in)


def attention_kv_b_and_k_rot_from_compressed_bf16(
    mesh_device,
    compressed_bsh: torch.Tensor,
    kv_lora_rank: int,
    kv_a_layernorm_weight_1d: torch.Tensor,
    kv_a_layernorm_eps: float,
    kv_b_weight_out_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    From ``kv_a_proj`` output ``compressed_bsh`` (host bf16), run ``kv_a_layernorm`` + ``kv_b_proj`` on device
    and return ``(kv_b_out, k_rot)`` with ``k_rot`` the rotary slice (host, unchanged).
    """
    kv_a_out = int(compressed_bsh.shape[-1])
    rope_d = kv_a_out - int(kv_lora_rank)
    if rope_d <= 0:
        raise ValueError("compressed last dim must exceed kv_lora_rank")
    k_pass, k_rot = torch.split(compressed_bsh, [int(kv_lora_rank), rope_d], dim=-1)
    k_normed = rms_norm_bf16(mesh_device, k_pass, kv_a_layernorm_weight_1d, epsilon=kv_a_layernorm_eps)
    k_after_b = linear_bf16_no_bias(mesh_device, k_normed, kv_b_weight_out_in)
    return k_after_b, k_rot


def attention_kv_after_kv_b_bottleneck_bf16(
    mesh_device,
    hidden_states_bsh: torch.Tensor,
    kv_a_weight_out_in: torch.Tensor,
    kv_lora_rank: int,
    kv_a_layernorm_weight_1d: torch.Tensor,
    kv_a_layernorm_eps: float,
    kv_b_weight_out_in: torch.Tensor,
) -> torch.Tensor:
    """
    HF ``Mistral4Attention`` KV **compressed** path up through ``kv_b_proj`` (before head ``view`` / transpose):

    ``kv_b_proj( kv_a_layernorm( k_pass ) )`` where ``k_pass`` is the first split of ``kv_a_proj_with_mqa``.

    The rotary slice ``k_rot`` from ``kv_a`` is ignored here (same as HF until RoPE).

    Args:
        hidden_states_bsh: ``[B, S, hidden_size]``.
        kv_a_weight_out_in: ``kv_a_proj_with_mqa.weight`` ``[kv_lora_rank + qk_rope_head_dim, hidden_size]``.
        kv_lora_rank: ``config.kv_lora_rank`` (length of the compressed ``k_pass`` channel).
        kv_a_layernorm_weight_1d: ``kv_a_layernorm.weight`` ``[kv_lora_rank]``.
        kv_b_weight_out_in: ``kv_b_proj.weight`` ``[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]``.

    Returns:
        Host bf16 ``[B, S, num_heads * (qk_nope_head_dim + v_head_dim)]``.

    Relevant dims should be multiples of 32 for current linear / RMSNorm paths.
    """
    kv_a_out = int(kv_a_weight_out_in.shape[0])
    rope_d = kv_a_out - int(kv_lora_rank)
    if rope_d <= 0:
        raise ValueError("kv_a weight rows must exceed kv_lora_rank (room for rope slice)")

    compressed = linear_bf16_no_bias(mesh_device, hidden_states_bsh, kv_a_weight_out_in)
    k_after_b, _k_rot = attention_kv_b_and_k_rot_from_compressed_bf16(
        mesh_device,
        compressed,
        kv_lora_rank,
        kv_a_layernorm_weight_1d,
        kv_a_layernorm_eps,
        kv_b_weight_out_in,
    )
    return k_after_b
