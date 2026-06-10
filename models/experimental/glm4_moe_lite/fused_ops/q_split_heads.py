# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fused Q head-split: replaces reshape+permute in prefill attention."""

from __future__ import annotations

import os

import ttnn


def _prefill_q_l1_enabled() -> bool:
    return os.environ.get("GLM4_MOE_LITE_PREFILL_Q_HEADS_L1", "1").strip() != "0"


def _prefill_q_l1_max_bytes(env_key: str, default: int) -> int:
    raw = os.environ.get(env_key, "").strip()
    return int(raw) if raw else default


def prefill_q_nlp_input_memory_config(*, seq_tokens: int, fused_q_dim: int) -> ttnn.MemoryConfig:
    """MC for nlp_create_qkv_heads in0 ``[1,1,S,fused_q_dim]`` (e.g. S=128, 5120 → ~1.25 MiB bf16)."""
    if not _prefill_q_l1_enabled():
        return ttnn.DRAM_MEMORY_CONFIG
    act_bytes = int(seq_tokens) * int(fused_q_dim) * 2  # bf16
    max_bytes = _prefill_q_l1_max_bytes("GLM4_MOE_LITE_PREFILL_Q_NLP_IN_L1_MAX_BYTES", 1536 * 1024)
    if act_bytes <= max_bytes:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def prefill_q_slice_memory_config(*, num_heads: int, seq_tokens: int, slice_dim: int) -> ttnn.MemoryConfig:
    """MC for a Q slice ``[1,H,S,slice_dim]`` (e.g. q_nope ~960 KiB fits L1 when full Q does not)."""
    if not _prefill_q_l1_enabled():
        return ttnn.DRAM_MEMORY_CONFIG
    act_bytes = int(num_heads) * int(seq_tokens) * int(slice_dim) * 2  # bf16
    max_bytes = _prefill_q_l1_max_bytes("GLM4_MOE_LITE_PREFILL_Q_HEADS_L1_MAX_BYTES", 1024 * 1024)
    if act_bytes <= max_bytes:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def prefill_q_heads_memory_config(*, num_heads: int, seq_tokens: int, head_dim: int) -> ttnn.MemoryConfig:
    """MC for nlp_create_qkv_heads output ``[1,H,S,D]``."""
    if not _prefill_q_l1_enabled():
        return ttnn.DRAM_MEMORY_CONFIG
    act_bytes = int(num_heads) * int(seq_tokens) * int(head_dim) * 2  # bf16
    max_bytes = _prefill_q_l1_max_bytes("GLM4_MOE_LITE_PREFILL_Q_HEADS_L1_MAX_BYTES", 1024 * 1024)
    if act_bytes <= max_bytes:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def _to_mc_if_needed(tensor: ttnn.Tensor, mc: ttnn.MemoryConfig) -> ttnn.Tensor:
    cur = tensor.memory_config()
    if cur.buffer_type == mc.buffer_type and cur.memory_layout == mc.memory_layout:
        return tensor
    return ttnn.to_memory_config(tensor, mc)


def q_split_heads(
    q,
    *,
    num_heads,
    head_dim,
    total_seq,
    batch,
    seq_len,
    device,
    memory_config=None,
):
    """Fused [1,1,T,H*D] → [B,H,S,D] using a single nlp_create_qkv_heads kernel pass.

    Replaces the two-op sequence:
        q = ttnn.reshape(q, (batch, seq_len, num_heads, head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))

    For batch=1 this is a single kernel pass (1 read + 1 write vs the original
    2 reads + 2 writes), saving ~175 µs per prefill step.
    For batch>1 the kernel is followed by a cheap reshape [1,H,T,D]→[B,H,S,D],
    which moves the same data volume as the original permute.

    Uses num_kv_heads=0 with three pre-allocated output_tensors to satisfy both
    bypass paths in nlp_create_qkv_heads_device_operation.cpp:
      - compute_output_specs returns specs from the pre-allocated tensors directly
      - create_output_tensors returns the tensors directly (no zero-volume alloc)
    This lets the program factory receive non-null K/V buffers while kv_out_c=0
    causes the writer kernel to skip all K/V tile writes.
    """
    fused_q_dim = int(num_heads) * int(head_dim)
    in_mc = prefill_q_nlp_input_memory_config(seq_tokens=total_seq, fused_q_dim=fused_q_dim)
    out_mc = (
        memory_config
        if memory_config is not None
        else prefill_q_heads_memory_config(num_heads=num_heads, seq_tokens=total_seq, head_dim=head_dim)
    )

    q = _to_mc_if_needed(q, in_mc)

    q_alloc = ttnn.empty(
        [1, num_heads, total_seq, head_dim],
        dtype=q.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mc,
    )
    k_dummy = ttnn.empty(
        [1, 1, ttnn.TILE_SIZE, head_dim],
        dtype=q.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mc,
    )
    v_dummy = ttnn.empty(
        [1, 1, ttnn.TILE_SIZE, head_dim],
        dtype=q.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mc,
    )

    q_out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
        q,
        None,
        num_heads=num_heads,
        num_kv_heads=0,
        transpose_k_heads=False,
        memory_config=out_mc,
        output_tensors=[q_alloc, k_dummy, v_dummy],
    )
    ttnn.deallocate(k_dummy, force=False)
    ttnn.deallocate(v_dummy, force=False)

    if batch > 1:
        q_out = ttnn.reshape(q_out, (batch, num_heads, seq_len, head_dim))

    return q_out
