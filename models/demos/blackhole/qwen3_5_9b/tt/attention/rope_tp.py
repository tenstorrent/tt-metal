# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Partial-RoPE helpers for the tensor-parallel attention path.

Ported from models/demos/qwen35_27b/tt/rope.py. Only the rotary portion
(rope_dim, e.g. 64 of 256) is rotated; the rest passes through. cos/sin are in
HuggingFace split-halves format. These operate on per-device head shards, so
they are unchanged by TP (each device rotates its local heads).
"""
import torch

import ttnn


def build_rope_tables(device, rope_dim, max_seq_len, theta):
    """Precompute replicated cos/sin tables [1, max_seq_len, rope_dim] (HF split-halves)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, rope_dim]
    cos = ttnn.from_torch(
        emb.cos().unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    sin = ttnn.from_torch(
        emb.sin().unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    return cos, sin


def rot_mats_decode(device, rope_dim, max_seq_len, theta, positions):
    """Return [cos, sin] each [1, B, 1, rope_dim] for the given per-user positions.

    positions: torch.Tensor [B] of int positions. Built on host (small) then
    replicated to the mesh — matches apply_partial_rope_decode's expected layout.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    pos = positions.float()
    freqs = torch.outer(pos, inv_freq)  # [B, rope_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [B, rope_dim]
    B = positions.shape[0]
    cos = emb.cos().reshape(1, B, 1, rope_dim).to(torch.bfloat16)
    sin = emb.sin().reshape(1, B, 1, rope_dim).to(torch.bfloat16)
    cos_tt = ttnn.from_torch(
        cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    sin_tt = ttnn.from_torch(
        sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    return cos_tt, sin_tt


def rot_mats_prefill(device, rope_dim, seq_len, theta):
    """Return [cos, sin] each [1, 1, seq_len, rope_dim] for positions 0..seq_len-1."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    emb = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)  # [seq_len, rope_dim]
    cos = ttnn.from_torch(
        emb.cos().reshape(1, 1, seq_len, rope_dim).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    sin = ttnn.from_torch(
        emb.sin().reshape(1, 1, seq_len, rope_dim).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    return cos, sin


def apply_partial_rope_decode(x, cos_tt, sin_tt, n_heads, batch_size, rope_dim):
    """x: [1, B, n_heads, HD]; cos/sin: [1, B, 1, rope_dim]; rotates first rope_dim dims."""
    hd = x.shape[-1]
    B = batch_size
    x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, B, n_heads, rope_dim))
    x_pass = ttnn.slice(x, (0, 0, 0, rope_dim), (1, B, n_heads, hd))
    r1 = ttnn.slice(x_rope, (0, 0, 0, 0), (1, B, n_heads, rope_dim // 2))
    r2 = ttnn.slice(x_rope, (0, 0, 0, rope_dim // 2), (1, B, n_heads, rope_dim))
    x_rot = ttnn.concat([ttnn.neg(r2), r1], dim=-1)
    ttnn.deallocate(r1)
    ttnn.deallocate(r2)
    roped = ttnn.add(ttnn.multiply(x_rope, cos_tt), ttnn.multiply(x_rot, sin_tt))
    ttnn.deallocate(x_rope)
    ttnn.deallocate(x_rot)
    roped = ttnn.to_memory_config(roped, ttnn.DRAM_MEMORY_CONFIG)
    x_pass = ttnn.to_memory_config(x_pass, ttnn.DRAM_MEMORY_CONFIG)
    result = ttnn.concat([roped, x_pass], dim=-1)
    ttnn.deallocate(roped)
    ttnn.deallocate(x_pass)
    return result


def apply_partial_rope_prefill(x, cos_tt, sin_tt, n_heads, rope_dim):
    """x: [1, n_heads, seq_len, HD]; cos/sin: [1, 1, seq_len, rope_dim]."""
    hd = x.shape[-1]
    seq_len = x.shape[-2]
    x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, n_heads, seq_len, rope_dim))
    x_pass = ttnn.slice(x, (0, 0, 0, rope_dim), (1, n_heads, seq_len, hd))
    r1 = ttnn.slice(x_rope, (0, 0, 0, 0), (1, n_heads, seq_len, rope_dim // 2))
    r2 = ttnn.slice(x_rope, (0, 0, 0, rope_dim // 2), (1, n_heads, seq_len, rope_dim))
    x_rot = ttnn.concat([ttnn.neg(r2), r1], dim=-1)
    ttnn.deallocate(r1)
    ttnn.deallocate(r2)
    roped = ttnn.add(ttnn.multiply(x_rope, cos_tt), ttnn.multiply(x_rot, sin_tt))
    ttnn.deallocate(x_rope)
    ttnn.deallocate(x_rot)
    roped = ttnn.to_memory_config(roped, ttnn.DRAM_MEMORY_CONFIG)
    x_pass = ttnn.to_memory_config(x_pass, ttnn.DRAM_MEMORY_CONFIG)
    result = ttnn.concat([roped, x_pass], dim=-1)
    ttnn.deallocate(roped)
    ttnn.deallocate(x_pass)
    return result
