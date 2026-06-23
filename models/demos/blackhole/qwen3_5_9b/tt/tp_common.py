# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel (multi-device) helpers for Qwen3.5 on Blackhole.

Ported from ``models/demos/qwen35_27b/tt/model_config.py`` (the proven TP=4
implementation of this exact architecture) and generalized so the same helpers
serve both the 9B (single device) and 27B (TP=4) configs. Everything here is
inert on a 1-device mesh — ``model_config.py`` only builds these configs and
calls the sharding helpers when ``num_devices > 1``.

Contents:
- Hardware constants
- Mesh tensor helpers (shard / replicate)
- Weight-prep helpers that reorder HF weights for clean per-device sharding
"""
import torch

import ttnn

# ── Hardware constants ──────────────────────────────────────────────────────
TILE_SIZE = 32


# ── Mesh tensor helpers ─────────────────────────────────────────────────────
def shard_w(torch_tensor, mesh, dim, memory_config, cache_path, dtype=ttnn.bfloat8_b):
    """Convert a torch weight [out, in] to a sharded mesh tensor.

    Transposes to [in, out] (ttnn.linear convention) and shards along ``dim``
    (after the transpose): dim=-1 = column-parallel, dim=0 = row-parallel.
    """
    w = torch_tensor.to(torch.bfloat16).T.contiguous()
    return ttnn.as_tensor(
        w,
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        cache_file_name=cache_path,
    )


def replicate(torch_tensor, mesh, cache_path, dtype=ttnn.bfloat16):
    """Small tensor (norms, biases) -> replicated on every device."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def shard_small(torch_tensor, mesh, cache_path, dim=-1, dtype=ttnn.bfloat16):
    """Small per-head tensor (conv taps, A_log, dt_bias) -> sharded across devices."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


# ── Weight-prep helpers (reorder HF weights for per-device sharding) ─────────
def prepare_gdn_qkv(qkv_w, key_dim, value_dim, nk, dk, nv, dv, tp):
    """Interleave GDN Q/K/V heads so ``shard_w(dim=-1)`` (column-parallel, which
    shards the output dim into ``tp`` equal contiguous chunks after the transpose)
    gives each device a contiguous block of (nk/tp) Q heads, (nk/tp) K heads and
    (nv/tp) V heads.

    qkv_w: [key_dim + key_dim + value_dim, hidden] (fused in_proj_qkv).
    """
    q_part = qkv_w[:key_dim, :]
    k_part = qkv_w[key_dim : 2 * key_dim, :]
    v_part = qkv_w[2 * key_dim :, :]

    q_per = nk // tp
    v_per = nv // tp
    shards = []
    for s in range(tp):
        q_s = q_part[s * q_per * dk : (s + 1) * q_per * dk, :]
        k_s = k_part[s * q_per * dk : (s + 1) * q_per * dk, :]
        v_s = v_part[s * v_per * dv : (s + 1) * v_per * dv, :]
        shards.append(torch.cat([q_s, k_s, v_s], dim=0))
    return torch.cat(shards, dim=0)


def prepare_attn_qkv(wq, wk, wv, n_heads, n_kv_heads, head_dim, tp):
    """Reorder the full-attention Q/K/V projection weights into the per-device
    [local_Q | local_K | local_V] grouping that ``nlp_create_qkv_heads`` expects.

    The op splits a fused [.., (n_heads + 2*n_kv_heads)*head_dim] tensor as
    all-Q-heads then all-K-heads then all-V-heads.

    For TP we must hand each device its OWN heads in that order, so we lay the rows out as
    cat_over_devices([q_shard | k_shard | v_shard]); ``shard_w(dim=-1)`` (column-parallel,
    which shards the out dim into ``tp`` equal contiguous chunks after the
    transpose) then gives device ``s`` exactly its [q_s | k_s | v_s] block.

    * wq: [n_heads*head_dim, hidden]   (gate already split out)
    * wk, wv: [n_kv_heads*head_dim, hidden]
    Returns the reordered [(n_heads + 2*n_kv_heads)*head_dim, hidden] weight.
    The attention analog of prepare_gdn_qkv; reduces to a plain cat at tp=1.
    """
    if tp == 1:
        return torch.cat([wq, wk, wv], dim=0)

    q_per = n_heads // tp
    kv_per = max(1, n_kv_heads // tp)
    kv_replicated = n_kv_heads < tp  # each device replicates the GQA-assigned KV head
    shards = []
    for s in range(tp):
        q_s = wq[s * q_per * head_dim : (s + 1) * q_per * head_dim, :]
        if kv_replicated:
            # GQA: device s's Q heads all map to this single KV head (mirrors gemma4).
            kv_idx = (s * q_per) * n_kv_heads // n_heads
            k_s = wk[kv_idx * head_dim : (kv_idx + 1) * head_dim, :]
            v_s = wv[kv_idx * head_dim : (kv_idx + 1) * head_dim, :]
        else:
            k_s = wk[s * kv_per * head_dim : (s + 1) * kv_per * head_dim, :]
            v_s = wv[s * kv_per * head_dim : (s + 1) * kv_per * head_dim, :]
        shards.append(torch.cat([q_s, k_s, v_s], dim=0))
    return torch.cat(shards, dim=0)


def prepare_conv_taps(conv_w, key_dim, nk, dk, nv, dv, kernel_size, tp):
    """Split the fused conv1d weight [qkv_dim, 1, kernel] into ``kernel`` taps,
    each reordered to match the per-device Q/K/V head grouping of prepare_gdn_qkv."""
    cw = conv_w.float()
    q_per = nk // tp
    v_per = nv // tp
    taps = []
    for j in range(kernel_size):
        tap = cw[:, 0, j]
        q_tap = tap[:key_dim]
        k_tap = tap[key_dim : 2 * key_dim]
        v_tap = tap[2 * key_dim :]
        shards = []
        for s in range(tp):
            q_s = q_tap[s * q_per * dk : (s + 1) * q_per * dk]
            k_s = k_tap[s * q_per * dk : (s + 1) * q_per * dk]
            v_s = v_tap[s * v_per * dv : (s + 1) * v_per * dv]
            shards.append(torch.cat([q_s, k_s, v_s]))
        taps.append(torch.cat(shards))
    return taps
