# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared pieces of the ring_joint cache-read tests on (8,4): the GQA shapes, the block-cyclic chunk builders and
gather, the dense SDPA configs, and the 1-element uint32 scalars the metadata path reads on-device."""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions

NQ, NKV, HEAD_DIM = 64, 4, 128
# SP over mesh rows, TP over mesh cols: the sequence (dim 2) shards over the SP axis, heads (dim 1) over the other.
SP_AXIS = 0
SHARD_DIMS = (2, 1) if SP_AXIS == 0 else (1, 2)
# The host-int path is held to this vs the torch golden (K/V live in the bf8 cache); the metadata path to torch.equal.
PCC_BF8_CACHE = 0.99


def torch_gqa_causal(q, k, v):
    rep = NQ // NKV
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = q.shape[2]
    scores = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    causal = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
    return torch.softmax(scores + causal, dim=-1) @ v  # [1, NQ, S, HD]


def bc_index(kv_actual, sp, chunk_local):
    """Global positions of the chunk starting at kv_actual, in the chip-major block-cyclic order the SP shards hold."""
    pos = rotated_chip_positions(kv_actual, sp, chunk_local)
    return torch.tensor([pos[c][r] for c in range(sp) for r in range(chunk_local)], dtype=torch.long)


def _shard(t, mesh_device, dtype, on_device=True):
    rows, cols = tuple(mesh_device.shape)
    placement = dict(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) if on_device else {}
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=SHARD_DIMS),
        **placement,
    )


def make_kv_chunk(src, kv_actual, mesh_device, chunk_local):
    """src [heads, S, HD] host K or V -> this chunk, block-cyclic, sharded, in the cache's bf8 dtype."""
    sp = mesh_device.shape[0]
    chunk = src[:, bc_index(kv_actual, sp, chunk_local), :].reshape(1, src.shape[0], sp * chunk_local, HEAD_DIM)
    return _shard(chunk, mesh_device, ttnn.bfloat8_b)


def make_q_chunk(q, kv_actual, mesh_device, chunk_local, on_device=True):
    """q [1, NQ, S, HD] host -> the chunk's queries, block-cyclic, sharded. on_device=False yields the host-side
    twin for copy_host_to_device_tensor (re-targeting a traced Q slab in place)."""
    idx = bc_index(kv_actual, mesh_device.shape[0], chunk_local)
    return _shard(q[:, :, idx, :], mesh_device, ttnn.bfloat16, on_device)


def sdpa_configs(mesh_device):
    """The M3 dense SDPA program / compute configs (minimax3_gqa_causal_perf in
    tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py)."""
    grid = mesh_device.compute_with_storage_grid_size()
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=128,
        k_chunk_size=512,
        exp_approx_mode=False,
    )
    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    return prog, kcfg


def gather_chunk(out, kv_actual, mesh_device, chunk_local):
    """Per-chip [1, NQ/tp, chunk_local, HD] block-cyclic over the chunk at kv_actual -> [1, NQ, chunk_global, HD] in
    natural order: one composed host read (rows -> seq, cols -> heads), then undo the block-cyclic permutation."""
    rows, cols = tuple(mesh_device.shape)
    chunk_global = rows * chunk_local
    full_bc = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=SHARD_DIMS)
    ).float()
    inv = torch.empty(chunk_global, dtype=torch.long)
    inv[bc_index(kv_actual, rows, chunk_local) - kv_actual] = torch.arange(chunk_global)
    return full_bc[:, :, inv, :]


def meta_scalar(val, mesh_device):
    """1-element uint32 replicated-DRAM scalar, the form update_padded_kv_cache and ring_joint read element [0] of."""
    return ttnn.from_torch(
        torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def host_scalar(val):
    """Host-side twin of meta_scalar, for copy_host_to_device_tensor re-targeting between trace replays."""
    return ttnn.from_torch(
        torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
