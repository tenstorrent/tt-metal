# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Indexed (block-cyclic, SP-sharded) RoPE tables for Gemma-4.

Two tables: sliding layers (theta 1e4, D=256, all 128 pairs rotate) and full layers (theta 1e6,
D=512, "proportional" rope: only the first 64 of 256 pairs rotate; the rest get cos=1, sin=0).
Meta interleaved convention (``[c0, c0, c1, c1, ...]``) to match ``rotary_embedding_indexed`` and the
reverse-permuted q/k projections + q/k norm weights (see attention/weights.py).
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
from models.demos.gemma4_26b_d_p.reference.config import RopeSpec
from models.tt_transformers.tt.common import get_rot_transformation_mat


def meta_cos_sin(seq_len: int, spec: RopeSpec):
    j = torch.arange(0, 2 * spec.rotated_pairs, 2, dtype=torch.float32)
    inv = 1.0 / (spec.theta ** (j / spec.head_dim))
    inv = torch.cat([inv, torch.zeros(spec.head_dim // 2 - spec.rotated_pairs)])
    freqs = torch.outer(torch.arange(seq_len).float(), inv)
    cos = torch.stack([freqs.cos(), freqs.cos()], -1).flatten(-2)[None, None]
    sin = torch.stack([freqs.sin(), freqs.sin()], -1).flatten(-2)[None, None]
    return cos, sin


def build_indexed_rope(mesh_device, spec: RopeSpec, *, max_seq_len, chunk_size, sp_axis=0, dtype=ttnn.bfloat16):
    sp = mesh_device.shape[sp_axis]
    assert chunk_size % (ttnn.TILE_SIZE * sp) == 0 and max_seq_len % chunk_size == 0
    cos, sin = meta_cos_sin(max_seq_len, spec)
    cos = block_cyclic_reorder(cos, chunk_size // sp, sp, seq_dim=2)
    sin = block_cyclic_reorder(sin, chunk_size // sp, sp, seq_dim=2)
    dims = [None, None]
    dims[sp_axis] = 2
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims))
    to_dev = lambda t: ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
    return [to_dev(cos), to_dev(sin)]


def build_transformation_mat(mesh_device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        get_rot_transformation_mat(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
