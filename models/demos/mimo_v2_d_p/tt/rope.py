# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Indexed (block-cyclic, SP-sharded) partial-RoPE tables for MiMo-V2.

HF rotates the first ``rope_dim`` (=64) of each 192-dim qk head in the NeoX ``rotate_half`` convention.
``rotary_embedding_indexed`` implements the Meta interleaved convention, so the q/k projection rows of the
rope sub-block are permuted per head (``rope_perm``: new[2j] = old[j], new[2j+1] = old[j + rope_dim/2]);
dot products are invariant. The tables are only ``rope_dim`` wide: ``rotary_embedding_indexed(rotary_dim=rope_dim)``
rotates the first 2 tiles of each 6-tile head and copies the rest.
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
from models.demos.mimo_v2_d_p.reference.config import AttnSpec
from models.tt_transformers.tt.common import get_rot_transformation_mat


def rope_perm(head_dim: int, rope_dim: int) -> torch.Tensor:
    half = rope_dim // 2
    p = torch.arange(head_dim)
    p[:rope_dim] = torch.stack([torch.arange(half), torch.arange(half) + half], -1).flatten()
    return p


def permute_heads(w: torch.Tensor, n_heads: int, head_dim: int, rope_dim: int) -> torch.Tensor:
    """Apply ``rope_perm`` to each head's rows of an (out, in) projection weight."""
    p = rope_perm(head_dim, rope_dim)
    return w.view(n_heads, head_dim, -1)[:, p].reshape(n_heads * head_dim, -1)


def meta_cos_sin(seq_len: int, spec: AttnSpec, width: int | None = None):
    """Meta-interleaved cos/sin [1, 1, S, width]; width defaults to rope_dim (the op then rotates only the
    first rope_dim dims of each head, in place). Dims past rope_dim get cos=1, sin=0."""
    width = width or spec.rope_dim
    inv = 1.0 / (spec.rope_theta ** (torch.arange(0, spec.rope_dim, 2, dtype=torch.int64).float() / spec.rope_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv)  # [S, rope_dim/2]
    cos = torch.ones(seq_len, width)
    sin = torch.zeros(seq_len, width)
    cos[:, : spec.rope_dim] = torch.stack([freqs.cos(), freqs.cos()], -1).flatten(-2)
    sin[:, : spec.rope_dim] = torch.stack([freqs.sin(), freqs.sin()], -1).flatten(-2)
    return cos[None, None], sin[None, None]


def build_indexed_rope(mesh_device, spec: AttnSpec, *, max_seq_len, chunk_size, sp_axis=0, dtype=ttnn.bfloat16):
    sp = mesh_device.shape[sp_axis]
    assert chunk_size % (ttnn.TILE_SIZE * sp) == 0 and max_seq_len % chunk_size == 0
    cos, sin = meta_cos_sin(max_seq_len, spec)
    cos = block_cyclic_reorder(cos, chunk_size // sp, sp, seq_dim=2)
    sin = block_cyclic_reorder(sin, chunk_size // sp, sp, seq_dim=2)
    dims = [None, None]
    dims[sp_axis] = 2
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims))
    to_dev = lambda t: ttnn.from_torch(
        t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
    )
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
