# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 device-side RoPE builders.

The YaRN table math lives in :mod:`rope_tables` (pure torch, no ttnn) and is re-exported here, so
the host-only conformance test can run without a TT runtime. See that module for why ``truncate``,
``mscale``/``mscale_all_dim`` and ``beta_fast`` matter.
"""

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
from models.tt_transformers.tt.common import get_rot_transformation_mat

from .rope_tables import (  # noqa: F401
    DEFAULT_ROPE_THETA,
    DEFAULT_YARN_BETA_FAST,
    DEFAULT_YARN_BETA_SLOW,
    DEFAULT_YARN_FACTOR,
    DEFAULT_YARN_MSCALE,
    DEFAULT_YARN_MSCALE_ALL_DIM,
    DEFAULT_YARN_ORIG_MAX_POS,
    DEFAULT_YARN_TRUNCATE,
    build_hf_cos_sin,
    build_yarn_cos_sin,
    yarn_inv_freq,
    yarn_params_from_config,
)


def build_transformation_mat(mesh_device, dtype=ttnn.bfloat16):
    """Replicated RoPE transformation matrix for rotary_embedding_llama / rotary_embedding_indexed."""
    return ttnn.from_torch(
        get_rot_transformation_mat(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def build_indexed_rope(
    mesh_device, *, head_dim, max_seq_len, chunk_size, sp_axis=0, dtype=ttnn.bfloat16, **yarn_kwargs
):
    """Whole-cache, block-cyclic, SP-sharded cos/sin for the on-device INDEXED RoPE, built ONCE.

    ``rotary_embedding_indexed`` picks this chunk's rows on-device from ``kv_actual_global`` + the
    device's SP mesh coordinate, so there is no per-chunk host reshard.

    Constraints (must agree with the KV-cache layout):
      * ``chunk_size % (TILE_SIZE * sp) == 0``
      * ``max_seq_len % chunk_size == 0``
    """
    sp = mesh_device.shape[sp_axis]
    assert (
        chunk_size % (ttnn.TILE_SIZE * sp) == 0
    ), f"chunk_size ({chunk_size}) must be a multiple of TILE_SIZE * sp ({ttnn.TILE_SIZE * sp})"
    assert max_seq_len % chunk_size == 0, f"max_seq_len ({max_seq_len}) must be a multiple of chunk_size ({chunk_size})"
    chunk_local = chunk_size // sp

    cos, sin = build_yarn_cos_sin(max_seq_len, head_dim, **yarn_kwargs)
    cos = block_cyclic_reorder(cos, chunk_local, sp, seq_dim=2)
    sin = block_cyclic_reorder(sin, chunk_local, sp, seq_dim=2)

    shard_dims = [None, None]
    shard_dims[sp_axis] = 2  # SP-shard the seq dim; replicate across TP
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(shard_dims))

    def _to_dev(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    return [_to_dev(cos), _to_dev(sin)]
