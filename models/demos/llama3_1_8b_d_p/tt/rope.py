# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B indexed RoPE setup for chunked prefill.

Structure copied from ``gpt_oss_d_p/tt/rope.py``: build the whole-cache, block-cyclic, SP-sharded
cos/sin ONCE (reused for every chunk), and let
``ttnn.experimental.deepseek_prefill.rotary_embedding_indexed`` derive each chunk's per-chip start
row on-device from a single ``kv_actual_global`` runtime arg — the same block-cyclic ``update_idxt``
math the KV-cache writer uses, so no per-chunk host reshard.

**The frequency math is NOT copied.** The donor computes YaRN (beta_fast / beta_slow / mscale);
Llama 3.1 uses *llama3* smooth-ramp scaling (low_freq_factor / high_freq_factor, no mscale). The two
produce different cos/sin from the same-looking `factor`, and the error grows with position — it is
invisible at short seq and collapses long-context K PCC. The frequencies come from
``reference/model.py::llama3_inv_freq``, which ``tests/torch/test_llama_reference.py`` pins against
``transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["llama3"]``.

Llama 3.1 RoPE is FULL rotary (rotary_dim == head_dim == 128). The cos/sin are built in the Meta
interleaved convention (``[c0, c0, c1, c1, ...]``) — the convention ``rotary_embedding_indexed`` +
``get_rot_transformation_mat`` expect, matching ``convert_hf_qkv_to_meta_format``-swizzled q/k
projections (see ``tt/model_config.py``).
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
from models.demos.llama3_1_8b_d_p.reference.config import Llama31_8BConfig, LlamaConfig
from models.demos.llama3_1_8b_d_p.reference.model import build_cos_sin_meta
from models.tt_transformers.tt.common import get_rot_transformation_mat

DEFAULT_ROPE_THETA = Llama31_8BConfig.ROPE_THETA
DEFAULT_ROPE_FACTOR = Llama31_8BConfig.ROPE_FACTOR
DEFAULT_LOW_FREQ_FACTOR = Llama31_8BConfig.ROPE_LOW_FREQ_FACTOR
DEFAULT_HIGH_FREQ_FACTOR = Llama31_8BConfig.ROPE_HIGH_FREQ_FACTOR
DEFAULT_ORIG_MAX_POS = Llama31_8BConfig.ROPE_ORIGINAL_MAX_POSITION


def _rope_config(
    head_dim,
    rope_theta=DEFAULT_ROPE_THETA,
    factor=DEFAULT_ROPE_FACTOR,
    low_freq_factor=DEFAULT_LOW_FREQ_FACTOR,
    high_freq_factor=DEFAULT_HIGH_FREQ_FACTOR,
    orig_max_pos=DEFAULT_ORIG_MAX_POS,
) -> LlamaConfig:
    """A LlamaConfig carrying only what the frequency math reads (head_dim + rope_scaling)."""
    return LlamaConfig(
        head_dim=head_dim,
        rope_theta=rope_theta,
        rope_scaling={
            "rope_type": "llama3",
            "factor": factor,
            "low_freq_factor": low_freq_factor,
            "high_freq_factor": high_freq_factor,
            "original_max_position_embeddings": orig_max_pos,
        },
    )


def build_llama3_cos_sin(seq_len, head_dim, **kw):
    """Meta interleaved cos/sin ``[1, 1, seq_len, head_dim]`` with llama3 scaling applied.

    Thin wrapper over the reference so the device path and the oracle cannot compute different
    frequencies — there is exactly one implementation of llama3 scaling in this package.
    """
    return build_cos_sin_meta(seq_len, _rope_config(head_dim, **kw))


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


def build_chunk_rope(mesh_device, *, head_dim, seq_len, dtype=ttnn.bfloat16, **kw):
    """Per-chunk, REPLICATED cos/sin ``[1, 1, seq_len, head_dim]`` for ``rotary_embedding_llama``.

    The non-indexed path: used by the single-shot unit tests and any caller that already holds the
    positions for exactly this chunk. Chunked prefill uses :func:`build_indexed_rope` instead.
    """
    cos, sin = build_llama3_cos_sin(seq_len, head_dim, **kw)

    def _to_dev(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return [_to_dev(cos), _to_dev(sin)]


def build_indexed_rope(
    mesh_device,
    *,
    head_dim,
    max_seq_len,
    chunk_size,
    sp_axis=0,
    dtype=ttnn.bfloat16,
    **kw,
):
    """Build the whole-cache, block-cyclic, SP-sharded cos/sin for the INDEXED on-device RoPE, ONCE.

    The cos/sin cover EVERY cache position (up to ``max_seq_len``), block-cyclic-reordered keyed by
    the per-chip chunk (``chunk_size // sp``) then SP-sharded on ``sp_axis``, so device ``c``'s
    contiguous shard holds — in local-cache-row order — the rope for every global position it will
    carry. ``rotary_embedding_indexed`` then picks this chunk's rows on-device from
    ``kv_actual_global``.

    Constraints (mirroring the block-cyclic / KV cache layout — these must agree with
    ``attention/kv_cache.allocate_kv_cache`` or the rope rows and the cache rows describe different
    tokens, with no error):
      * ``chunk_size % (TILE_SIZE * sp) == 0``
      * ``max_seq_len % chunk_size == 0``

    Returns ``[cos_tt, sin_tt]`` (persistent — reused across all chunks; do NOT deallocate per chunk).
    """
    sp = mesh_device.shape[sp_axis]
    assert (
        chunk_size % (ttnn.TILE_SIZE * sp) == 0
    ), f"chunk_size ({chunk_size}) must be a multiple of TILE_SIZE * sp ({ttnn.TILE_SIZE * sp})"
    assert max_seq_len % chunk_size == 0, f"max_seq_len ({max_seq_len}) must be a multiple of chunk_size ({chunk_size})"
    chunk_local = chunk_size // sp

    cos, sin = build_llama3_cos_sin(max_seq_len, head_dim, **kw)
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
