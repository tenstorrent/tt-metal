# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Llama 3.1 8B indexed RoPE setup for chunked prefill.

Builds the whole-cache, block-cyclic, SP-sharded cos/sin ONCE (reused for every chunk). Uses the
Llama3 frequency scaling (not YaRN). ``ttnn.experimental.deepseek_prefill.rotary_embedding_indexed``
then derives each chunk's per-chip start row on-device from a single ``kv_actual_global`` runtime arg.

Llama3 RoPE is FULL rotary (rotary_dim == head_dim == 128) with the llama3 frequency scaling
(rope_theta 500000, factor 8, low_freq_factor 1, high_freq_factor 4, original_max_position 8192).
The cos/sin are built in the Meta interleaved convention (``[c0, c0, c1, c1, ...]``) — exactly the
convention ``rotary_embedding_indexed`` + ``get_rot_transformation_mat`` expect, and matching
``convert_hf_qkv_to_meta_format``-swizzled q/k projections.
"""

import math

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
from models.tt_transformers.tt.common import get_rot_transformation_mat

# Llama 3.1 8B RoPE defaults.
DEFAULT_ROPE_THETA = 500000.0
DEFAULT_ROPE_FACTOR = 8.0
DEFAULT_LOW_FREQ_FACTOR = 1.0
DEFAULT_HIGH_FREQ_FACTOR = 4.0
DEFAULT_ORIG_MAX_POS = 8192


def llama3_inv_freq(
    head_dim,
    base=DEFAULT_ROPE_THETA,
    factor=DEFAULT_ROPE_FACTOR,
    low_freq_factor=DEFAULT_LOW_FREQ_FACTOR,
    high_freq_factor=DEFAULT_HIGH_FREQ_FACTOR,
    orig_max_pos=DEFAULT_ORIG_MAX_POS,
):
    """Llama3 inverse frequencies. Matches transformers ``_compute_llama3_parameters``.

    The llama3 scaling interpolates between the original frequencies (for short wavelengths) and
    frequencies divided by ``factor`` (for long wavelengths), with a smooth transition in between.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    low_freq_wavelen = orig_max_pos / low_freq_factor
    high_freq_wavelen = orig_max_pos / high_freq_factor

    wavelen = 2 * math.pi / inv_freq

    # Long wavelengths (> low_freq_wavelen): scale down by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

    # Smooth interpolation for medium wavelengths
    smooth_factor = (orig_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smooth_factor = smooth_factor.clamp(0, 1)

    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    smoothed = (1 - smooth_factor) * inv_freq / factor + smooth_factor * inv_freq
    inv_freq_llama = torch.where(is_medium, smoothed, inv_freq_llama)

    return inv_freq_llama


def build_llama3_cos_sin(
    seq_len,
    head_dim,
    *,
    rope_theta=DEFAULT_ROPE_THETA,
    factor=DEFAULT_ROPE_FACTOR,
    low_freq_factor=DEFAULT_LOW_FREQ_FACTOR,
    high_freq_factor=DEFAULT_HIGH_FREQ_FACTOR,
    orig_max_pos=DEFAULT_ORIG_MAX_POS,
):
    """Meta interleaved cos/sin ``[1, 1, seq_len, head_dim]`` with Llama3 frequency scaling.

    Meta convention (matches ttnn.rotary_embedding_llama / rotary_embedding_indexed +
    reverse_permute'd weights): ``[c0, c0, c1, c1, ...]`` (interleave the per-freq value, not the
    concat-halves HF convention).
    """
    inv_freq = llama3_inv_freq(head_dim, rope_theta, factor, low_freq_factor, high_freq_factor, orig_max_pos)
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, inv_freq)  # [seq_len, head_dim/2]
    cos_half = torch.cos(freqs)
    sin_half = torch.sin(freqs)
    cos_meta = torch.stack([cos_half, cos_half], dim=-1).flatten(-2)[None, None]  # [1,1,seq_len,head_dim]
    sin_meta = torch.stack([sin_half, sin_half], dim=-1).flatten(-2)[None, None]
    return cos_meta, sin_meta


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
    mesh_device,
    *,
    head_dim,
    max_seq_len,
    chunk_size,
    sp_axis=0,
    rope_theta=DEFAULT_ROPE_THETA,
    factor=DEFAULT_ROPE_FACTOR,
    low_freq_factor=DEFAULT_LOW_FREQ_FACTOR,
    high_freq_factor=DEFAULT_HIGH_FREQ_FACTOR,
    orig_max_pos=DEFAULT_ORIG_MAX_POS,
    dtype=ttnn.bfloat16,
):
    """Build the whole-cache, block-cyclic, SP-sharded cos/sin for the INDEXED on-device RoPE, ONCE.

    The cos/sin cover EVERY cache position (up to ``max_seq_len``), block-cyclic-reordered keyed by the
    per-chip chunk (``chunk_size // sp``) then SP-sharded on ``sp_axis``, so device ``c``'s contiguous
    shard holds — in local-cache-row order — the rope for every global position it will carry.
    ``rotary_embedding_indexed`` then picks this chunk's rows on-device from ``kv_actual_global``.

    Constraints:
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

    cos, sin = build_llama3_cos_sin(
        max_seq_len,
        head_dim,
        rope_theta=rope_theta,
        factor=factor,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        orig_max_pos=orig_max_pos,
    )
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
