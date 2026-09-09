# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS indexed RoPE setup for chunked prefill.

Builds the whole-cache, block-cyclic, SP-sharded cos/sin ONCE (reused for every chunk), mirroring
``minimax_m3/tt/tt_prefill_runtime._build_indexed_rope`` and DeepSeek's
``RotarySetup.get_rope_tensors_indexed``. ``ttnn.experimental.deepseek_prefill.rotary_embedding_indexed``
then derives each chunk's per-chip start row on-device from a single ``kv_actual_global`` runtime arg
(the same block-cyclic ``update_idxt`` math the KV-cache writer uses), so no per-chunk host reshard.

gpt-oss RoPE is FULL rotary (rotary_dim == head_dim == 64) with YaRN scaling (rope_theta 150000,
factor 32, original_max_position 4096). The cos/sin are built in the Meta interleaved convention
(``[c0, c0, c1, c1, ...]``) with the YaRN mscale/attention_factor folded in — exactly the convention
``rotary_embedding_indexed`` + ``get_rot_transformation_mat`` expect (see the DeepSeek indexed-rope
op test), and matching ``convert_hf_qkv_to_meta_format``-swizzled q/k projections.
"""

import math

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
from models.tt_transformers.tt.common import get_rot_transformation_mat

# GPT-OSS-120B YaRN defaults (configs/gpt-oss-120b/config.json).
DEFAULT_ROPE_THETA = 150000.0
DEFAULT_YARN_FACTOR = 32.0
DEFAULT_YARN_ORIG_MAX_POS = 4096
DEFAULT_YARN_BETA_FAST = 32.0
DEFAULT_YARN_BETA_SLOW = 1.0


def yarn_inv_freq(
    head_dim,
    base=DEFAULT_ROPE_THETA,
    factor=DEFAULT_YARN_FACTOR,
    orig_max_pos=DEFAULT_YARN_ORIG_MAX_POS,
    beta_fast=DEFAULT_YARN_BETA_FAST,
    beta_slow=DEFAULT_YARN_BETA_SLOW,
):
    """YaRN inverse frequencies + attention_factor (mscale). Matches transformers
    ``_compute_yarn_parameters`` (see gpt_oss_d_p/tests/unit/test_attention_vs_ref.py)."""

    def find_correction_dim(num_rotations):
        return (head_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    # gpt-oss sets rope_scaling.truncate=False, so HF keeps the correction dims as FLOATS (no
    # floor/ceil). Truncating shifts the interpolation<->extrapolation ramp and injects a per-freq
    # inv_freq error that grows linearly with position: invisible at short seq (~0.02 rad by pos 100,
    # so the unit test passes) but ~1.1 rad phase drift by pos 5000 -> long-context K PCC collapse
    # (layer-0 K fell to ~0.69 at the tail). Match HF exactly: no floor/ceil (verified inv_freq to 1e-7).
    low = max(find_correction_dim(beta_fast), 0.0)
    high = min(find_correction_dim(beta_slow), head_dim - 1)

    pos_freqs = base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    if low == high:
        high += 0.001
    ramp = ((torch.arange(head_dim // 2).float() - low) / (high - low)).clamp(0, 1)
    inv_freq_extrapolation_factor = 1.0 - ramp

    inv_freq = (
        inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    attention_factor = 0.1 * math.log(factor) + 1.0
    return inv_freq, attention_factor


def build_yarn_cos_sin(
    seq_len,
    head_dim,
    *,
    rope_theta=DEFAULT_ROPE_THETA,
    yarn_factor=DEFAULT_YARN_FACTOR,
    yarn_orig_max_pos=DEFAULT_YARN_ORIG_MAX_POS,
    yarn_beta_fast=DEFAULT_YARN_BETA_FAST,
    yarn_beta_slow=DEFAULT_YARN_BETA_SLOW,
):
    """Meta interleaved cos/sin ``[1, 1, seq_len, head_dim]`` with YaRN mscale folded in.

    Meta convention (matches ttnn.rotary_embedding_llama / rotary_embedding_indexed +
    reverse_permute'd weights): ``[c0, c0, c1, c1, ...]`` (interleave the per-freq value, not the
    concat-halves HF convention).
    """
    inv_freq, attn_factor = yarn_inv_freq(
        head_dim, rope_theta, yarn_factor, yarn_orig_max_pos, yarn_beta_fast, yarn_beta_slow
    )
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, inv_freq)  # [seq_len, head_dim/2]
    cos_half = torch.cos(freqs) * attn_factor
    sin_half = torch.sin(freqs) * attn_factor
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
    yarn_factor=DEFAULT_YARN_FACTOR,
    yarn_orig_max_pos=DEFAULT_YARN_ORIG_MAX_POS,
    yarn_beta_fast=DEFAULT_YARN_BETA_FAST,
    yarn_beta_slow=DEFAULT_YARN_BETA_SLOW,
    dtype=ttnn.bfloat16,
):
    """Build the whole-cache, block-cyclic, SP-sharded cos/sin for the INDEXED on-device RoPE, ONCE.

    The cos/sin cover EVERY cache position (up to ``max_seq_len``), block-cyclic-reordered keyed by the
    per-chip chunk (``chunk_size // sp``) then SP-sharded on ``sp_axis``, so device ``c``'s contiguous
    shard holds — in local-cache-row order — the rope for every global position it will carry.
    ``rotary_embedding_indexed`` then picks this chunk's rows on-device from ``kv_actual_global``.

    Constraints (mirroring the block-cyclic / cache layout, see RotarySetup.get_rope_tensors_indexed):
      * ``chunk_size % (TILE_SIZE * sp) == 0``
      * ``max_seq_len % chunk_size == 0``

    Returns ``[cos_tt, sin_tt]`` (persistent — reused across all chunks; do NOT deallocate per chunk).
    Use with :func:`build_transformation_mat` and ``apply_rope(..., kv_actual_global=cached_len,
    cluster_axis=sp_axis)``.
    """
    sp = mesh_device.shape[sp_axis]
    assert (
        chunk_size % (ttnn.TILE_SIZE * sp) == 0
    ), f"chunk_size ({chunk_size}) must be a multiple of TILE_SIZE * sp ({ttnn.TILE_SIZE * sp})"
    assert max_seq_len % chunk_size == 0, f"max_seq_len ({max_seq_len}) must be a multiple of chunk_size ({chunk_size})"
    chunk_local = chunk_size // sp

    cos, sin = build_yarn_cos_sin(
        max_seq_len,
        head_dim,
        rope_theta=rope_theta,
        yarn_factor=yarn_factor,
        yarn_orig_max_pos=yarn_orig_max_pos,
        yarn_beta_fast=yarn_beta_fast,
        yarn_beta_slow=yarn_beta_slow,
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
