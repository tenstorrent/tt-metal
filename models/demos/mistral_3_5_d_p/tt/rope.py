# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 indexed RoPE setup for chunked prefill.

Adapted from ``gpt_oss_d_p/tt/rope.py`` (donor ``compute.rope``), which is already the right shape:
YaRN + FULL rotary + the whole-cache block-cyclic SP-sharded indexed rope built once, in the Meta
interleaved convention that ``rotary_embedding_indexed`` and ``get_rot_transformation_mat`` expect.
The cos/sin cover EVERY cache position and are reused for every chunk;
``ttnn.experimental.deepseek_prefill.rotary_embedding_indexed`` derives each chunk's per-chip start
row on-device from a single ``kv_actual_global`` runtime arg (the same block-cyclic ``update_idxt``
math the KV-cache writer uses), so there is no per-chunk host reshard.

What changed from the donor, and why:

  * **YaRN constants** come from ``config.text_config.rope_parameters`` (theta 1e6, factor 64,
    beta_fast 4.0, beta_slow 1.0, original_max_position 4096) rather than gpt-oss's module defaults,
    and ``head_dim`` is 128 rather than 64.

  * **``truncate`` is True.** The donor floors NOTHING — gpt-oss's config sets
    ``rope_scaling.truncate = False``, so HF keeps the correction dims as floats, and the donor's
    inline comment warns that truncating collapses long-context K PCC. That comment does **not**
    transfer. HF reads ``rope_parameters.get("truncate", True)``, and Mistral's config carries no
    ``truncate`` key, so upstream DOES floor/ceil. Matching HF is the whole point of the table, so
    this file truncates by default and ``tests/unit/test_reference_config.py`` pins it both ways
    (the float variant must NOT match HF).

  * **mscale.** Mistral's ``rope_parameters`` carries DeepSeek-style ``mscale: 1.0`` /
    ``mscale_all_dim: 0.0``. HF only takes the ratio branch when BOTH are truthy, and 0.0 is falsy,
    so it falls through to ``get_mscale(factor) = 0.1*log(factor) + 1`` — the same expression the
    donor hard-codes. Verified against ``_compute_yarn_parameters`` in the config test.

Alignment constraints shared with the KV cache: ``chunk_size % (TILE_SIZE * sp) == 0`` and the
whole-cache length must be a whole number of chunks (see ``spec.cache_capacity``).
"""

import math

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.tt_transformers.tt.common import get_rot_transformation_mat

# Mistral-Medium-3.5-128B YaRN defaults (configs/Mistral-Medium-3.5-128B/config.json).
DEFAULT_ROPE_THETA = C.ROPE_THETA
DEFAULT_YARN_FACTOR = C.YARN_FACTOR
DEFAULT_YARN_ORIG_MAX_POS = C.YARN_ORIG_MAX_POS
DEFAULT_YARN_BETA_FAST = C.YARN_BETA_FAST
DEFAULT_YARN_BETA_SLOW = C.YARN_BETA_SLOW
DEFAULT_YARN_TRUNCATE = C.YARN_TRUNCATE


def yarn_params_from_config(hf_config) -> dict:
    """Pull the YaRN kwargs out of an HF text config's ``rope_parameters``.

    Reading the config (rather than trusting the module defaults above) is what keeps a config edit
    from silently diverging from the device. ``truncate`` follows HF's own default of True.
    """
    rp = dict(getattr(hf_config, "rope_parameters", None) or {})
    return {
        "rope_theta": rp.get("rope_theta", DEFAULT_ROPE_THETA),
        "yarn_factor": rp.get("factor", DEFAULT_YARN_FACTOR),
        "yarn_orig_max_pos": rp.get("original_max_position_embeddings", DEFAULT_YARN_ORIG_MAX_POS),
        "yarn_beta_fast": rp.get("beta_fast", DEFAULT_YARN_BETA_FAST),
        "yarn_beta_slow": rp.get("beta_slow", DEFAULT_YARN_BETA_SLOW),
        "truncate": rp.get("truncate", DEFAULT_YARN_TRUNCATE),
    }


def yarn_inv_freq(
    head_dim,
    base=DEFAULT_ROPE_THETA,
    factor=DEFAULT_YARN_FACTOR,
    orig_max_pos=DEFAULT_YARN_ORIG_MAX_POS,
    beta_fast=DEFAULT_YARN_BETA_FAST,
    beta_slow=DEFAULT_YARN_BETA_SLOW,
    truncate=DEFAULT_YARN_TRUNCATE,
):
    """YaRN inverse frequencies + attention_factor (mscale). Matches transformers
    ``_compute_yarn_parameters`` for this config exactly (pinned by test_reference_config.py)."""

    def find_correction_dim(num_rotations):
        return (head_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    # Mistral ships no `truncate` key, so HF's default (True) applies and the correction dims are
    # floored/ceiled. Keeping them as floats (the gpt-oss donor's branch) shifts the
    # interpolation<->extrapolation ramp and injects a per-freq inv_freq error (~3.4e-4 here) that
    # grows linearly with position: invisible at short seq, fatal to long-context K PCC.
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low = max(low, 0.0)
    high = min(high, head_dim - 1)

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
    # mscale_all_dim is 0.0 (falsy) in this config, so HF's `mscale and mscale_all_dim` ratio branch
    # is not taken and attention_factor is the plain get_mscale(factor).
    attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
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
    truncate=DEFAULT_YARN_TRUNCATE,
):
    """Meta interleaved cos/sin ``[1, 1, seq_len, head_dim]`` with the YaRN mscale folded in.

    Meta convention (what ``ttnn.experimental.rotary_embedding_llama`` /
    ``rotary_embedding_indexed`` expect, alongside ``reverse_permute``'d q/k weights):
    ``[c0, c0, c1, c1, ...]`` — interleave the per-freq value, rather than HF's concat-halves.
    """
    inv_freq, attn_factor = yarn_inv_freq(
        head_dim, rope_theta, yarn_factor, yarn_orig_max_pos, yarn_beta_fast, yarn_beta_slow, truncate
    )
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, inv_freq)  # [seq_len, head_dim/2]
    cos_half = torch.cos(freqs) * attn_factor
    sin_half = torch.sin(freqs) * attn_factor
    cos_meta = torch.stack([cos_half, cos_half], dim=-1).flatten(-2)[None, None]  # [1,1,seq_len,head_dim]
    sin_meta = torch.stack([sin_half, sin_half], dim=-1).flatten(-2)[None, None]
    return cos_meta, sin_meta


def hf_to_meta_head_permutation(head_dim: int, rotary_dim: int | None = None) -> torch.Tensor:
    """Index tensor mapping an HF half-split head layout onto the Meta interleaved one.

    Used to reconcile a golden K (HF convention) against the device K, which is swizzled because the
    q/k projections went through ``convert_hf_qkv_to_meta_format``. Full rotary here, so it covers
    the whole head.
    """
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    half = rotary_dim // 2
    src = list(range(head_dim))
    for m in range(rotary_dim):
        src[m] = half * (m % 2) + (m // 2)
    return torch.tensor(src, dtype=torch.long)


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
    truncate=DEFAULT_YARN_TRUNCATE,
    dtype=ttnn.bfloat16,
):
    """Build the whole-cache, block-cyclic, SP-sharded cos/sin for the INDEXED on-device RoPE, ONCE.

    The cos/sin cover EVERY cache position (up to ``max_seq_len``, which for the production runtime is
    ``spec.cache_capacity``), block-cyclic-reordered keyed by the per-chip chunk (``chunk_size // sp``)
    then SP-sharded on ``sp_axis``, so device ``c``'s contiguous shard holds — in local-cache-row order
    — the rope for every global position it will carry. ``rotary_embedding_indexed`` then picks this
    chunk's rows on-device from ``kv_actual_global``.

    Constraints (mirroring the block-cyclic / cache layout):
      * ``chunk_size % (TILE_SIZE * sp) == 0``
      * ``max_seq_len % chunk_size == 0`` — the cache must be a whole number of chunks. The spec's
        262144 is 51.2 chunks of 5120, which is why the runtime allocates ``spec.cache_capacity``
        (266240 = 52 chunks) and treats the tail as capacity only.

    Returns ``[cos_tt, sin_tt]`` (persistent — reused across all chunks; do NOT deallocate per chunk).
    Use with :func:`build_transformation_mat` and ``apply_rope(..., kv_actual_global=cached_len,
    cluster_axis=sp_axis)``.
    """
    sp = mesh_device.shape[sp_axis]
    assert (
        chunk_size % (ttnn.TILE_SIZE * sp) == 0
    ), f"chunk_size ({chunk_size}) must be a multiple of TILE_SIZE * sp ({ttnn.TILE_SIZE * sp})"
    assert max_seq_len % chunk_size == 0, (
        f"max_seq_len ({max_seq_len}) must be a multiple of chunk_size ({chunk_size}); pass "
        f"spec.cache_capacity, not spec.max_seq_len"
    )
    chunk_local = chunk_size // sp

    cos, sin = build_yarn_cos_sin(
        max_seq_len,
        head_dim,
        rope_theta=rope_theta,
        yarn_factor=yarn_factor,
        yarn_orig_max_pos=yarn_orig_max_pos,
        yarn_beta_fast=yarn_beta_fast,
        yarn_beta_slow=yarn_beta_slow,
        truncate=truncate,
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
