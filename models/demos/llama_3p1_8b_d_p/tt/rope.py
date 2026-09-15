# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B indexed RoPE for chunked prefill (tt-blaze#4145).

Two independent things have to be right here, and getting either wrong produces a failure that
every byte-level migration gate passes:

**1. Frequency scaling.** Llama-3.1 uses ``rope_type: "llama3"`` — theta 500000, factor 8.0,
low_freq_factor 1.0, high_freq_factor 4.0, original_max_position_embeddings 8192. The
DeepSeek / Kimi / gpt-oss prefill lineage this package borrows from implements **YaRN**, which is a
different function of position; ``gpt_oss_d_p/tt/rope.py`` is the structural template for this file
but its ``yarn_inv_freq`` must not be inherited. RoPE is the identity at position 0, so a wrong
frequency curve is invisible at the start of a sequence and compounds as position advances.

**2. Interleave convention.** RoPE rotates pairs of elements inside each 128-dim head, and there are
two conventions for which elements pair up:

  * Meta / interleaved — adjacent pairs ``(x0,x1), (x2,x3), ...``
  * HF / half-split — ``(x0,x64), (x1,x65), ...`` (``rotate_half``)

Same weights, different resulting tensor. **blaze decode writes K in the Meta-interleaved frame**
(``blaze/ops/rope/kernels/op.hpp``: "for each interleaved (Meta) pair"; ``blaze/ops/rope/op.py``'s
golden is ``rotate_half_meta_style``; ``blaze/weights/llama31_8b/provider.py:make_cos_sin`` builds
the table as ``stack((cos,cos),-1).flatten(-2)``, i.e. each frequency duplicated *adjacently* rather
than ``cat((cos,cos),-1)``). Prefill therefore emits the same frame: Meta-interleaved rotation with
adjacently-duplicated tables, against q/k projections un-permuted by
``convert_hf_qkv_to_meta_format`` (HF ships them permuted precisely so its ``rotate_half``
reproduces Meta's rotation). ``rotary_embedding_indexed`` + ``get_rot_transformation_mat`` are
interleaved-only and expect exactly this.

If the frame is wrong, KV migration copies bytes faithfully and decode reads a permutation: the
``dst-bytes`` gate passes, and per-layer KV PCC passes too whenever the golden was generated with
prefill's own convention instead of decode's. Hence the unit tests grade against HuggingFace and
against a local restatement of blaze's decode golden, never against this module's own output.

**Table width.** Frequencies are a property of ``head_dim`` (64 frequencies over 128 dims) and are
identical for every head, so the per-head table is *tiled* across heads. Building one table of
width ``n_heads * head_dim`` instead spreads ``n_heads * head_dim / 2`` distinct frequencies across
the projection, so head 0's frequencies decay ``n_heads`` times too slowly and every later head gets
a different set — see the warning in ``provider.py:make_cos_sin_tiled``. That error is also zero at
position 0, and invisible to device-vs-golden comparisons because both sides share the wrong table.
Everything here is built at ``head_dim`` width and asserted to be.
"""

import math

import torch

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.tt_transformers.tt.common import get_rot_transformation_mat


def block_cyclic_reorder(matrix: torch.Tensor, chunk_local: int, sp_factor: int, seq_dim: int = 2) -> torch.Tensor:
    """Reorder a ``[.., seq, ..]`` tensor into block-cyclic order keyed by ``chunk_local``.

    Splits the sequence into blocks of ``chunk_local`` rows and concatenates them so that device
    ``c``'s contiguous shard (after a plain SP shard over ``seq_dim``) holds blocks
    ``c, c+sp, c+2sp, ...`` — the same block-cyclic layout the per-chip KV cache is written in. That
    is what makes the indexed-RoPE op's contiguous, offset read of each device's cos/sin shard land
    on the right global positions, including the boundary chip's older-then-wrap rows.

    Restated from ``deepseek_v3_d_p/tt/mla/utils.py`` rather than imported. It is a dozen lines of
    pure index arithmetic, but importing it puts ``models.demos.deepseek_v3_d_p.tt.mla`` on this
    module's import path, which pulls in **safetensors and transformers** — and this module is on
    the prefill runtime's path, which the H2D producers import. ``test_rope_vs_ref`` grades this
    copy against the DeepSeek original, so the two cannot drift.
    """
    seq_len = matrix.shape[seq_dim]
    if seq_len % chunk_local:
        raise ValueError(f"seq_len {seq_len} must be a multiple of chunk_local {chunk_local}")
    num_blocks = seq_len // chunk_local
    if num_blocks % sp_factor:
        raise ValueError(f"num_blocks {num_blocks} must be a multiple of sp_factor {sp_factor}")
    blocks = list(torch.split(matrix, chunk_local, dim=seq_dim))
    order = [b for c in range(sp_factor) for b in range(c, num_blocks, sp_factor)]
    return torch.cat([blocks[b] for b in order], dim=seq_dim)


def llama3_inv_freq(
    head_dim: int = Llama31_8BConfig.HEAD_DIM,
    theta: float = Llama31_8BConfig.ROPE_THETA,
    factor: float = Llama31_8BConfig.ROPE_SCALING_FACTOR,
    low_freq_factor: float = Llama31_8BConfig.ROPE_LOW_FREQ_FACTOR,
    high_freq_factor: float = Llama31_8BConfig.ROPE_HIGH_FREQ_FACTOR,
    orig_max_pos: int = Llama31_8BConfig.ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
) -> torch.Tensor:
    """Llama-3.1 (``rope_type="llama3"``) inverse frequencies, ``[head_dim // 2]``.

    The HF formula verbatim (``transformers.modeling_rope_utils._compute_llama3_parameters``, and
    the same arithmetic as blaze's ``_apply_llama3_rope_scaling``): wavelengths longer than
    ``orig_max_pos / low_freq_factor`` are divided by ``factor``, wavelengths shorter than
    ``orig_max_pos / high_freq_factor`` are left alone, and the band between is interpolated.

    Unlike YaRN there is no mscale / attention_factor to fold into the tables — HF's llama3 branch
    returns an attention factor of exactly 1.0, and ``test_rope_vs_ref`` pins that against HF so a
    transformers change cannot silently introduce one.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    low_freq_wavelen = orig_max_pos / low_freq_factor
    high_freq_wavelen = orig_max_pos / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (orig_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


def build_llama3_cos_sin(
    seq_len: int,
    head_dim: int = Llama31_8BConfig.HEAD_DIM,
    **freq_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Meta-interleaved cos/sin tables, each ``[1, 1, seq_len, head_dim]``.

    ``[c0, c0, c1, c1, ...]`` — every frequency duplicated *adjacently*, which is what the
    interleaved device op consumes and what ``provider.py:make_cos_sin`` builds for decode. The HF
    convention would be ``cat((cos, cos), -1)`` = ``[c0, c1, ..., c0, c1, ...]``; the two are the
    same numbers in a different order, so neither shape nor magnitude catches a mix-up.

    Built at ``head_dim`` width, never at ``n_heads * head_dim`` — see the module docstring.
    """
    inv_freq = llama3_inv_freq(head_dim=head_dim, **freq_kwargs)
    assert inv_freq.shape == (head_dim // 2,), f"expected {head_dim // 2} frequencies, got {tuple(inv_freq.shape)}"

    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [seq_len, head_dim/2]
    cos = torch.stack([freqs.cos(), freqs.cos()], dim=-1).flatten(-2)[None, None]
    sin = torch.stack([freqs.sin(), freqs.sin()], dim=-1).flatten(-2)[None, None]
    assert cos.shape == (1, 1, seq_len, head_dim)
    return cos, sin


def build_transformation_mat(mesh_device, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
    """Replicated RoPE transformation matrix for ``rotary_embedding_indexed``."""
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
    head_dim: int = Llama31_8BConfig.HEAD_DIM,
    max_seq_len: int,
    chunk_size: int,
    sp_axis: int = 0,
    dtype: ttnn.DataType = ttnn.bfloat16,
    **freq_kwargs,
) -> list[ttnn.Tensor]:
    """Whole-cache, block-cyclic, SP-sharded cos/sin for the indexed on-device RoPE, built ONCE.

    The tables cover every cache position up to ``max_seq_len``, block-cyclic-reordered by the
    per-chip chunk (``chunk_size // sp``) and then SP-sharded on ``sp_axis``, so device ``c``'s
    contiguous shard holds — in local cache-row order — the rope for every global position it will
    ever carry. ``rotary_embedding_indexed`` derives this chunk's start row on-device from the
    single ``kv_actual_global`` runtime argument, using the same block-cyclic walk the KV-cache
    writer uses, so there is no per-chunk host reshard.

    Returns ``[cos_tt, sin_tt]``, which are persistent across every chunk — do not deallocate them
    per chunk.
    """
    sp = mesh_device.shape[sp_axis]
    # Enforced rather than assumed: both constraints are shared with the KV-cache block-cyclic
    # layout, and violating either places rope rows on a different chip than the tokens they rotate.
    if chunk_size % (ttnn.TILE_SIZE * sp):
        raise ValueError(f"chunk_size ({chunk_size}) must be a multiple of TILE_SIZE * sp ({ttnn.TILE_SIZE * sp})")
    if max_seq_len % chunk_size:
        raise ValueError(f"max_seq_len ({max_seq_len}) must be a multiple of chunk_size ({chunk_size})")

    cos, sin = build_llama3_cos_sin(max_seq_len, head_dim=head_dim, **freq_kwargs)
    chunk_local = chunk_size // sp
    cos = block_cyclic_reorder(cos, chunk_local, sp, seq_dim=2)
    sin = block_cyclic_reorder(sin, chunk_local, sp, seq_dim=2)

    shard_dims = [None, None]
    shard_dims[sp_axis] = 2  # SP-shard the sequence dim; replicate across the TP axis
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(shard_dims))

    def to_device(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    return [to_device(cos), to_device(sin)]
