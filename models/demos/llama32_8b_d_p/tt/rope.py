# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3-scaled RoPE tables for prefill.

HF anchor: ``transformers.models.llama.modeling_llama.LlamaRotaryEmbedding`` +
``apply_rotary_pos_emb`` (with ``rope_type="llama3"``,
``transformers.modeling_rope_utils._compute_llama3_parameters``).
Template: ``models/demos/gpt_oss_d_p/tt/rope.py:103`` (``build_transformation_mat``, copied
verbatim) and ``:115`` (``build_indexed_rope``, mirrored) — with gpt-oss's YaRN frequency builder
(``:75`` ``build_yarn_cos_sin``) replaced by the llama3 pair
``models/tt_transformers/tt/common.py:489`` ``precompute_freqs`` +  ``:525`` ``gather_cos_sin``.

**Assembly of imported helpers; no new RoPE math here** (``DEC-007``). Llama is FULL rotary
(rotary_dim == head_dim == 128), θ = 500000.0, llama3 scaling factor 8.0 over an original context
of 8192 (``00_MODEL_CARD.md`` §2).

**Convention: Meta / interleaved.** ``ttnn.experimental.rotary_embedding_llama`` (and its indexed
sibling ``ttnn.experimental.deepseek_prefill.rotary_embedding_indexed``) expect
``[c0, c0, c1, c1, ...]`` — the per-frequency value *interleaved*, which is what
``gather_cos_sin`` produces (``common.py:529`` ``torch.stack([cos, cos], -1).flatten(-2)``). HF
instead *concatenates* halves and rotates with ``rotate_half``. Both encode the same
``[S, head_dim/2]`` frequency table; mixing the two expansions is the classic RoPE bug and is what
produces "attention PCC 0.5-0.9, norms fine". The Meta convention additionally requires the Q/K
**projection weights** to be ``reverse_permute``d at load
(``models/tt_transformers/tt/load_checkpoints.py:451`` ``convert_hf_qkv_to_meta_format``, ``:891``
``reverse_permute``) — P5.5's job, not this file's. The alternative that removes the permute is
``ttnn.experimental.rotary_embedding_hf``; ``DEC-007`` records why the Meta path is taken (both
prefill templates use it, so the surrounding scaffolding already assumes it).

``get_rot_transformation_mat`` is called with **no argument**: ``common.py:564`` re-assigns
``dhead = 32`` and ignores what it was passed (``R-010``).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
from models.tt_transformers.tt.common import (
    gather_cos_sin,
    get_prefill_rot_mat,
    get_rot_transformation_mat,
    precompute_freqs,
)

# `models/tt_transformers/tt/common.py:405` compute_llama3_parameters hard-codes the two limb
# factors as LOCAL constants (`:407` low = 1, `:408` high = 4) rather than reading them from the
# config, so a checkpoint that changed either would be silently ignored. llama_hf_config() asserts
# them at construction; _assert_llama3_scaling re-asserts here, from the object, so this module
# cannot be handed a config the delegate would misinterpret. R-006 / DEC-007.
_HARDCODED_LOW_FREQ_FACTOR = 1.0
_HARDCODED_HIGH_FREQ_FACTOR = 4.0
_SUPPORTED_ROPE_TYPE = "llama3"


def _assert_llama3_scaling(hf_config) -> None:
    assert (
        hf_config.rope_type == _SUPPORTED_ROPE_TYPE
    ), f"tt/rope.py only builds llama3-scaled RoPE; got rope_type={hf_config.rope_type!r}"
    assert (
        hf_config.rope_low_freq_factor == _HARDCODED_LOW_FREQ_FACTOR
    ), f"low_freq_factor={hf_config.rope_low_freq_factor}, but models/tt_transformers/tt/common.py:407 hard-codes {_HARDCODED_LOW_FREQ_FACTOR} and would silently ignore it"
    assert (
        hf_config.rope_high_freq_factor == _HARDCODED_HIGH_FREQ_FACTOR
    ), f"high_freq_factor={hf_config.rope_high_freq_factor}, but models/tt_transformers/tt/common.py:408 hard-codes {_HARDCODED_HIGH_FREQ_FACTOR} and would silently ignore it"


def build_transformation_mat(mesh_device, dtype=ttnn.bfloat16):
    """Replicated ``[1, 1, 32, 32]`` RoPE transformation matrix.

    Consumed by ``rotary_embedding_llama`` / ``rotary_embedding_indexed``. Copied verbatim from
    ``models/demos/gpt_oss_d_p/tt/rope.py:103``; single-tile and model-independent (the op works a
    tile at a time, hence 32 and not ``head_dim``).
    """
    return ttnn.from_torch(
        get_rot_transformation_mat(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def build_prefill_rope(mesh_device, hf_config, *, seq_len, start_pos=0):
    """Per-chunk, **replicated** Meta cos/sin for positions ``[start_pos, start_pos + seq_len)``.

    The single-shot / non-cache prefill path (P5-P6): the tables are already sliced to this chunk's
    positions, so ``rotary_embedding_llama`` needs no index arithmetic. Delegates to
    ``models/tt_transformers/tt/common.py:534`` ``get_prefill_rot_mat``, which is
    ``precompute_freqs`` (llama3-scaled) -> ``gather_cos_sin`` (Meta interleave) -> two replicated
    ``ttnn.from_torch`` calls.

    Returns ``[cos_tt, sin_tt]``, each ``[1, 1, seq_len, head_dim]`` bfloat16 TILE, replicated.

    **`start_pos` is bounded by `seq_len`** — measured, not assumed (``DEC-029``). The delegate
    precomputes a frequency table of exactly ``seq_len * 2`` rows (``common.py:536``) and then
    gathers positions ``[start_pos, start_pos + seq_len)`` from it (``:538``), so ``start_pos``
    above ``seq_len`` raises ``RuntimeError: index N is out of bounds`` from inside
    ``gather_cos_sin``. That is chunk 3 of a chunked prefill (``start_pos = 2 * chunk``), i.e. the
    third call P7 would make. The chunked path must use :func:`build_indexed_rope` instead, and the
    assert below turns a confusing delegate error into a message that says so.
    """
    _assert_llama3_scaling(hf_config)
    assert start_pos <= seq_len, (
        f"build_prefill_rope(start_pos={start_pos}, seq_len={seq_len}): get_prefill_rot_mat only "
        f"precomputes seq_len*2={seq_len * 2} positions, so start_pos must be <= seq_len. "
        f"For chunked prefill past the second chunk use build_indexed_rope()."
    )
    return get_prefill_rot_mat(
        hf_config.head_dim,
        mesh_device,
        seq_len,
        theta=hf_config.rope_theta,
        scale_factor=hf_config.rope_scaling_factor,
        orig_context_len=hf_config.rope_orig_context_len,
        start_pos=start_pos,
    )


def build_indexed_rope(mesh_device, hf_config, *, max_seq_len, chunk_size, sp_axis=0, dtype=ttnn.bfloat16):
    """Build the whole-cache, block-cyclic, SP-sharded Meta cos/sin for the INDEXED RoPE, ONCE.

    Mirrors ``models/demos/gpt_oss_d_p/tt/rope.py:115``, with the llama3 frequency table in place
    of YaRN. The tables cover EVERY cache position (up to ``max_seq_len``), block-cyclic-reordered
    keyed by the per-chip chunk (``chunk_size // sp``) then SP-sharded on ``sp_axis``, so device
    ``c``'s contiguous shard holds — in local-cache-row order — the RoPE for every global position
    it will carry. ``ttnn.experimental.deepseek_prefill.rotary_embedding_indexed`` then derives this
    chunk's per-chip start row on-device from a single ``kv_actual_global`` runtime argument (the
    same block-cyclic ``update_idxt`` math the KV-cache writer uses), so there is no per-chunk host
    reshard.

    Constraints kept verbatim from the template (``rope.py:146``, ``:148``); at SP=4 the first is
    ``chunk_size % 128 == 0`` (``00_MODEL_CARD.md`` §4.4):
      * ``chunk_size % (TILE_SIZE * sp) == 0``
      * ``max_seq_len % chunk_size == 0``

    Returns ``[cos_tt, sin_tt]``, each ``[1, 1, max_seq_len/sp, head_dim]`` per chip — persistent,
    reused across all chunks; do NOT deallocate per chunk. Use with
    :func:`build_transformation_mat` and ``apply_rope(..., kv_actual_global=cached_len,
    cluster_axis=sp_axis)``.
    """
    _assert_llama3_scaling(hf_config)
    sp = mesh_device.shape[sp_axis]
    assert (
        chunk_size % (ttnn.TILE_SIZE * sp) == 0
    ), f"chunk_size ({chunk_size}) must be a multiple of TILE_SIZE * sp ({ttnn.TILE_SIZE * sp})"
    assert max_seq_len % chunk_size == 0, f"max_seq_len ({max_seq_len}) must be a multiple of chunk_size ({chunk_size})"
    chunk_local = chunk_size // sp

    cos, sin = build_meta_cos_sin(hf_config, max_seq_len)
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


def build_meta_cos_sin(hf_config, seq_len, start_pos=0):
    """Host-side Meta-interleaved cos/sin ``[1, 1, seq_len, head_dim]``, llama3-scaled.

    The host half of :func:`build_prefill_rope`, exposed separately because the unit test needs the
    torch tables to build the *HF-convention* pair from the same frequencies (the structure
    ``models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:83`` ``_build_cos_sin`` uses, so
    a test cannot silently compare two different RoPEs and call it a pass), and because
    :func:`build_indexed_rope` needs them before the block-cyclic reorder.
    """
    _assert_llama3_scaling(hf_config)
    cos, sin = precompute_freqs(
        hf_config.head_dim,
        start_pos + seq_len,
        theta=hf_config.rope_theta,
        scale_factor=hf_config.rope_scaling_factor,
        orig_context_len=hf_config.rope_orig_context_len,
        rope_type=hf_config.rope_type,
    )
    cos_meta, sin_meta = gather_cos_sin(torch.arange(start_pos, start_pos + seq_len), cos, sin)
    assert cos_meta.shape == sin_meta.shape == (1, 1, seq_len, hf_config.head_dim)
    return cos_meta, sin_meta
