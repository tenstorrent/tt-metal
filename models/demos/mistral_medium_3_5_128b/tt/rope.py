# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 YaRN RoPE setup for (chunked) prefill. Structure from ``gpt_oss_d_p/tt/rope.py``.

**The YaRN constants are not re-derived here.** ``gpt_oss_d_p``'s copy of ``_compute_yarn_parameters``
deliberately drops the ``floor``/``ceil`` on the correction dims because *its* config sets
``rope_scaling.truncate = false``. This checkpoint omits the key, so transformers uses the default
``truncate = True`` and *does* floor/ceil — the opposite branch. Getting that wrong is invisible at
short sequence length and worth ~1 rad of phase drift by position 5000, so this module imports
``reference.modeling.yarn_inv_freq`` (validated against ``ROPE_INIT_FUNCTIONS["yarn"]`` in
``tests/unit/test_reference_modeling.py``) rather than carrying a second implementation.

Convention: the cos/sin built here are **Meta interleaved** (``[c0, c0, c1, c1, ...]``), which is what
``ttnn.experimental.rotary_embedding_llama`` + ``get_rot_transformation_mat`` expect, and they pair with
Q/K projection rows permuted HF -> Meta by ``reference.modeling.hf_to_meta_head_perm``. The YaRN
``attention_scaling`` (0.1*ln(64)+1 = 1.4158883...) is folded into both matrices.

Sharding: cos/sin for absolute positions ``[start, end)`` are SP-sharded contiguously on the mesh rows
and replicated across the TP cols, matching the activation layout. Because every chunk starts at a
multiple of ``chunk_size``, the block-cyclic KV layout degenerates to exactly this contiguous split
(``rotated_chip_positions`` at a chunk-aligned offset is ``chip*chunk_local + row``), so one convention
covers the activations, the rope and the cache. A whole-cache indexed RoPE
(``rotary_embedding_indexed``) would avoid the per-chunk host build; it is a perf lever, not a
correctness one, and is out of scope here.
"""

import torch

import ttnn
from models.demos.mistral_medium_3_5_128b.reference.modeling import yarn_inv_freq
from models.tt_transformers.tt.common import get_rot_transformation_mat


def yarn_cos_sin_meta(config, start: int, end: int):
    """Meta-interleaved ``(cos, sin)`` torch tensors ``[1, 1, end-start, head_dim]``.

    Absolute positions ``[start, end)`` — the chunk's true offset in the sequence, so a chunk at
    ``cached_len`` gets the same rotation it would have had in a one-shot run. ``attention_scaling``
    is folded in. Computed in fp32 (as transformers does) and returned in fp32; the caller casts.
    """
    assert 0 <= start < end, f"empty or negative position range [{start}, {end})"
    inv_freq = yarn_inv_freq(config)  # [head_dim/2] fp32
    pos = torch.arange(start, end, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [end-start, head_dim/2]
    cos_half = freqs.cos() * config.attention_scaling
    sin_half = freqs.sin() * config.attention_scaling
    # Meta interleaved: [c0, c0, c1, c1, ...]. The reference's HF form is cat((freqs, freqs), -1)
    # instead; the two agree exactly once q/k rows are permuted by hf_to_meta_head_perm.
    cos = torch.stack([cos_half, cos_half], dim=-1).flatten(-2)[None, None]
    sin = torch.stack([sin_half, sin_half], dim=-1).flatten(-2)[None, None]
    return cos, sin


def build_transformation_mat(mesh_device, dtype=ttnn.bfloat16):
    """The replicated RoPE transformation matrix ``rotary_embedding_llama`` needs."""
    return ttnn.from_torch(
        get_rot_transformation_mat(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def build_rope_mats(
    mesh_device,
    config,
    start: int,
    end: int,
    *,
    mesh_config,
    dtype=ttnn.bfloat16,
    sequence_parallel: bool = True,
):
    """``[cos_tt, sin_tt]`` for absolute positions ``[start, end)``.

    With ``sequence_parallel`` (the production layout) the matrices are split contiguously over the
    SP rows to line up with the activations, and ``end - start`` must be a multiple of
    ``TILE_SIZE * sp`` so each chip's shard is tile-aligned. Without it they are replicated on
    every device, matching the non-SP diagnostic path in ``attention/prefill.py``.

    Returns the pair in the order ``operations.apply_rope`` expects.
    """
    cos, sin = yarn_cos_sin_meta(config, start, end)

    if sequence_parallel:
        sp = mesh_config.sp
        span = end - start
        assert span % (ttnn.TILE_SIZE * sp) == 0, (
            f"chunk length ({span}) must be a multiple of TILE_SIZE * sp ({ttnn.TILE_SIZE * sp}) so "
            f"each SP shard is tile-aligned"
        )
        mapper = mesh_config.shard_seq_sp(mesh_device, seq_dim=2)
    else:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)

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
