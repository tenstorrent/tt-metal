# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-free RoPE tests for Llama-3.1-8B prefill (tt-blaze#4145).

RoPE is the one cross-side contract that survives every byte-level migration gate: if prefill
rotates in a different frame than blaze decode writes, migration copies bytes faithfully and decode
reads a permutation. The ``dst-bytes`` gate passes, and per-layer KV PCC passes too whenever the
golden was generated with prefill's own convention rather than decode's.

So nothing here grades ``tt/rope.py`` against its own output. The oracles are:

  * **HuggingFace transformers** for the frequency curve and the rotation —
    ``ROPE_INIT_FUNCTIONS["llama3"]``, ``LlamaRotaryEmbedding`` and ``apply_rotary_pos_emb``, driven
    by a ``LlamaConfig`` carrying the real Llama-3.1-8B rope parameters.
  * **blaze's decode golden, restated locally** — ``rotate_half_meta_style`` from
    ``blaze/ops/rope/op.py`` and the adjacent-duplication table build from
    ``blaze/weights/llama31_8b/provider.py:make_cos_sin``. Restated rather than imported because
    tt-metal cannot depend on tt-blaze; the restatement is four lines and is what pins the frame.

The bridge between the two is ``convert_hf_qkv_to_meta_format``: HF ships q/k permuted so its
half-split ``rotate_half`` reproduces Meta's interleaved rotation, so "HF rope on HF-layout
activations" and "Meta rope on un-permuted activations" must agree exactly. That equivalence is the
real content of the acceptance criterion, and is what ``test_meta_frame_equals_hf_after_permute``
checks.
"""

from __future__ import annotations

import math

import pytest
import torch
from loguru import logger

from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tt.rope import build_llama3_cos_sin, llama3_inv_freq

HEAD_DIM = Llama31_8BConfig.HEAD_DIM  # 128
SERVED_CONTEXT = 8192  # sweep bound; ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS, where llama3 scaling bites

# Positions chosen to straddle the interesting regions: 0 (rope is the identity, where every bug is
# invisible), inside the original 8192 window, and well past it where the scaling dominates.
SWEEP_POSITIONS = [0, 1, 31, 32, 127, 1024, 4096, 8191, 8192, 16384, 65536, 131071]


def _hf_llama31_config():
    """A ``LlamaConfig`` carrying the real Llama-3.1-8B rope frame."""
    from transformers import LlamaConfig

    return LlamaConfig(
        hidden_size=Llama31_8BConfig.EMB_SIZE,
        num_attention_heads=Llama31_8BConfig.NUM_ATTENTION_HEADS,
        num_key_value_heads=Llama31_8BConfig.NUM_KEY_VALUE_HEADS,
        max_position_embeddings=Llama31_8BConfig.MAX_POSITION_EMBEDDINGS,
        rope_theta=Llama31_8BConfig.ROPE_THETA,
        rope_scaling={
            "rope_type": Llama31_8BConfig.ROPE_TYPE,
            "factor": Llama31_8BConfig.ROPE_SCALING_FACTOR,
            "low_freq_factor": Llama31_8BConfig.ROPE_LOW_FREQ_FACTOR,
            "high_freq_factor": Llama31_8BConfig.ROPE_HIGH_FREQ_FACTOR,
            "original_max_position_embeddings": Llama31_8BConfig.ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
        },
    )


def _rotate_half_meta(x: torch.Tensor) -> torch.Tensor:
    """blaze decode's rotation (``blaze/ops/rope/op.py:golden``), restated.

    ``[a0, a1, a2, a3, ...] -> [-a1, a0, -a3, a2, ...]`` — adjacent (interleaved) pairs.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _interleave_to_halfsplit(x: torch.Tensor) -> torch.Tensor:
    """Meta interleaved ``[r0,i0,r1,i1,...]`` -> HF half-split ``[r0,r1,...,i0,i1,...]``.

    The activation-space counterpart of ``load_checkpoints.permute`` on a q/k weight matrix, and the
    inverse of ``reverse_permute_1d``.
    """
    return torch.cat((x[..., ::2], x[..., 1::2]), dim=-1)


def _halfsplit_to_interleave(x: torch.Tensor) -> torch.Tensor:
    """HF half-split -> Meta interleaved. Same as ``load_checkpoints.reverse_permute_1d``."""
    half = x.shape[-1] // 2
    return torch.stack((x[..., :half], x[..., half:]), dim=-1).flatten(-2)


def test_llama3_inv_freq_matches_hf():
    """The frequency curve is HF's ``llama3``, and carries no attention factor.

    Pins both halves of the scaling contract: the per-frequency values, and the fact that llama3 —
    unlike YaRN, which the DeepSeek/Kimi/gpt-oss lineage in this package implements — has no mscale
    to fold into the tables. If transformers ever returns an attention factor other than 1.0 for
    llama3, the tables here would need to scale by it and this fails rather than drifting.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    config = _hf_llama31_config()
    hf_inv_freq, hf_attention_factor = ROPE_INIT_FUNCTIONS["llama3"](config, device="cpu")

    assert hf_attention_factor == 1.0, (
        f"HF llama3 returned attention_factor={hf_attention_factor}; the cos/sin tables in tt/rope.py "
        f"assume 1.0 and would need to fold it in"
    )
    torch.testing.assert_close(llama3_inv_freq(), hf_inv_freq.float(), rtol=0, atol=1e-9)


def test_llama3_scaling_actually_changes_the_frequencies():
    """Guard against silently shipping plain-theta tables.

    ``llama3`` scaling only touches wavelengths longer than ``orig_max_pos / low_freq_factor``, so a
    missing scaling step leaves the high-frequency half of the table untouched and identical. Every
    comparison against a golden built with the same omission would still pass, so assert here that
    the scaled curve genuinely differs from unscaled theta, and specifically at the low-frequency
    end where the division by ``factor`` applies.
    """
    scaled = llama3_inv_freq()
    plain = 1.0 / (Llama31_8BConfig.ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))

    assert not torch.allclose(scaled, plain), "llama3 scaling had no effect — tables are plain theta"

    # Longest wavelength (lowest frequency) is the last entry, and sits above low_freq_wavelen, so
    # it must be exactly inv_freq / factor.
    wavelen = 2 * math.pi / plain
    low_freq_wavelen = Llama31_8BConfig.ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS / Llama31_8BConfig.ROPE_LOW_FREQ_FACTOR
    assert wavelen[-1] > low_freq_wavelen
    torch.testing.assert_close(scaled[-1], plain[-1] / Llama31_8BConfig.ROPE_SCALING_FACTOR, rtol=0, atol=1e-12)


@pytest.mark.parametrize("position", SWEEP_POSITIONS)
def test_cos_sin_matches_hf_over_position_sweep(position):
    """Our Meta-interleaved tables carry HF's values at every swept position, out past the served
    context.

    HF emits the half-split layout (``cat((freqs, freqs), -1)``), ours the interleaved one
    (``stack(...).flatten``). Same numbers, different order — so the comparison de-interleaves ours
    rather than comparing raw, which is exactly the step that would hide a convention mix-up if it
    were applied to both sides.
    """
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    config = _hf_llama31_config()
    rotary = LlamaRotaryEmbedding(config=config)

    position_ids = torch.tensor([[position]], dtype=torch.long)
    dummy = torch.zeros(1, 1, 1, HEAD_DIM)
    hf_cos, hf_sin = rotary(dummy, position_ids)  # [1, 1, head_dim], half-split
    hf_cos_half = hf_cos[0, 0, : HEAD_DIM // 2]
    hf_sin_half = hf_sin[0, 0, : HEAD_DIM // 2]

    cos, sin = build_llama3_cos_sin(position + 1, head_dim=HEAD_DIM)
    # Interleaved: even lanes are the head_dim/2 distinct frequencies, odd lanes duplicate them.
    cos_row, sin_row = cos[0, 0, position], sin[0, 0, position]

    torch.testing.assert_close(cos_row[::2], hf_cos_half.float(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(sin_row[::2], hf_sin_half.float(), rtol=1e-6, atol=1e-6)
    # Adjacent duplication is the decode-side table shape (provider.py:make_cos_sin), not cat-halves.
    torch.testing.assert_close(cos_row[::2], cos_row[1::2], rtol=0, atol=0)
    torch.testing.assert_close(sin_row[::2], sin_row[1::2], rtol=0, atol=0)


def test_table_is_adjacent_duplication_not_cat_halves():
    """The two table conventions are distinguishable, and we build decode's.

    Both are the same multiset of values, so a length or magnitude check cannot tell them apart.
    Asserting they actually differ keeps this test honest: without it, the adjacency check above
    would also pass for a table where every frequency happened to be equal.
    """
    seq_len = 64
    cos, _ = build_llama3_cos_sin(seq_len, head_dim=HEAD_DIM)
    inv_freq = llama3_inv_freq(head_dim=HEAD_DIM)
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)

    adjacent = torch.stack([freqs.cos(), freqs.cos()], dim=-1).flatten(-2)  # decode / Meta
    cat_halves = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # HF

    torch.testing.assert_close(cos[0, 0], adjacent, rtol=0, atol=0)
    assert not torch.allclose(adjacent, cat_halves), "the two table layouts are indistinguishable here"


@pytest.mark.parametrize("position", [0, 1, 37, 1024, 8192, 100000])
def test_meta_frame_equals_hf_after_permute(position):
    """The applied rotation is the frame blaze decode writes.

    HF ships q/k permuted so that its half-split ``rotate_half`` reproduces Meta's interleaved
    rotation; ``convert_hf_qkv_to_meta_format`` un-permutes them for the interleaved device op. So
    for any activation, rotating in the Meta frame with our adjacently-duplicated tables must equal
    rotating the *permuted* activation in HF's frame with HF's tables and permuting the result back.

    This is the acceptance criterion's "compared against a decode-generated reference, not prefill's
    own golden": the rotation is blaze decode's ``rotate_half_meta_style``, the tables are ours, and
    the oracle on the other side is HuggingFace's own ``apply_rotary_pos_emb``. Position 0 is
    included to show the test is not vacuous there, and 100000 to catch frame errors that only
    compound with position.
    """
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

    torch.manual_seed(0)
    n_heads = 4
    q_meta = torch.randn(1, n_heads, 1, HEAD_DIM, dtype=torch.float32)  # device (Meta) layout

    # --- Meta / decode frame: our tables + blaze's interleaved rotation ---
    cos, sin = build_llama3_cos_sin(position + 1, head_dim=HEAD_DIM)
    cos_row = cos[:, :, position : position + 1, :]  # [1,1,1,head_dim]
    sin_row = sin[:, :, position : position + 1, :]
    out_meta = q_meta * cos_row + _rotate_half_meta(q_meta) * sin_row

    # --- HF frame: permute the activation, use HF's own tables and rotation ---
    config = _hf_llama31_config()
    rotary = LlamaRotaryEmbedding(config=config)
    position_ids = torch.tensor([[position]], dtype=torch.long)
    hf_cos, hf_sin = rotary(q_meta, position_ids)

    q_hf = _interleave_to_halfsplit(q_meta)
    out_hf, _ = apply_rotary_pos_emb(q_hf, q_hf, hf_cos, hf_sin)
    out_hf_as_meta = _halfsplit_to_interleave(out_hf)

    torch.testing.assert_close(out_meta, out_hf_as_meta.float(), rtol=1e-5, atol=1e-5)
    logger.info(f"Meta frame == HF frame after permutation at position {position}")


def test_permute_helpers_are_inverses():
    """The two layout maps used above really are inverses.

    If they were not, ``test_meta_frame_equals_hf_after_permute`` could pass by applying
    compensating errors on both sides.
    """
    x = torch.randn(2, 3, HEAD_DIM)
    torch.testing.assert_close(_halfsplit_to_interleave(_interleave_to_halfsplit(x)), x, rtol=0, atol=0)
    torch.testing.assert_close(_interleave_to_halfsplit(_halfsplit_to_interleave(x)), x, rtol=0, atol=0)


def test_tables_are_per_head_dim_not_full_projection_width():
    """Tables must be built at ``head_dim`` and tiled across heads, never at ``n_heads * head_dim``.

    Building one table of width ``n_heads * head_dim`` spreads ``n_heads * head_dim / 2`` distinct
    frequencies over the projection, so head 0's frequencies decay ``n_heads`` times too slowly and
    each later head gets a different set (``provider.py:make_cos_sin_tiled``). The reason it needs a
    dedicated test: the error is exactly zero at position 0 and grows with position, and a
    device-vs-golden comparison cannot see it at all because both sides would share the wrong table.
    """
    n_heads, seq_len = 8, 256
    cos_head, _ = build_llama3_cos_sin(seq_len, head_dim=HEAD_DIM)
    tiled = cos_head[0, 0].repeat(1, n_heads)  # the correct full-width table
    assert tiled.shape == (seq_len, n_heads * HEAD_DIM)

    wrong, _ = build_llama3_cos_sin(seq_len, head_dim=n_heads * HEAD_DIM)
    wrong = wrong[0, 0]

    # Invisible at position 0 ...
    torch.testing.assert_close(tiled[0], wrong[0], rtol=0, atol=0)
    # ... and badly wrong once position advances.
    assert not torch.allclose(tiled[1], wrong[1], atol=1e-3), "the full-width trap is not detectable here"
    drift = (tiled[seq_len - 1] - wrong[seq_len - 1]).abs().max()
    assert drift > 0.1, f"expected the full-width table to diverge by the end of the sweep, max drift {drift}"
    logger.info(f"full-width table drift at position {seq_len - 1}: {drift:.4f}")

    # Every head shares one frequency set, which is what makes tiling correct in the first place.
    per_head = tiled.reshape(seq_len, n_heads, HEAD_DIM)
    for h in range(1, n_heads):
        torch.testing.assert_close(per_head[:, 0], per_head[:, h], rtol=0, atol=0)


def test_build_indexed_rope_enforces_layout_constraints(expect_error):
    """``chunk_size % (TILE_SIZE * sp) == 0`` and ``max_seq_len % chunk_size == 0`` are enforced.

    Both constraints are shared with the KV cache's block-cyclic layout; violating either puts rope
    rows on a different chip than the tokens they are supposed to rotate, which is a silent
    wrong-answer rather than a crash. Checked with a stub mesh so this stays device-free — the
    validation runs before any ttnn call.
    """
    from models.demos.llama_3p1_8b_d_p.tt.rope import build_indexed_rope

    class StubMesh:
        shape = (4, 8)  # SP=4 rows, TP=8 cols — the Llama-3.1-8B target

    with expect_error(ValueError, "multiple of TILE_SIZE"):
        build_indexed_rope(StubMesh(), max_seq_len=4096, chunk_size=100, sp_axis=0)
    with expect_error(ValueError, "multiple of chunk_size"):
        build_indexed_rope(StubMesh(), max_seq_len=5000, chunk_size=512, sp_axis=0)


def test_block_cyclic_reorder_matches_the_deepseek_original():
    """This module's ``block_cyclic_reorder`` is byte-identical to the one it was restated from.

    It is restated rather than imported because importing ``deepseek_v3_d_p.tt.mla.utils`` pulls
    safetensors and transformers onto the prefill runtime's import path (see the function's
    docstring). That trade is only safe if the copy cannot drift, so grade it against the original
    here — this test is device-free but not import-light, which is exactly the point: the heavy
    import lives in the test instead of in the module under serving.
    """
    from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder as original
    from models.demos.llama_3p1_8b_d_p.tt.rope import block_cyclic_reorder as ours

    for sp, chunk_local, seq_len in ((4, 64, 1024), (4, 32, 512), (2, 128, 1024), (1, 64, 256), (8, 32, 2048)):
        table = torch.arange(seq_len, dtype=torch.float32).reshape(1, 1, seq_len, 1).expand(1, 1, seq_len, 4)
        torch.testing.assert_close(
            ours(table.contiguous(), chunk_local, sp, seq_dim=2),
            original(table.contiguous(), chunk_local, sp, seq_dim=2),
            rtol=0,
            atol=0,
        )


def test_block_cyclic_reorder_rejects_indivisible_layouts(expect_error):
    """The two divisibility rules are raised, not asserted away under ``python -O``.

    The DeepSeek original uses bare ``assert``, which ``-O`` strips; a stripped check here would
    silently reorder a partial block and put rope rows on the wrong chip.
    """
    from models.demos.llama_3p1_8b_d_p.tt.rope import block_cyclic_reorder

    table = torch.zeros(1, 1, 100, 4)
    with expect_error(ValueError, "multiple of chunk_local"):
        block_cyclic_reorder(table, 32, 4, seq_dim=2)
    with expect_error(ValueError, "multiple of sp_factor"):
        block_cyclic_reorder(torch.zeros(1, 1, 96, 4), 32, 4, seq_dim=2)
