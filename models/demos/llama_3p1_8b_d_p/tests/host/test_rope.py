# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-free numerical contract tests for Llama-3.1 RoPE."""

import math
import os

import pytest
import torch
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import ROPE_INIT_FUNCTIONS, LlamaRotaryEmbedding, apply_rotary_pos_emb

from models.demos.llama_3p1_8b_d_p.tt import rope

HF_MODEL = os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct")
BOUNDARY_POSITIONS = (0, 31, 32, 255, 256, 1023, 1024, 8191, 8192)


def _hf_config():
    return AutoConfig.from_pretrained(HF_MODEL)


def _rotate_meta(x):
    """Independent adjacent-pair rotation used only as the Meta-frame test oracle."""
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = -x[..., 1::2]
    rotated[..., 1::2] = x[..., 0::2]
    return rotated


# Compare every Llama3 scaling band with HF; exact agreement catches a wrong scaling formula or branch.
def test_llama3_inv_freq_matches_hf_in_all_wavelength_bands():
    """Catches default/linear/YaRN scaling or an incorrect smoothing branch."""
    config = _hf_config()
    expected, attention_factor = ROPE_INIT_FUNCTIONS["llama3"](config, device=torch.device("cpu"))
    actual = rope.llama3_inv_freq()

    base_inv_freq = 1.0 / (
        config.rope_parameters["rope_theta"]
        ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
    )
    wavelengths = 2 * math.pi / base_inv_freq
    high_boundary = (
        config.rope_parameters["original_max_position_embeddings"] / config.rope_parameters["high_freq_factor"]
    )
    low_boundary = (
        config.rope_parameters["original_max_position_embeddings"] / config.rope_parameters["low_freq_factor"]
    )
    masks = (
        wavelengths < high_boundary,
        (wavelengths >= high_boundary) & (wavelengths <= low_boundary),
        wavelengths > low_boundary,
    )

    assert attention_factor == 1.0
    assert actual.dtype == torch.float32
    assert actual.shape == (config.head_dim // 2,)
    assert all(mask.any() for mask in masks), "the oracle must exercise all three Llama3 wavelength bands"
    for mask in masks:
        torch.testing.assert_close(actual[mask], expected[mask], rtol=1e-6, atol=1e-8)


# Sample tile, chunk, and context edges against HF; matching Meta-pair tables catch bad origins and layout.
def test_cos_sin_tables_match_hf_at_tile_chunk_and_context_boundaries():
    """Catches half-split tables, wrong position origin, and off-by-one table construction."""
    config = _hf_config()
    position_ids = torch.tensor([BOUNDARY_POSITIONS], dtype=torch.long)
    hf_cos, hf_sin = LlamaRotaryEmbedding(config)(torch.empty(1, dtype=torch.float32), position_ids)

    cos, sin = rope.build_llama3_cos_sin(max(BOUNDARY_POSITIONS) + 1)
    assert cos.shape == sin.shape == (1, 1, 8193, config.head_dim)
    assert cos.dtype == sin.dtype == torch.float32

    # HF duplicates by half; Meta duplicates each frequency adjacently.
    expected_cos = torch.repeat_interleave(hf_cos[..., : config.head_dim // 2], 2, dim=-1).unsqueeze(1)
    expected_sin = torch.repeat_interleave(hf_sin[..., : config.head_dim // 2], 2, dim=-1).unsqueeze(1)
    torch.testing.assert_close(cos[:, :, position_ids[0]], expected_cos, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(sin[:, :, position_ids[0]], expected_sin, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cos[..., 0::2], cos[..., 1::2], rtol=0, atol=0)
    torch.testing.assert_close(sin[..., 0::2], sin[..., 1::2], rtol=0, atol=0)


# Convert a hand-written vector to Meta order and back; exact literals catch a reversible but wrong permutation.
def test_hf_meta_conversion_has_known_adjacent_pair_coordinates():
    """Catches a half-split table accidentally treated as already Meta-interleaved."""
    hf = torch.tensor([[[0, 1, 2, 3, 10, 11, 12, 13]]], dtype=torch.float64)
    expected_meta = torch.tensor([[[0, 10, 1, 11, 2, 12, 3, 13]]], dtype=torch.float64)

    meta = rope.hf_to_meta(hf)
    assert meta.dtype == hf.dtype and meta.device == hf.device
    torch.testing.assert_close(meta, expected_meta, rtol=0, atol=0)
    torch.testing.assert_close(rope.meta_to_hf(expected_meta), hf, rtol=0, atol=0)


# Apply the same coordinate conversion to full Q and K head counts; both must preserve the intended pair mapping.
def test_projection_conversion_covers_q_and_k_head_counts():
    """Catches omitting or differently permuting either projected Q or projected K."""
    q_hf = torch.arange(32 * 128, dtype=torch.float32).reshape(1, 32, 1, 128)
    k_hf = (10000 + torch.arange(8 * 128, dtype=torch.float32)).reshape(1, 8, 1, 128)

    for projected_hf in (q_hf, k_hf):
        projected_meta = rope.hf_to_meta(projected_hf)
        torch.testing.assert_close(projected_meta[..., 0::2], projected_hf[..., :64], rtol=0, atol=0)
        torch.testing.assert_close(projected_meta[..., 1::2], projected_hf[..., 64:], rtol=0, atol=0)


# Rotate random full-size Q and K through both frames; matching HF catches frame-compatible-looking math errors.
def test_meta_rotation_of_random_q_and_k_matches_hf_llama():
    """Catches a coordinate conversion that is reversible but incompatible with HF rotation."""
    torch.manual_seed(20260915)
    positions = torch.tensor([BOUNDARY_POSITIONS], dtype=torch.long)
    q_hf = torch.randn(1, 32, len(BOUNDARY_POSITIONS), 128)
    k_hf = torch.randn(1, 8, len(BOUNDARY_POSITIONS), 128)

    hf_cos, hf_sin = LlamaRotaryEmbedding(_hf_config())(q_hf, positions)
    expected_q, expected_k = apply_rotary_pos_emb(q_hf, k_hf, hf_cos, hf_sin)

    meta_cos, meta_sin = rope.build_llama3_cos_sin(max(BOUNDARY_POSITIONS) + 1)
    meta_cos = meta_cos[:, :, positions[0]]
    meta_sin = meta_sin[:, :, positions[0]]
    q_meta = rope.hf_to_meta(q_hf)
    k_meta = rope.hf_to_meta(k_hf)
    actual_q = rope.meta_to_hf(q_meta * meta_cos + _rotate_meta(q_meta) * meta_sin)
    actual_k = rope.meta_to_hf(k_meta * meta_cos + _rotate_meta(k_meta) * meta_sin)

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-5, atol=1e-5)


# Round-trip several host dtypes and shapes exactly; any value, dtype, or shape change exposes a lossy conversion.
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.float64))
def test_coordinate_conversion_round_trip_preserves_tensor_properties(dtype):
    """Catches lossy reshaping, dtype conversion, or loss of arbitrary leading dimensions."""
    source = torch.randn(2, 3, 5, 128, dtype=dtype)
    restored = rope.meta_to_hf(rope.hf_to_meta(source))

    assert restored.shape == source.shape
    assert restored.dtype == source.dtype
    assert restored.device == source.device
    torch.testing.assert_close(restored, source, rtol=0, atol=0)


# Reject an unpaired final coordinate in both directions; a clear error prevents silent truncation.
@pytest.mark.parametrize("name", ("hf_to_meta", "meta_to_hf"))
def test_coordinate_conversion_rejects_odd_head_dimension(name):
    """Catches silent truncation of the unpaired final coordinate."""
    with pytest.raises(ValueError, match="even"):  # allow-pytest.raises: pure-host test uses --noconftest
        getattr(rope, name)(torch.zeros(2, 3, 127))


# Reject odd dimensions in both frequency builders; this prevents constructing incomplete rotary pairs.
def test_frequency_and_table_builders_reject_odd_head_dimension():
    """Catches construction of an incomplete rotary pair."""
    with pytest.raises(ValueError, match="even"):  # allow-pytest.raises: pure-host test uses --noconftest
        rope.llama3_inv_freq(head_dim=127)
    with pytest.raises(ValueError, match="even"):  # allow-pytest.raises: pure-host test uses --noconftest
        rope.build_llama3_cos_sin(32, head_dim=127)


# Size tables for a full padded tail and round to chunks; 3072 rows catch accidental logical-only allocation.
def test_indexed_rope_table_capacity_covers_the_last_physical_chunk():
    """Catches sizing tables only to the logical KV limit instead of the kernel's padded reads."""

    assert rope.indexed_rope_table_capacity(max_seq_len=2048, chunk_size=1024) == 3072
    assert rope.indexed_rope_table_capacity(max_seq_len=2049, chunk_size=1024) == 4096


# Feed invalid limits, axes, dimensions, and local chunks; each must fail clearly before TTNN device setup.
@pytest.mark.parametrize(
    "mesh_shape,max_seq_len,chunk_size,sp_axis,match",
    (
        ((4, 8), 0, 1024, 0, "max_seq_len"),
        ((4, 8), 2048, 0, 0, "chunk_size"),
        ((4, 8), 2048, 1024, 2, "sp_axis"),
        ((0, 8), 2048, 1024, 0, "positive"),
        ((4, 8), 2048, 1000, 0, "tile-aligned"),
    ),
)
def test_indexed_rope_setup_rejects_unsupported_geometry(mesh_shape, max_seq_len, chunk_size, sp_axis, match):
    """Catches malformed SP geometry reaching TTNN as wrong or non-tile table shards."""
    fake_mesh = type("FakeMesh", (), {"shape": mesh_shape})()

    with pytest.raises(ValueError, match=match):  # allow-pytest.raises: validation runs before local TTNN import
        rope.build_indexed_rope(
            fake_mesh,
            max_seq_len=max_seq_len,
            chunk_size=chunk_size,
            sp_axis=sp_axis,
        )
