# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit test for Qwen3.6 partial-RoPE math.

Verifies that the math expression in
``models/demos/qwen3_6_galaxy_v2/tt/llama_rope.py::apply_qwen36_partial_rope_torch``
matches the HF reference (via the same ``_rotate_half`` formula used in
``models/demos/qwen3_6_galaxy/reference/qwen36.py``):

    x_rot, x_pass = x[..., :rd], x[..., rd:]
    x1, x2 = x_rot[..., :rd//2], x_rot[..., rd//2:]
    rotate_half = cat([-x2, x1], dim=-1)
    x_rot_out = x_rot * cos + rotate_half * sin
    out = cat([x_rot_out, x_pass], dim=-1)

No TTNN device is required — pure-pytorch comparison only.
The device wiring of the same expression will be exercised in V2-7.
"""

import torch

from models.demos.qwen3_6_galaxy.reference.qwen36 import _rotate_half, build_mrope_cos_sin
from models.demos.qwen3_6_galaxy_v2.tt.llama_rope import (
    apply_qwen36_partial_rope_torch,
    build_qwen36_partial_rope_tables,
)

# ---------------------------------------------------------------------------
# HF-style oracle (pure pytorch)
# ---------------------------------------------------------------------------


def hf_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Match HF ``apply_rotary_pos_emb`` for partial RoPE.

    Args:
        x:        [..., head_dim]
        cos, sin: broadcastable to [..., rope_dim]

    Identical to the per-tensor RoPE block in
    ``qwen3_6_galaxy/reference/qwen36.py::GatedAttention.forward``.
    """
    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x_rot_out = (x_rot * cos) + (_rotate_half(x_rot) * sin)
    return torch.cat([x_rot_out, x_pass], dim=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_partial_rope_matches_hf():
    """Our partial-RoPE math agrees with the HF reference to atol=1e-6."""
    head_dim = 256
    rope_dim = 64
    rope_theta = 10_000_000.0
    T = 4
    num_heads = 3

    # Deterministic inputs.
    q = torch.arange(1 * num_heads * T * head_dim, dtype=torch.float32).reshape(1, num_heads, T, head_dim) * 1e-4
    k = torch.arange(1 * num_heads * T * head_dim, dtype=torch.float32).reshape(1, num_heads, T, head_dim) * 1e-4
    # Offset k to differentiate from q.
    k = k + 0.5

    # Build cos/sin via our v2 helper.
    cos_table, sin_table = build_qwen36_partial_rope_tables(max_seq_len=T, rope_dim=rope_dim, rope_theta=rope_theta)
    # cos_table: [T, rope_dim] -> reshape to [1, 1, T, rope_dim] for broadcasting.
    cos = cos_table.unsqueeze(0).unsqueeze(0)  # [1, 1, T, rope_dim]
    sin = sin_table.unsqueeze(0).unsqueeze(0)

    # Apply our partial RoPE.
    q_ours = apply_qwen36_partial_rope_torch(q, cos, sin, rope_dim=rope_dim)
    k_ours = apply_qwen36_partial_rope_torch(k, cos, sin, rope_dim=rope_dim)

    # Apply HF reference.
    q_hf = hf_partial_rope(q, cos, sin, rope_dim=rope_dim)
    k_hf = hf_partial_rope(k, cos, sin, rope_dim=rope_dim)

    assert torch.allclose(
        q_ours, q_hf, atol=1e-6
    ), f"q mismatch: max abs diff = {(q_ours - q_hf).abs().max().item():.3e}"
    assert torch.allclose(
        k_ours, k_hf, atol=1e-6
    ), f"k mismatch: max abs diff = {(k_ours - k_hf).abs().max().item():.3e}"

    # Pass-through channels must equal the input verbatim.
    assert torch.allclose(q_ours[..., rope_dim:], q[..., rope_dim:], atol=0.0)
    assert torch.allclose(k_ours[..., rope_dim:], k[..., rope_dim:], atol=0.0)


def test_cos_sin_tables_match_build_mrope_text_only():
    """Our cos/sin tables agree with the v1 MRoPE oracle in text-only mode.

    In text-only inference all 3 MRoPE position axes equal the token index,
    and (since the section grouping is by *channel pairs*, not positions) the
    result is numerically identical to a single-axis partial RoPE.
    """
    head_dim = 256
    rope_dim = 64
    rope_theta = 10_000_000.0
    mrope_section = [11, 11, 10]
    partial_rotary_factor = rope_dim / head_dim
    T = 8

    # v1 MRoPE oracle (text-only: all three axes = arange(T))
    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos_mrope, sin_mrope = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=head_dim,
        partial_rotary_factor=partial_rotary_factor,
        mrope_section=mrope_section,
        theta=rope_theta,
    )
    # cos_mrope: [1, T, rope_dim] -> squeeze to [T, rope_dim]
    cos_mrope = cos_mrope.squeeze(0)
    sin_mrope = sin_mrope.squeeze(0)

    # v2 helper
    cos_v2, sin_v2 = build_qwen36_partial_rope_tables(max_seq_len=T, rope_dim=rope_dim, rope_theta=rope_theta)

    assert torch.allclose(
        cos_v2, cos_mrope, atol=1e-6
    ), f"cos mismatch: max abs diff = {(cos_v2 - cos_mrope).abs().max().item():.3e}"
    assert torch.allclose(
        sin_v2, sin_mrope, atol=1e-6
    ), f"sin mismatch: max abs diff = {(sin_v2 - sin_mrope).abs().max().item():.3e}"


def test_decode_shape_broadcast():
    """Decode-style cos/sin of shape [1, 1, 1, rope_dim] broadcasts cleanly."""
    head_dim = 256
    rope_dim = 64
    rope_theta = 10_000_000.0
    pos = 17
    num_heads = 2
    B = 1

    x = torch.randn(B, num_heads, 1, head_dim, dtype=torch.float32)
    cos_table, sin_table = build_qwen36_partial_rope_tables(
        max_seq_len=pos + 1, rope_dim=rope_dim, rope_theta=rope_theta
    )
    cos = cos_table[pos : pos + 1].unsqueeze(0).unsqueeze(0)  # [1, 1, 1, rope_dim]
    sin = sin_table[pos : pos + 1].unsqueeze(0).unsqueeze(0)

    x_ours = apply_qwen36_partial_rope_torch(x, cos, sin, rope_dim=rope_dim)
    x_hf = hf_partial_rope(x, cos, sin, rope_dim=rope_dim)
    assert torch.allclose(
        x_ours, x_hf, atol=1e-6
    ), f"decode-shape mismatch: max abs diff = {(x_ours - x_hf).abs().max().item():.3e}"


if __name__ == "__main__":
    test_partial_rope_matches_hf()
    test_cos_sin_tables_match_build_mrope_text_only()
    test_decode_shape_broadcast()
    print("All partial-RoPE math tests passed.")
