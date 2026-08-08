# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test RoPE format differences between official qwen_tts and TTNN.

Official qwen_tts uses non-interleaved pairing: rotate_half splits at head_dim/2
and pairs dimension i with i + head_dim/2.

TTNN rotary_embedding_llama rotates adjacent pairs (LLaMa layout). Bridge with
``rearrange_to_interleaved`` on q and on cos/sin, then ``rearrange_to_noninterleaved``
on the output. MROPE section-concatenated cos/sin already match ``get_mrope_tensors``;
do not use a naive ``interleave_cos_sin`` on only the first half (it breaks MROPE).
"""

import argparse
from pathlib import Path

import pytest
import torch

import ttnn


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    a = a.flatten().float()
    b = b.flatten().float()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    cov = (a_centered * b_centered).sum()
    std_a = torch.sqrt((a_centered**2).sum())
    std_b = torch.sqrt((b_centered**2).sum())
    if std_a == 0 or std_b == 0:
        return 0.0
    return (cov / (std_a * std_b)).item()


def rotate_half(x):
    """Official qwen_tts rotate_half - NON-INTERLEAVED."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_noninterleaved(q, cos, sin):
    """Apply RoPE in non-interleaved format (official qwen_tts style)."""
    # cos/sin: [1, seq, head_dim] or [1, 1, seq, head_dim]
    # q: [batch, heads, seq, head_dim]
    return (q * cos) + (rotate_half(q) * sin)


def test_rope_formats(device):
    """Test RoPE format differences."""
    print("=" * 80)
    print("Testing RoPE Format Differences")
    print("=" * 80)

    # Load official MROPE data
    rotary_path = Path("/tmp/qwen_tts_tensors/rotary_data.pt")
    if not rotary_path.exists():
        pytest.skip(f"Missing {rotary_path}; run models/demos/qwen3_tts/demo/extract_rope.py first")

    rotary_data = torch.load(rotary_path)
    cos_official = rotary_data["cos"]  # [3, 1, seq, 128] — seq depends on extract_rope prompt
    sin_official = rotary_data["sin"]

    print(f"Official MROPE cos shape: {cos_official.shape}")
    print(f"Official MROPE sin shape: {sin_official.shape}")

    # For text-only, all 3 modalities have same positions, so use modality 0
    # But we need to reconstruct based on interleaved=True setting
    mrope_section = [24, 20, 20]  # [48, 40, 40] dims
    section_sizes = [s * 2 for s in mrope_section]

    # Reconstruct the non-interleaved cos/sin from MROPE
    cos_parts = []
    sin_parts = []
    start = 0
    for i, size in enumerate(section_sizes):
        cos_parts.append(cos_official[i, :, :, start : start + size])
        sin_parts.append(sin_official[i, :, :, start : start + size])
        start += size

    # Same layout as get_mrope_tensors: per-section [cos_w, cos_w] pairs (TTNN-ready).
    cos_mrope_layout = torch.cat(cos_parts, dim=-1)  # [1, seq, 128]
    sin_mrope_layout = torch.cat(sin_parts, dim=-1)

    seq_len = int(cos_mrope_layout.shape[1])
    head_dim = int(cos_mrope_layout.shape[2])
    pos_show = min(5, seq_len - 1) if seq_len > 0 else 0

    print(f"\nMROPE-layout cos shape: {cos_mrope_layout.shape} (seq_len={seq_len})")

    # Show some values
    print(f"\nPosition {pos_show} (capped for short sequences):")
    print(f"  cos_mrope_layout[0,{pos_show},0:8]: {cos_mrope_layout[0, pos_show, 0:8]}")
    print(f"  cos_mrope_layout[0,{pos_show},64:72]: {cos_mrope_layout[0, pos_show, 64:72]}")

    # Create test query tensor — seq_len must match rotary_data.pt (not a fixed 111)
    batch = 1
    num_heads = 16

    q = torch.randn(batch, num_heads, seq_len, head_dim)

    # Apply non-interleaved RoPE (official qwen style: pairs dim i with i + head_dim/2)
    cos_broadcast = cos_mrope_layout.unsqueeze(1)  # [1, 1, seq, 128]
    sin_broadcast = sin_mrope_layout.unsqueeze(1)

    q_rotated_official = apply_rotary_pos_emb_noninterleaved(q, cos_broadcast, sin_broadcast)

    print(f"\nOfficial RoPE output shape: {q_rotated_official.shape}")

    # TTNN rotary_embedding_llama pairs adjacent dims; permute q and use MROPE cos/sin as in get_mrope_tensors.
    from models.demos.qwen3_tts.tt.rope import (
        get_transformation_mat,
        rearrange_to_interleaved,
        rearrange_to_noninterleaved,
    )

    q_llama_layout = rearrange_to_interleaved(q)
    # cos/sin must use the same head_dim permutation as q for rotary_embedding_llama.
    cos_llama_layout = rearrange_to_interleaved(cos_mrope_layout)
    sin_llama_layout = rearrange_to_interleaved(sin_mrope_layout)

    pad_seq = ((seq_len + 31) // 32) * 32
    padding = pad_seq - seq_len

    cos_padded = torch.nn.functional.pad(cos_llama_layout.unsqueeze(0), (0, 0, 0, padding))  # [1, 1, pad_seq, 128]
    sin_padded = torch.nn.functional.pad(sin_llama_layout.unsqueeze(0), (0, 0, 0, padding))

    cos_tt = ttnn.from_torch(cos_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_tt = ttnn.from_torch(sin_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    trans_mat = get_transformation_mat(head_dim, device)

    q_padded = torch.nn.functional.pad(q_llama_layout, (0, 0, 0, padding))  # [1, 16, pad_seq, 128]
    q_tt = ttnn.from_torch(q_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    q_rotated_tt = ttnn.experimental.rotary_embedding_llama(q_tt, cos_tt, sin_tt, trans_mat, is_decode_mode=False)
    q_rotated_ttnn = rearrange_to_noninterleaved(ttnn.to_torch(q_rotated_tt)[:, :, :seq_len, :])

    # Compare
    pcc = compute_pcc(q_rotated_official, q_rotated_ttnn)
    print(f"\nPCC(official_rope, ttnn_rope): {pcc:.6f}")

    # Detailed comparison
    print(f"\nHead 0, position {pos_show}, first 8 dims:")
    print(f"  Official: {q_rotated_official[0, 0, pos_show, :8]}")
    print(f"  TTNN:     {q_rotated_ttnn[0, 0, pos_show, :8]}")

    max_diff = (q_rotated_official - q_rotated_ttnn).abs().max()
    print(f"\nMax absolute difference: {max_diff:.6f}")

    if pcc > 0.99:
        print("\n*** TTNN RoPE matches official! ***")
    else:
        print(f"\n*** MISMATCH: PCC={pcc:.4f}, need >0.99 ***")
        print("\n=== Debug: Understanding the format ===")
        q_identity = torch.zeros(1, 1, 1, head_dim)
        q_identity[0, 0, 0, 0] = 1.0
        q_identity[0, 0, 0, 64] = 1.0
        q_rot_official = apply_rotary_pos_emb_noninterleaved(
            q_identity,
            cos_broadcast[:, :, pos_show : pos_show + 1, :],
            sin_broadcast[:, :, pos_show : pos_show + 1, :],
        )
        print(f"\nIdentity rotation test (pos {pos_show}):")
        print("  Input: [1, 0, ..., 0, 1, 0, ..., 0] (dim 0 and 64)")
        print(f"  Official output first 4: {q_rot_official[0, 0, 0, :4]}")
        print(f"  Official output dim 64-68: {q_rot_official[0, 0, 0, 64:68]}")

    assert pcc > 0.99, (
        f"TTNN rotary_embedding_llama vs official non-interleaved RoPE PCC={pcc:.6f} "
        "(need >0.99). Check rearrange_to_interleaved / cos layout vs get_mrope_tensors."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    device = ttnn.open_device(device_id=args.device_id)
    try:
        test_rope_formats(device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
