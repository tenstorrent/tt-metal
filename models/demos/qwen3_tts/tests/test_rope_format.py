# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test RoPE format differences between official qwen_tts and TTNN.

Official qwen_tts uses NON-INTERLEAVED format:
    - rotate_half: split at middle, concat(-x2, x1)
    - cos/sin shape: [seq, head_dim] with sequential frequencies

TTNN rotary_embedding_llama uses INTERLEAVED format:
    - Rotation applied to adjacent pairs
    - cos/sin shape: [seq, head_dim] with interleaved frequencies [cos0, cos0, cos1, cos1, ...]
"""

import argparse
from pathlib import Path

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


def interleave_cos_sin(cos_noninterleaved, sin_noninterleaved):
    """
    Convert non-interleaved cos/sin to interleaved format.

    Non-interleaved: [cos_0, cos_1, ..., cos_63] with freq_i for dim i
    Interleaved: [cos_0, cos_0, cos_1, cos_1, ...] with freq_i for dims 2i and 2i+1

    But actually, the transformation is:
    - In non-interleaved, freq[i] is applied to dims i and i+dim/2
    - In interleaved, freq[i] is applied to dims 2i and 2i+1

    To convert: rearrange so freq[i] goes to positions 2i and 2i+1
    """
    # cos_noninterleaved: [batch, seq, head_dim]
    # Split into two halves
    dim = cos_noninterleaved.shape[-1]
    half_dim = dim // 2

    # Non-interleaved has freqs for first half, need to duplicate for interleaved
    cos_first_half = cos_noninterleaved[..., :half_dim]
    sin_first_half = sin_noninterleaved[..., :half_dim]

    # Interleave: [cos0, cos0, cos1, cos1, ...]
    cos_interleaved = torch.stack([cos_first_half, cos_first_half], dim=-1).flatten(-2)
    sin_interleaved = torch.stack([sin_first_half, sin_first_half], dim=-1).flatten(-2)

    return cos_interleaved, sin_interleaved


def test_rope_formats(device):
    """Test RoPE format differences."""
    print("=" * 80)
    print("Testing RoPE Format Differences")
    print("=" * 80)

    # Load official MROPE data
    rotary_path = Path("/tmp/qwen_tts_tensors/rotary_data.pt")
    if not rotary_path.exists():
        print(f"ERROR: {rotary_path} not found")
        return

    rotary_data = torch.load(rotary_path)
    cos_official = rotary_data["cos"]  # [3, 1, 111, 128]
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

    cos_noninterleaved = torch.cat(cos_parts, dim=-1)  # [1, 111, 128]
    sin_noninterleaved = torch.cat(sin_parts, dim=-1)

    print(f"\nNon-interleaved cos shape: {cos_noninterleaved.shape}")

    # Show some values
    print(f"\nPosition 5 (middle of sequence):")
    print(f"  cos_noninterleaved[0,5,0:8]: {cos_noninterleaved[0,5,0:8]}")
    print(f"  cos_noninterleaved[0,5,64:72]: {cos_noninterleaved[0,5,64:72]}")

    # Create test query tensor
    batch = 1
    num_heads = 16
    seq_len = 111
    head_dim = 128

    q = torch.randn(batch, num_heads, seq_len, head_dim)

    # Apply non-interleaved RoPE (official style)
    cos_broadcast = cos_noninterleaved.unsqueeze(1)  # [1, 1, 111, 128]
    sin_broadcast = sin_noninterleaved.unsqueeze(1)

    q_rotated_official = apply_rotary_pos_emb_noninterleaved(q, cos_broadcast, sin_broadcast)

    print(f"\nOfficial RoPE output shape: {q_rotated_official.shape}")

    # Now test TTNN rotary_embedding_llama
    # First, we need to convert cos/sin to interleaved format
    cos_interleaved, sin_interleaved = interleave_cos_sin(cos_noninterleaved, sin_noninterleaved)

    print(f"Interleaved cos shape: {cos_interleaved.shape}")

    # For TTNN, we need [1, 1, seq, head_dim]
    pad_seq = ((seq_len + 31) // 32) * 32
    padding = pad_seq - seq_len

    cos_padded = torch.nn.functional.pad(cos_interleaved.unsqueeze(0), (0, 0, 0, padding))  # [1, 1, pad_seq, 128]
    sin_padded = torch.nn.functional.pad(sin_interleaved.unsqueeze(0), (0, 0, 0, padding))

    cos_tt = ttnn.from_torch(cos_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_tt = ttnn.from_torch(sin_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Get transformation matrix
    from models.demos.qwen3_tts.tt.rope import get_transformation_mat

    trans_mat = get_transformation_mat(head_dim, device)

    # Prepare q for TTNN: [batch, heads, seq, head_dim]
    q_padded = torch.nn.functional.pad(q, (0, 0, 0, padding))  # [1, 16, pad_seq, 128]
    q_tt = ttnn.from_torch(q_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply TTNN RoPE
    q_rotated_tt = ttnn.experimental.rotary_embedding_llama(q_tt, cos_tt, sin_tt, trans_mat, is_decode_mode=False)
    q_rotated_ttnn = ttnn.to_torch(q_rotated_tt)[:, :, :seq_len, :]

    # Compare
    pcc = compute_pcc(q_rotated_official, q_rotated_ttnn)
    print(f"\nPCC(official_rope, ttnn_rope): {pcc:.6f}")

    # Detailed comparison
    print(f"\nHead 0, position 5, first 8 dims:")
    print(f"  Official: {q_rotated_official[0, 0, 5, :8]}")
    print(f"  TTNN:     {q_rotated_ttnn[0, 0, 5, :8]}")

    max_diff = (q_rotated_official - q_rotated_ttnn).abs().max()
    print(f"\nMax absolute difference: {max_diff:.6f}")

    if pcc > 0.99:
        print("\n*** TTNN RoPE matches official! ***")
        return True
    else:
        print(f"\n*** MISMATCH: PCC={pcc:.4f}, need >0.99 ***")

        # Debug: Try direct format comparison
        print("\n=== Debug: Understanding the format ===")

        # Test with identity q to see the rotation pattern
        q_identity = torch.zeros(1, 1, 1, head_dim)
        q_identity[0, 0, 0, 0] = 1.0
        q_identity[0, 0, 0, 64] = 1.0

        q_rot_official = apply_rotary_pos_emb_noninterleaved(
            q_identity, cos_broadcast[:, :, 5:6, :], sin_broadcast[:, :, 5:6, :]
        )
        print(f"\nIdentity rotation test (pos 5):")
        print(f"  Input: [1, 0, ..., 0, 1, 0, ..., 0] (dim 0 and 64)")
        print(f"  Official output first 4: {q_rot_official[0, 0, 0, :4]}")
        print(f"  Official output dim 64-68: {q_rot_official[0, 0, 0, 64:68]}")

        return False


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
