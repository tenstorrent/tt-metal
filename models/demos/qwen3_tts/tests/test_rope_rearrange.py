# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test RoPE with dimension rearrangement.

Official qwen_tts uses non-interleaved RoPE that pairs dimensions (i, i+64).
TTNN rotary_embedding_llama uses interleaved RoPE that pairs dimensions (2i, 2i+1).

Solution: Rearrange Q/K dimensions before TTNN RoPE, then rearrange back.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

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


def rearrange_to_interleaved(x):
    """
    Rearrange from non-interleaved to interleaved format.

    Input:  [..., d0, d1, ..., d63, d64, d65, ..., d127]
    Output: [..., d0, d64, d1, d65, ..., d63, d127]

    This maps pairs (i, i+64) to pairs (2i, 2i+1).
    """
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    # Split into two halves
    x1 = x[..., :half_dim]  # dims 0-63
    x2 = x[..., half_dim:]  # dims 64-127

    # Interleave
    result = torch.stack([x1, x2], dim=-1).flatten(-2)  # [..., d0, d64, d1, d65, ...]
    return result


def rearrange_to_noninterleaved(x):
    """
    Rearrange from interleaved to non-interleaved format (inverse of above).

    Input:  [..., d0, d64, d1, d65, ..., d63, d127]
    Output: [..., d0, d1, ..., d63, d64, d65, ..., d127]
    """
    head_dim = x.shape[-1]

    # De-interleave: take even and odd indices
    x1 = x[..., 0::2]  # dims 0, 2, 4, ... (originally d0, d1, d2, ...)
    x2 = x[..., 1::2]  # dims 1, 3, 5, ... (originally d64, d65, d66, ...)

    # Concatenate
    result = torch.cat([x1, x2], dim=-1)  # [..., d0, d1, ..., d63, d64, d65, ...]
    return result


def run_test(device):
    """Test RoPE with dimension rearrangement."""
    print("=" * 80)
    print("Testing RoPE with Dimension Rearrangement")
    print("=" * 80)

    # Load official tensors
    data_path = Path("/tmp/qwen_tts_tensors/layer0_attention_tensors.pt")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        return

    data = torch.load(data_path)

    q_before = data["q_before_rope"]  # [1, 16, 111, 128]
    q_after_official = data["q_after_rope"]
    cos_official = data["cos"]  # [3, 1, 111, 128]
    sin_official = data["sin"]

    batch, num_heads, seq_len, head_dim = q_before.shape
    print(f"Q shape: {q_before.shape}")

    # Step 1: Process MROPE cos/sin to get HF format
    print("\n" + "=" * 80)
    print("Step 1: Process MROPE cos/sin")
    print("=" * 80)

    mrope_section = [24, 20, 20]
    modality_num = 3
    dim = cos_official.shape[-1]

    def apply_interleaved_rope(x, modality_num, mrope_section):
        x_t = x[0].clone()
        for i, n in enumerate(mrope_section[1:], 1):
            beg_idx = i
            end_idx = n * modality_num
            x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
        return x_t

    # Get HF format cos/sin (after MROPE interleaving)
    cos_hf = torch.cat([apply_interleaved_rope(cos_official[..., : dim // 2], modality_num, mrope_section)] * 2, dim=-1)
    sin_hf = torch.cat([apply_interleaved_rope(sin_official[..., : dim // 2], modality_num, mrope_section)] * 2, dim=-1)

    print(f"cos_hf shape: {cos_hf.shape}")  # [1, 111, 128]

    # Step 2: Test dimension rearrangement
    print("\n" + "=" * 80)
    print("Step 2: Test dimension rearrangement")
    print("=" * 80)

    # Rearrange Q to interleaved format
    q_interleaved = rearrange_to_interleaved(q_before)
    print(f"q_interleaved shape: {q_interleaved.shape}")

    # Verify round-trip
    q_roundtrip = rearrange_to_noninterleaved(q_interleaved)
    roundtrip_pcc = compute_pcc(q_before, q_roundtrip)
    print(f"Round-trip PCC: {roundtrip_pcc:.6f}")

    # Also rearrange cos/sin (we need the first half only, duplicated)
    # cos_hf is [c0, c1, ..., c63, c0, c1, ..., c63]
    # We need [c0, c0, c1, c1, ..., c63, c63] for TTNN
    cos_unique = cos_hf[..., : head_dim // 2]  # [1, 111, 64]
    sin_unique = sin_hf[..., : head_dim // 2]

    cos_llama = torch.repeat_interleave(cos_unique, repeats=2, dim=-1)  # [1, 111, 128]
    sin_llama = torch.repeat_interleave(sin_unique, repeats=2, dim=-1)

    print(f"cos_llama shape: {cos_llama.shape}")

    # Step 3: Apply TTNN RoPE
    print("\n" + "=" * 80)
    print("Step 3: Apply TTNN RoPE")
    print("=" * 80)

    # Pad for tile alignment
    pad_seq = ((seq_len + 31) // 32) * 32
    padding = pad_seq - seq_len

    # Prepare cos/sin for TTNN
    cos_padded = F.pad(cos_llama.unsqueeze(0), (0, 0, 0, padding))  # [1, 1, pad_seq, 128]
    sin_padded = F.pad(sin_llama.unsqueeze(0), (0, 0, 0, padding))

    cos_tt = ttnn.from_torch(cos_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_tt = ttnn.from_torch(sin_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Get transformation matrix
    from models.demos.qwen3_tts.tt.rope import get_transformation_mat

    trans_mat = get_transformation_mat(head_dim, device)

    # Prepare Q (rearranged) for TTNN
    q_interleaved_padded = F.pad(q_interleaved, (0, 0, 0, padding))
    q_tt = ttnn.from_torch(
        q_interleaved_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Apply TTNN RoPE
    q_rotated_tt = ttnn.experimental.rotary_embedding_llama(q_tt, cos_tt, sin_tt, trans_mat, is_decode_mode=False)
    q_rotated_interleaved = ttnn.to_torch(q_rotated_tt)[:, :, :seq_len, :]

    # Step 4: Rearrange back
    print("\n" + "=" * 80)
    print("Step 4: Rearrange back and compare")
    print("=" * 80)

    q_rotated_final = rearrange_to_noninterleaved(q_rotated_interleaved)
    print(f"q_rotated_final shape: {q_rotated_final.shape}")

    # Compare with official
    pcc = compute_pcc(q_after_official, q_rotated_final)
    print(f"\nPCC(official, ttnn_with_rearrange): {pcc:.6f}")

    # Detailed comparison
    print(f"\nHead 0, position 50, first 8 dims:")
    print(f"  Official: {q_after_official[0, 0, 50, :8]}")
    print(f"  TTNN:     {q_rotated_final[0, 0, 50, :8]}")

    max_diff = (q_after_official - q_rotated_final.float()).abs().max()
    mean_diff = (q_after_official - q_rotated_final.float()).abs().mean()
    print(f"\nMax diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    if pcc > 0.99:
        print("\n*** SUCCESS: TTNN RoPE with rearrangement matches official! ***")
        return True
    else:
        print(f"\n*** MISMATCH: PCC={pcc:.4f}, need >0.99 ***")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    device = ttnn.open_device(device_id=args.device_id)
    try:
        run_test(device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
