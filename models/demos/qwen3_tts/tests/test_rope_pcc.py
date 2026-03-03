# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test TTNN RoPE against official qwen_tts Q/K outputs.

This tests if we can match the official RoPE output by:
1. Using the exact same Q/K input
2. Comparing TTNN rotary_embedding_llama output vs official output
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


def apply_official_mrope(q, cos, sin, mrope_section=(24, 20, 20), interleaved=True):
    """
    Apply MROPE exactly as official qwen_tts does.
    """

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    if interleaved:

        def apply_interleaved_rope(x, modality_num):
            x_t = x[0].clone()
            for i, n in enumerate(mrope_section[1:], 1):
                beg_idx = i
                end_idx = n * modality_num
                x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
            return x_t

        dim = cos.shape[-1]  # 128
        modality_num = len(mrope_section)  # 3

        # Process cos/sin
        cos_interleaved = torch.cat([apply_interleaved_rope(cos[..., : dim // 2], modality_num)] * 2, dim=-1)
        sin_interleaved = torch.cat([apply_interleaved_rope(sin[..., : dim // 2], modality_num)] * 2, dim=-1)

        cos_final = cos_interleaved.unsqueeze(1)  # [1, 1, seq, head_dim]
        sin_final = sin_interleaved.unsqueeze(1)
    else:
        # Non-interleaved: just combine sections from dimension 0
        section_sizes = [s * 2 for s in mrope_section]
        cos_parts = []
        sin_parts = []
        start = 0
        for i, size in enumerate(section_sizes):
            cos_parts.append(cos[i, :, :, start : start + size])
            sin_parts.append(sin[i, :, :, start : start + size])
            start += size
        cos_final = torch.cat(cos_parts, dim=-1).unsqueeze(1)
        sin_final = torch.cat(sin_parts, dim=-1).unsqueeze(1)

    q_rotated = (q * cos_final) + (rotate_half(q) * sin_final)
    return q_rotated


def convert_hf_to_llama_rope_format(cos_hf, sin_hf):
    """
    Convert HuggingFace style cos/sin to Meta/Llama style that TTNN expects.

    HF format: [c0, c1, ..., c63, c0, c1, ..., c63] (duplicated halves)
    Llama format: [c0, c0, c1, c1, ..., c63, c63] (interleaved pairs)
    """
    head_dim = cos_hf.shape[-1]
    half_dim = head_dim // 2

    cos_unique = cos_hf[..., :half_dim]
    sin_unique = sin_hf[..., :half_dim]

    cos_llama = torch.repeat_interleave(cos_unique, repeats=2, dim=-1)
    sin_llama = torch.repeat_interleave(sin_unique, repeats=2, dim=-1)

    return cos_llama, sin_llama


def run_test(device):
    """Test TTNN RoPE against official."""
    print("=" * 80)
    print("Testing TTNN RoPE Against Official qwen_tts")
    print("=" * 80)

    # Load official tensors
    data_path = Path("/tmp/qwen_tts_tensors/layer0_attention_tensors.pt")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        return

    data = torch.load(data_path)
    print("Loaded official tensors:")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    q_before = data["q_before_rope"]  # [1, 16, 111, 128]
    q_after_official = data["q_after_rope"]
    cos_official = data["cos"]  # [3, 1, 111, 128]
    sin_official = data["sin"]

    batch, num_heads, seq_len, head_dim = q_before.shape

    # First, verify that official MROPE implementation matches
    print("\n" + "=" * 80)
    print("Step 1: Verify PyTorch MROPE implementation")
    print("=" * 80)

    q_rotated_pytorch = apply_official_mrope(q_before, cos_official, sin_official)
    pcc_pytorch = compute_pcc(q_after_official, q_rotated_pytorch)
    print(f"  PCC(official, pytorch_mrope): {pcc_pytorch:.6f}")

    if pcc_pytorch > 0.9999:
        print("  PyTorch MROPE implementation matches official!")
    else:
        print("  WARNING: PyTorch MROPE doesn't match official")

    # Step 2: Convert MROPE cos/sin to format for TTNN
    print("\n" + "=" * 80)
    print("Step 2: Convert MROPE to TTNN format")
    print("=" * 80)

    # After MROPE interleaving, we get cos/sin in HF format [c0, c1, ..., c63, c0, ...]
    # Then convert to Llama format [c0, c0, c1, c1, ...]

    def apply_interleaved_rope(x, modality_num, mrope_section):
        x_t = x[0].clone()
        for i, n in enumerate(mrope_section[1:], 1):
            beg_idx = i
            end_idx = n * modality_num
            x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
        return x_t

    mrope_section = [24, 20, 20]
    modality_num = 3
    dim = cos_official.shape[-1]

    # Get HF format (after MROPE processing)
    cos_hf = torch.cat([apply_interleaved_rope(cos_official[..., : dim // 2], modality_num, mrope_section)] * 2, dim=-1)
    sin_hf = torch.cat([apply_interleaved_rope(sin_official[..., : dim // 2], modality_num, mrope_section)] * 2, dim=-1)

    print(f"  cos_hf shape: {cos_hf.shape}")  # [1, 111, 128]

    # Convert HF to Llama format
    cos_llama, sin_llama = convert_hf_to_llama_rope_format(cos_hf, sin_hf)
    print(f"  cos_llama shape: {cos_llama.shape}")

    # Step 3: Test TTNN rotary_embedding_llama
    print("\n" + "=" * 80)
    print("Step 3: Test TTNN rotary_embedding_llama")
    print("=" * 80)

    # Pad for tile alignment
    pad_seq = ((seq_len + 31) // 32) * 32
    padding = pad_seq - seq_len

    # Prepare cos/sin for TTNN: [1, 1, seq, head_dim]
    cos_padded = F.pad(cos_llama.unsqueeze(0), (0, 0, 0, padding))
    sin_padded = F.pad(sin_llama.unsqueeze(0), (0, 0, 0, padding))

    cos_tt = ttnn.from_torch(cos_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_tt = ttnn.from_torch(sin_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Get transformation matrix
    from models.demos.qwen3_tts.tt.rope import get_transformation_mat

    trans_mat = get_transformation_mat(head_dim, device)

    # Prepare Q for TTNN: [batch, heads, seq, head_dim]
    q_padded = F.pad(q_before, (0, 0, 0, padding))
    q_tt = ttnn.from_torch(q_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply TTNN RoPE
    q_rotated_tt = ttnn.experimental.rotary_embedding_llama(q_tt, cos_tt, sin_tt, trans_mat, is_decode_mode=False)
    q_rotated_ttnn = ttnn.to_torch(q_rotated_tt)[:, :, :seq_len, :]

    # Compare
    pcc_ttnn = compute_pcc(q_after_official, q_rotated_ttnn)
    print(f"  PCC(official, ttnn_rope): {pcc_ttnn:.6f}")

    # Detailed comparison
    print(f"\n  Head 0, position 50, first 8 dims:")
    print(f"    Official: {q_after_official[0, 0, 50, :8]}")
    print(f"    TTNN:     {q_rotated_ttnn[0, 0, 50, :8]}")

    max_diff = (q_after_official - q_rotated_ttnn.float()).abs().max()
    print(f"\n  Max absolute difference: {max_diff:.6f}")

    if pcc_ttnn > 0.99:
        print("\n*** TTNN RoPE matches official! ***")
    else:
        print(f"\n*** MISMATCH: PCC={pcc_ttnn:.4f}, need >0.99 ***")

        # Debug: try without conversion to see raw difference
        print("\n  Debug: Testing without HF->Llama conversion")
        cos_raw = F.pad(cos_hf.unsqueeze(0), (0, 0, 0, padding))
        sin_raw = F.pad(sin_hf.unsqueeze(0), (0, 0, 0, padding))
        cos_raw_tt = ttnn.from_torch(
            cos_raw.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        sin_raw_tt = ttnn.from_torch(
            sin_raw.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        q_tt2 = ttnn.from_torch(
            q_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        q_rotated_raw = ttnn.experimental.rotary_embedding_llama(
            q_tt2, cos_raw_tt, sin_raw_tt, trans_mat, is_decode_mode=False
        )
        q_rotated_raw_torch = ttnn.to_torch(q_rotated_raw)[:, :, :seq_len, :]

        pcc_raw = compute_pcc(q_after_official, q_rotated_raw_torch)
        print(f"    PCC without conversion: {pcc_raw:.6f}")


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
