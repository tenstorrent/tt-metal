# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test TTNN layers with official MROPE cos/sin tensors.

This tests if using the exact cos/sin from official qwen_tts improves PCC.
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


def convert_mrope_for_ttnn(cos_mrope, sin_mrope, mrope_section=(24, 20, 20), interleaved=True):
    """
    Convert MROPE cos/sin [3, 1, seq, head_dim] to TTNN format [1, 1, seq, head_dim].

    The official implementation with interleaved=True does:
    - cos is shape [3, 1, seq, head_dim] with different frequencies per section
    - For each section i, use cos[i % 3]

    Since for text, all 3 position dimensions are the same, but the frequencies differ.
    """
    # cos_mrope: [3, batch, seq, head_dim]
    # For interleaved mode, we need to reconstruct the cos/sin

    if interleaved:
        # The interleaved mode takes dimension chunks and assigns them to different modalities
        # mrope_section = [24, 20, 20] means:
        # - dims 0-47 (24*2) use cos[0]
        # - dims 48-87 (20*2) use cos[1]
        # - dims 88-127 (20*2) use cos[2]

        modality_num = len(mrope_section)
        section_sizes = [s * 2 for s in mrope_section]  # [48, 40, 40]

        # Reconstruct the interleaved cos/sin
        cos_parts = []
        sin_parts = []

        start = 0
        for i, size in enumerate(section_sizes):
            # For each section, use the corresponding modality
            cos_parts.append(cos_mrope[i, :, :, start : start + size])
            sin_parts.append(sin_mrope[i, :, :, start : start + size])
            start += size

        # Concatenate along head_dim
        cos_combined = torch.cat(cos_parts, dim=-1)  # [1, seq, head_dim]
        sin_combined = torch.cat(sin_parts, dim=-1)

        # Add batch dim: [1, 1, seq, head_dim]
        cos_combined = cos_combined.unsqueeze(0)
        sin_combined = sin_combined.unsqueeze(0)

        return cos_combined, sin_combined
    else:
        # Non-interleaved: just use the first modality
        return cos_mrope[0:1, :, :, :], sin_mrope[0:1, :, :, :]


def run_test(device):
    """Run test with official MROPE tensors."""
    print("=" * 80)
    print("Test TTNN Layers with Official MROPE Cos/Sin")
    print("=" * 80)

    # Load hidden states
    hidden_states_path = Path("/tmp/qwen_tts_tensors/layer_hidden_states.pt")
    if not hidden_states_path.exists():
        print(f"ERROR: {hidden_states_path} not found.")
        return

    hidden_states = torch.load(hidden_states_path)
    print(f"Loaded {len(hidden_states)} hidden state tensors")

    # Load official MROPE tensors
    rotary_path = Path("/tmp/qwen_tts_tensors/rotary_data.pt")
    if not rotary_path.exists():
        print(f"ERROR: {rotary_path} not found. Run extract_rope.py first.")
        return

    rotary_data = torch.load(rotary_path)
    print(f"Loaded MROPE tensors")
    print(f"  position_ids: {rotary_data['position_ids'].shape}")
    print(f"  cos: {rotary_data['cos'].shape}")
    print(f"  sin: {rotary_data['sin'].shape}")

    # Convert MROPE to TTNN format
    cos_mrope = rotary_data["cos"]  # [3, 1, 111, 128]
    sin_mrope = rotary_data["sin"]

    cos_ttnn_format, sin_ttnn_format = convert_mrope_for_ttnn(cos_mrope, sin_mrope)
    print(f"  Converted cos: {cos_ttnn_format.shape}")
    print(f"  Converted sin: {sin_ttnn_format.shape}")

    # Also generate standard RoPE for comparison
    from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies

    config = {
        "hidden_size": 2048,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 6144,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
    }

    seq_len = 111
    pad_seq = ((seq_len + 31) // 32) * 32

    # Standard RoPE
    cos_std, sin_std = compute_rope_frequencies(config["head_dim"], pad_seq, config["rope_theta"])
    position_ids = torch.arange(seq_len)
    cos_std_gathered = cos_std[position_ids].unsqueeze(0).unsqueeze(0)  # [1, 1, 111, 128]
    sin_std_gathered = sin_std[position_ids].unsqueeze(0).unsqueeze(0)

    # Compare standard RoPE vs official MROPE
    print("\nComparing RoPE frequencies:")
    pcc_cos = compute_pcc(cos_std_gathered, cos_ttnn_format)
    pcc_sin = compute_pcc(sin_std_gathered, sin_ttnn_format)
    print(f"  PCC(standard_cos, mrope_cos): {pcc_cos:.6f}")
    print(f"  PCC(standard_sin, mrope_sin): {pcc_sin:.6f}")

    # Show first few values
    print(f"\n  Standard cos[0,0,0,:8]: {cos_std_gathered[0,0,0,:8]}")
    print(f"  MROPE cos[0,0,0,:8]:    {cos_ttnn_format[0,0,0,:8]}")
    print(f"  Standard sin[0,0,0,:8]: {sin_std_gathered[0,0,0,:8]}")
    print(f"  MROPE sin[0,0,0,:8]:    {sin_ttnn_format[0,0,0,:8]}")

    # Load weights
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("\nLoading model weights...")
    model_path = snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    state_dict = {}
    for f in model_path.glob("*.safetensors"):
        state_dict.update(load_file(f))
    print(f"  Loaded {len(state_dict)} weight tensors")

    # Pad cos/sin for tile alignment
    padding = pad_seq - seq_len
    cos_padded = F.pad(cos_ttnn_format, (0, 0, 0, padding))  # [1, 1, pad_seq, 128]
    sin_padded = F.pad(sin_ttnn_format, (0, 0, 0, padding))

    # Convert to TTNN
    cos_tt = ttnn.from_torch(cos_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_tt = ttnn.from_torch(sin_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Get transformation matrix
    from models.demos.qwen3_tts.tt.rope import get_transformation_mat

    trans_mat = get_transformation_mat(config["head_dim"], device)

    # Test single layer with official cos/sin
    print("\n" + "=" * 80)
    print("Testing Single Layer with Official MROPE")
    print("=" * 80)

    from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig

    talker_config = Qwen3TTSTalkerConfig()

    # Test layer 0
    layer_idx = 0
    layer_input = hidden_states[f"layer_{layer_idx}_input"]
    layer_output = hidden_states[f"layer_{layer_idx}_output"]

    input_padded = F.pad(layer_input, (0, 0, 0, padding))
    input_4d = input_padded.unsqueeze(1)

    input_tt = ttnn.from_torch(input_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    layer = DecoderLayer(
        device=device,
        hidden_size=talker_config.hidden_size,
        num_heads=talker_config.num_attention_heads,
        num_kv_heads=talker_config.num_key_value_heads,
        head_dim=talker_config.head_dim,
        intermediate_size=talker_config.intermediate_size,
        state_dict=state_dict,
        layer_idx=layer_idx,
        layer_prefix="talker.model",
        rms_norm_eps=talker_config.rms_norm_eps,
        weight_dtype=ttnn.bfloat16,
    )

    output_tt = layer(input_tt, cos_tt, sin_tt, trans_mat, attention_mask=None)
    output_torch = ttnn.to_torch(output_tt).squeeze(1)[:, :seq_len, :]

    pcc_layer0_mrope = compute_pcc(layer_output, output_torch)
    print(f"  Layer 0 with official MROPE: PCC={pcc_layer0_mrope:.6f}")

    # Compare with standard RoPE
    cos_std_padded = F.pad(cos_std_gathered, (0, 0, 0, padding))
    sin_std_padded = F.pad(sin_std_gathered, (0, 0, 0, padding))
    cos_std_tt = ttnn.from_torch(
        cos_std_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    sin_std_tt = ttnn.from_torch(
        sin_std_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn.deallocate(output_tt)
    ttnn.deallocate(input_tt)

    input_tt = ttnn.from_torch(input_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_std_tt = layer(input_tt, cos_std_tt, sin_std_tt, trans_mat, attention_mask=None)
    output_std_torch = ttnn.to_torch(output_std_tt).squeeze(1)[:, :seq_len, :]

    pcc_layer0_std = compute_pcc(layer_output, output_std_torch)
    print(f"  Layer 0 with standard RoPE: PCC={pcc_layer0_std:.6f}")

    print(f"\n  Improvement from MROPE: {pcc_layer0_mrope - pcc_layer0_std:.6f}")

    if pcc_layer0_mrope > pcc_layer0_std + 0.01:
        print("\n*** MROPE significantly improves PCC! ***")
    elif pcc_layer0_mrope > pcc_layer0_std:
        print("\n*** MROPE slightly improves PCC ***")
    else:
        print("\n*** MROPE did NOT improve PCC (issue is elsewhere) ***")

    # Run full model with official MROPE
    print("\n" + "=" * 80)
    print("Running Full Model with Official MROPE")
    print("=" * 80)

    # Get fresh input
    layer0_input = hidden_states["layer_0_input"]
    input_padded = F.pad(layer0_input, (0, 0, 0, padding))
    input_4d = input_padded.unsqueeze(1)

    hidden_tt = ttnn.from_torch(input_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    print("Running layers:")
    for i in range(28):
        layer = DecoderLayer(
            device=device,
            hidden_size=talker_config.hidden_size,
            num_heads=talker_config.num_attention_heads,
            num_kv_heads=talker_config.num_key_value_heads,
            head_dim=talker_config.head_dim,
            intermediate_size=talker_config.intermediate_size,
            state_dict=state_dict,
            layer_idx=i,
            layer_prefix="talker.model",
            rms_norm_eps=talker_config.rms_norm_eps,
            weight_dtype=ttnn.bfloat16,
        )

        hidden_tt = layer(hidden_tt, cos_tt, sin_tt, trans_mat, attention_mask=None)

        if i in [0, 1, 5, 10, 15, 20, 27]:
            temp_torch = ttnn.to_torch(hidden_tt).squeeze(1)[:, :seq_len, :]
            official_key = f"layer_{i}_output"
            if official_key in hidden_states:
                official = hidden_states[official_key]
                pcc = compute_pcc(official, temp_torch)
                print(f"  After layer {i}: PCC={pcc:.6f}")

    # Final
    final_output = ttnn.to_torch(hidden_tt).squeeze(1)[:, :seq_len, :]
    official_final = hidden_states["layer_27_output"]
    final_pcc = compute_pcc(official_final, final_output)

    print(f"\nFinal PCC (after all layers): {final_pcc:.6f}")

    if final_pcc > 0.9:
        print("*** MROPE FIXED THE ISSUE! ***")
    elif final_pcc > 0.7:
        print("*** MROPE improved PCC but issue persists ***")
    else:
        print("*** MROPE did not fix the issue ***")


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
