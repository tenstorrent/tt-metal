# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Detailed Layer-by-Layer PCC Test

Compares TTNN output with official output at every layer to find
where errors compound.
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


def load_weights():
    """Load HuggingFace weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_path = snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    state_dict = {}
    for f in model_path.glob("*.safetensors"):
        state_dict.update(load_file(f))

    return state_dict


def run_test(device_id: int = 0):
    """Run detailed layer-by-layer PCC test."""
    print("=" * 80)
    print("Detailed Layer-by-Layer PCC Test")
    print("=" * 80)

    # Load extracted data
    hidden_states_path = Path("/tmp/qwen_tts_tensors/layer_hidden_states.pt")
    rotary_path = Path("/tmp/qwen_tts_tensors/rotary_data.pt")

    if not hidden_states_path.exists() or not rotary_path.exists():
        print("ERROR: Required tensor files not found.")
        return

    print("\nLoading extracted tensors...")
    official_states = torch.load(hidden_states_path)
    rotary_data = torch.load(rotary_path)

    # Load weights
    print("Loading weights...")
    state_dict = load_weights()

    # Open device
    print(f"\nOpening TT device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
        from models.demos.qwen3_tts.tt.rope import compute_mrope_cos_sin_for_ttnn, get_transformation_mat

        config = Qwen3TTSTalkerConfig()

        # Prepare MROPE cos/sin
        cos_mrope = rotary_data["cos"]
        sin_mrope = rotary_data["sin"]
        cos_ttnn_format, sin_ttnn_format = compute_mrope_cos_sin_for_ttnn(cos_mrope, sin_mrope)

        layer0_input = official_states["layer_0_input"]
        seq_len = layer0_input.shape[1]
        pad_seq = ((seq_len + 31) // 32) * 32
        padding = pad_seq - seq_len

        cos_padded = F.pad(cos_ttnn_format.unsqueeze(0), (0, 0, 0, padding))
        sin_padded = F.pad(sin_ttnn_format.unsqueeze(0), (0, 0, 0, padding))

        cos_tt = ttnn.from_torch(
            cos_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        sin_tt = ttnn.from_torch(
            sin_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        trans_mat = get_transformation_mat(config.head_dim, device)

        print("\n" + "=" * 80)
        print("Test 1: Layer-by-layer with OFFICIAL input at each layer")
        print("=" * 80)
        print("(Using official input for each layer, not cascading TTNN output)")
        print()

        for layer_idx in range(28):
            # Get official input for this layer
            official_input = official_states[f"layer_{layer_idx}_input"]
            official_output = official_states[f"layer_{layer_idx}_output"]

            # Prepare input
            input_padded = F.pad(official_input, (0, 0, 0, padding))
            input_4d = input_padded.unsqueeze(1)
            hidden_tt = ttnn.from_torch(input_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            # Run single layer
            layer = DecoderLayer(
                device=device,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                state_dict=state_dict,
                layer_idx=layer_idx,
                layer_prefix="talker.model",
                rms_norm_eps=config.rms_norm_eps,
                weight_dtype=ttnn.bfloat16,
            )

            output_tt = layer(hidden_tt, cos_tt, sin_tt, trans_mat, attention_mask=None)
            ttnn_output = ttnn.to_torch(output_tt).squeeze(1)[:, :seq_len, :]

            # Compare
            pcc = compute_pcc(official_output, ttnn_output)
            max_diff = (official_output - ttnn_output.float()).abs().max().item()
            mean_diff = (official_output - ttnn_output.float()).abs().mean().item()

            status = "OK" if pcc > 0.99 else "WARN" if pcc > 0.97 else "BAD"
            print(f"Layer {layer_idx:2d}: PCC={pcc:.6f} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} [{status}]")

        print("\n" + "=" * 80)
        print("Test 2: Cascading TTNN output through all layers")
        print("=" * 80)
        print("(Using layer 0 official input, then cascading TTNN output)")
        print()

        # Start with official layer 0 input
        current_input = official_states["layer_0_input"]
        input_padded = F.pad(current_input, (0, 0, 0, padding))
        input_4d = input_padded.unsqueeze(1)
        hidden_tt = ttnn.from_torch(input_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        for layer_idx in range(28):
            official_output = official_states[f"layer_{layer_idx}_output"]

            # Run layer
            layer = DecoderLayer(
                device=device,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                state_dict=state_dict,
                layer_idx=layer_idx,
                layer_prefix="talker.model",
                rms_norm_eps=config.rms_norm_eps,
                weight_dtype=ttnn.bfloat16,
            )

            hidden_tt = layer(hidden_tt, cos_tt, sin_tt, trans_mat, attention_mask=None)
            ttnn_output = ttnn.to_torch(hidden_tt).squeeze(1)[:, :seq_len, :]

            # Compare
            pcc = compute_pcc(official_output, ttnn_output)
            max_diff = (official_output - ttnn_output.float()).abs().max().item()

            status = "OK" if pcc > 0.99 else "WARN" if pcc > 0.97 else "BAD"
            print(f"Layer {layer_idx:2d}: PCC={pcc:.6f} max_diff={max_diff:.6f} [{status}]")

            # Convert back for next layer (keeps it on device for efficiency)
            # hidden_tt is already on device

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print("Test 1 shows per-layer accuracy when using official input.")
        print("Test 2 shows cumulative error when cascading TTNN output.")
        print()
        print("If Test 1 shows high PCC but Test 2 degrades, errors are compounding.")
        print("If Test 1 shows low PCC for specific layers, fix those layers first.")

    finally:
        ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    run_test(device_id=args.device_id)


if __name__ == "__main__":
    main()
