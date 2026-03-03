# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Layer-by-Layer PCC Test: TTNN vs Official qwen_tts

Uses extracted hidden states from official qwen_tts to verify
each TTNN decoder layer produces matching outputs.
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

    a_mean = a.mean()
    b_mean = b.mean()

    a_centered = a - a_mean
    b_centered = b - b_mean

    cov = (a_centered * b_centered).sum()
    std_a = torch.sqrt((a_centered**2).sum())
    std_b = torch.sqrt((b_centered**2).sum())

    if std_a == 0 or std_b == 0:
        return 0.0

    return (cov / (std_a * std_b)).item()


def test_single_layer(device, layer_idx, hidden_states, model_weights):
    """Test a single decoder layer."""
    from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    config = Qwen3TTSTalkerConfig()

    print(f"\n--- Testing Layer {layer_idx} ---")

    # Get input/output from official model
    layer_input = hidden_states[f"layer_{layer_idx}_input"]
    layer_output = hidden_states[f"layer_{layer_idx}_output"]

    batch, seq_len, hidden = layer_input.shape
    print(f"  Input shape: {layer_input.shape}")

    # Pad to tile alignment
    pad_seq = ((seq_len + 31) // 32) * 32
    padding = pad_seq - seq_len

    # Initialize TTNN layer
    layer = DecoderLayer(
        device=device,
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        intermediate_size=config.intermediate_size,
        state_dict=model_weights,
        layer_idx=layer_idx,
        layer_prefix="talker.model",
        rms_norm_eps=config.rms_norm_eps,
        weight_dtype=ttnn.bfloat16,
    )

    # Prepare input
    input_padded = F.pad(layer_input, (0, 0, 0, padding))  # [1, pad_seq, 2048]
    input_4d = input_padded.unsqueeze(1)  # [1, 1, pad_seq, 2048]

    input_tt = ttnn.from_torch(
        input_4d,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Get RoPE tensors
    position_ids = torch.arange(pad_seq)
    cos, sin = get_rope_tensors(device, config.head_dim, pad_seq, position_ids, config.rope_theta)
    trans_mat = get_transformation_mat(config.head_dim, device)

    # Run TTNN layer
    output_tt = layer(input_tt, cos, sin, trans_mat, attention_mask=None)
    output_torch = ttnn.to_torch(output_tt).squeeze(1)[:, :seq_len, :]

    # Compare
    pcc = compute_pcc(layer_output, output_torch)
    print(f"  Official output stats: mean={layer_output.mean():.4f}, std={layer_output.std():.4f}")
    print(f"  TTNN output stats: mean={output_torch.mean():.4f}, std={output_torch.std():.4f}")
    print(f"  PCC: {pcc:.6f}")

    status = "PASS" if pcc > 0.95 else "FAIL"
    print(f"  Status: {status}")

    # Cleanup
    ttnn.deallocate(input_tt)
    ttnn.deallocate(output_tt)

    return pcc


def load_weights():
    """Load HF weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("Loading model weights...")
    model_path = snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    state_dict = {}
    for f in model_path.glob("*.safetensors"):
        state_dict.update(load_file(f))

    print(f"  Loaded {len(state_dict)} weight tensors")
    return state_dict


def run_tests(device, layers_to_test=None):
    """Run layer-by-layer PCC tests."""
    print("=" * 80)
    print("Layer-by-Layer PCC Test: TTNN vs Official qwen_tts")
    print("=" * 80)

    # Load hidden states
    hidden_states_path = Path("/tmp/qwen_tts_tensors/layer_hidden_states.pt")
    if not hidden_states_path.exists():
        print(f"ERROR: {hidden_states_path} not found.")
        print("Run extract_hidden_states.py first.")
        return

    hidden_states = torch.load(hidden_states_path)
    print(f"Loaded {len(hidden_states)} hidden state tensors")

    # Load weights
    weights = load_weights()

    # Test layers
    if layers_to_test is None:
        layers_to_test = [0, 1, 13, 27]  # Test first, second, middle, and last

    results = {}
    for layer_idx in layers_to_test:
        if f"layer_{layer_idx}_input" not in hidden_states:
            print(f"Skipping layer {layer_idx} (not found in hidden states)")
            continue

        pcc = test_single_layer(device, layer_idx, hidden_states, weights)
        results[f"layer_{layer_idx}"] = pcc

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for name, pcc in results.items():
        status = "PASS" if pcc > 0.95 else "FAIL"
        print(f"  {name}: PCC={pcc:.6f} [{status}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--layers", type=str, default="0,1,13,27", help="Comma-separated layer indices to test")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    device = ttnn.open_device(device_id=args.device_id)
    try:
        run_tests(device, layers)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
