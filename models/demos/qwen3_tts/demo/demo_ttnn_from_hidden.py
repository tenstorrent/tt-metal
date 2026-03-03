# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
TTNN Demo using extracted hidden states from official qwen_tts.

This demo bypasses the embedding/prompt construction complexity by starting
from the extracted hidden states and running TTNN decoder layers.

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_ttnn_from_hidden.py
"""

import argparse
import time
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


def load_hf_weights(model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base") -> dict:
    """Load weights from HuggingFace."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print(f"Loading model weights from: {model_id}")
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    state_dict = {}
    for f in model_path.glob("*.safetensors"):
        state_dict.update(load_file(f))

    print(f"  Loaded {len(state_dict)} weight tensors")
    return state_dict


def run_demo(device_id: int = 0):
    """Run TTNN demo using extracted hidden states."""
    print("=" * 80)
    print("TTNN Demo with Extracted Hidden States")
    print("=" * 80)

    # Load extracted hidden states
    hidden_states_path = Path("/tmp/qwen_tts_tensors/layer_hidden_states.pt")
    if not hidden_states_path.exists():
        print(f"ERROR: {hidden_states_path} not found.")
        print("Run extract_hidden_states.py in qwen_tts environment first.")
        return

    hidden_states = torch.load(hidden_states_path)
    print(f"Loaded {len(hidden_states)} hidden state tensors")

    # Get the input to layer 0 (this is after embedding)
    layer0_input = hidden_states["layer_0_input"]  # [1, 111, 2048]
    layer27_output = hidden_states["layer_27_output"]  # [1, 111, 2048] - official final output

    print(f"\nUsing layer_0_input as starting point: {layer0_input.shape}")

    # Load weights
    state_dict = load_hf_weights()

    # Open device
    print(f"\nOpening TT device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        # Initialize TTNN model
        print("\nInitializing TTNN model...")
        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
        from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

        talker_config = Qwen3TTSTalkerConfig()
        cp_config = Qwen3TTSCodePredictorConfig()

        model = Qwen3TTS(
            device=device,
            state_dict=state_dict,
            talker_config=talker_config,
            code_predictor_config=cp_config,
        )
        print("Model initialized!")

        # Prepare input
        batch, seq_len, hidden = layer0_input.shape
        pad_seq = ((seq_len + 31) // 32) * 32
        padding = pad_seq - seq_len

        input_padded = F.pad(layer0_input, (0, 0, 0, padding))
        input_4d = input_padded.unsqueeze(1)  # [1, 1, pad_seq, 2048]

        print(f"\nInput (padded): {input_4d.shape}")

        # Convert to TTNN
        hidden_tt = ttnn.from_torch(
            input_4d,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Get RoPE tensors
        position_ids = torch.arange(pad_seq)
        talker_cos, talker_sin = get_rope_tensors(
            device, talker_config.head_dim, pad_seq, position_ids, talker_config.rope_theta
        )
        cp_cos, cp_sin = get_rope_tensors(device, cp_config.head_dim, pad_seq, position_ids, cp_config.rope_theta)
        talker_trans_mat = get_transformation_mat(talker_config.head_dim, device)
        cp_trans_mat = get_transformation_mat(cp_config.head_dim, device)

        # Run Talker layers (starting from hidden states)
        print("\n" + "=" * 80)
        print("Running TTNN Talker Layers")
        print("=" * 80)

        start_time = time.time()

        # Run through talker layers, tracking PCC at each step
        print("Layer-by-layer PCC tracking:")
        for i, layer in enumerate(model.talker.layers):
            hidden_tt = layer(hidden_tt, talker_cos, talker_sin, talker_trans_mat, attention_mask=None)

            # Check PCC at key layers
            if i in [0, 1, 5, 10, 15, 20, 27] or i == len(model.talker.layers) - 1:
                temp_torch = ttnn.to_torch(hidden_tt).squeeze(1)[:, :seq_len, :]
                official_key = f"layer_{i}_output"
                if official_key in hidden_states:
                    official = hidden_states[official_key]
                    pcc = compute_pcc(official, temp_torch)
                    print(f"  After layer {i}: PCC={pcc:.6f}")

        # Final norm
        hidden_tt = model.talker.norm(hidden_tt)

        ttnn.synchronize_device(device)
        talker_time = time.time() - start_time

        # Convert to torch and compare with official
        hidden_torch = ttnn.to_torch(hidden_tt).squeeze(1)[:, :seq_len, :]  # [1, 111, 2048]

        pcc = compute_pcc(layer27_output, hidden_torch)
        print(f"Talker time: {talker_time*1000:.2f} ms")
        print(f"PCC vs official (after all layers + norm): {pcc:.6f}")

        # Get codec logits
        print("\n" + "=" * 80)
        print("Getting Codec Logits")
        print("=" * 80)

        codec_logits = model.talker.get_codec_logits(hidden_tt)
        codec_logits_torch = ttnn.to_torch(codec_logits).squeeze(1)[:, :seq_len, :]

        print(f"Codec logits shape: {codec_logits_torch.shape}")
        print(f"Codec logits stats: mean={codec_logits_torch.mean():.4f}, std={codec_logits_torch.std():.4f}")

        # Get predicted tokens from last position
        last_logits = codec_logits_torch[:, -1, :]  # [1, 3072]
        predicted_token = torch.argmax(last_logits, dim=-1)
        print(f"Predicted next token: {predicted_token.item()}")

        # Run Code Predictor
        print("\n" + "=" * 80)
        print("Running Code Predictor")
        print("=" * 80)

        start_time = time.time()
        cp_logits_list = model.code_predictor(hidden_tt, cp_cos, cp_sin, cp_trans_mat, attention_mask=None)
        ttnn.synchronize_device(device)
        cp_time = time.time() - start_time

        print(f"Code Predictor time: {cp_time*1000:.2f} ms")
        print(f"Number of LM heads: {len(cp_logits_list)}")

        # Sample from each head
        print("\nSampled tokens from last position:")
        for i, logits in enumerate(cp_logits_list[:5]):  # Show first 5
            logits_torch = ttnn.to_torch(logits).squeeze(1)[:, -1, :]  # [1, 2048]
            token = torch.argmax(logits_torch, dim=-1)
            print(f"  Head {i}: token={token.item()}")

        # Summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Talker PCC vs official: {pcc:.6f}")
        print(f"Talker inference time: {talker_time*1000:.2f} ms")
        print(f"Code Predictor time: {cp_time*1000:.2f} ms")
        print(f"Total model time: {(talker_time + cp_time)*1000:.2f} ms")

        if pcc > 0.99:
            print("\n*** TTNN model output matches official implementation! ***")
        else:
            print(f"\n*** PCC is {pcc:.4f}, target is >0.99 ***")

    finally:
        ttnn.close_device(device)

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="TTNN Demo with Extracted Hidden States")
    parser.add_argument("--device-id", type=int, default=0, help="TTNN device ID")
    args = parser.parse_args()

    run_demo(device_id=args.device_id)


if __name__ == "__main__":
    main()
