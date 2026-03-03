# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
TTNN Audio Generation Demo

This demo generates audio using the TTNN Talker model with the RoPE fix.
It uses:
1. Official qwen_tts for speech tokenizer encoder (audio → codec tokens)
2. TTNN Talker for hidden state computation (with fixed RoPE)
3. Speech tokenizer decoder (hybrid TTNN + PyTorch) for audio output

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_ttnn_audio.py
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


def load_weights():
    """Load HuggingFace weights for both Talker and Speech Tokenizer."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("Loading model weights...")
    model_path = snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    # Load main model weights (Talker + CodePredictor)
    state_dict = {}
    for f in model_path.glob("*.safetensors"):
        state_dict.update(load_file(f))
    print(f"  Loaded {len(state_dict)} main model weight tensors")

    # Load speech tokenizer weights
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    speech_tokenizer_dict = {}
    if speech_tokenizer_path.exists():
        raw_dict = load_file(speech_tokenizer_path)
        # Strip "decoder." prefix from keys
        for k, v in raw_dict.items():
            if k.startswith("decoder."):
                speech_tokenizer_dict[k[8:]] = v  # Remove "decoder." prefix
            else:
                speech_tokenizer_dict[k] = v
        print(f"  Loaded {len(speech_tokenizer_dict)} speech tokenizer weight tensors")
    else:
        print(f"  Warning: Speech tokenizer weights not found at {speech_tokenizer_path}")

    return state_dict, speech_tokenizer_dict


def run_ttnn_from_hidden_states(device, state_dict, hidden_input, cos_tt, sin_tt, trans_mat, config):
    """Run TTNN Talker from hidden states input."""
    from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer

    batch, seq_len, hidden = hidden_input.shape
    pad_seq = ((seq_len + 31) // 32) * 32
    padding = pad_seq - seq_len

    # Prepare input
    input_padded = F.pad(hidden_input, (0, 0, 0, padding))
    input_4d = input_padded.unsqueeze(1)

    hidden_tt = ttnn.from_torch(input_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run all 28 layers
    print("Running TTNN Talker (28 layers)...")
    start_time = time.time()

    for i in range(28):
        layer = DecoderLayer(
            device=device,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            intermediate_size=config.intermediate_size,
            state_dict=state_dict,
            layer_idx=i,
            layer_prefix="talker.model",
            rms_norm_eps=config.rms_norm_eps,
            weight_dtype=ttnn.bfloat16,
        )

        hidden_tt = layer(hidden_tt, cos_tt, sin_tt, trans_mat, attention_mask=None)

    elapsed = time.time() - start_time
    print(f"  Talker forward: {elapsed*1000:.1f} ms")

    # Get output BEFORE norm (for comparison with layer_27_output)
    output_before_norm = ttnn.to_torch(hidden_tt).squeeze(1)[:, :seq_len, :]

    # Final norm
    from models.demos.qwen3_tts.tt.rmsnorm import RMSNorm

    norm = RMSNorm(
        device=device,
        dim=config.hidden_size,
        state_dict=state_dict,
        weight_key="talker.model.norm.weight",
        eps=config.rms_norm_eps,
    )
    hidden_tt = norm(hidden_tt)

    # Get output AFTER norm (for codec head)
    output_after_norm = ttnn.to_torch(hidden_tt).squeeze(1)[:, :seq_len, :]
    return output_before_norm, output_after_norm


def run_demo(device_id: int = 0):
    """Run the TTNN audio generation demo."""
    print("=" * 80)
    print("TTNN Audio Generation Demo")
    print("=" * 80)

    # Check for extracted tensors
    hidden_states_path = Path("/tmp/qwen_tts_tensors/layer_hidden_states.pt")
    rotary_path = Path("/tmp/qwen_tts_tensors/rotary_data.pt")

    if not hidden_states_path.exists() or not rotary_path.exists():
        print("ERROR: Required tensor files not found.")
        print("Please run the extraction scripts first in qwen_tts environment:")
        print("  1. python models/demos/qwen3_tts/demo/extract_hidden_states.py")
        print("  2. python models/demos/qwen3_tts/demo/extract_rope.py")
        return

    # Load extracted data
    print("\nLoading extracted tensors...")
    hidden_states = torch.load(hidden_states_path)
    rotary_data = torch.load(rotary_path)

    layer0_input = hidden_states["layer_0_input"]
    layer27_output_official = hidden_states["layer_27_output"]

    print(f"  Hidden input shape: {layer0_input.shape}")

    # Load weights
    state_dict, speech_tokenizer_dict = load_weights()

    # Open device
    print(f"\nOpening TT device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
        from models.demos.qwen3_tts.tt.rope import compute_mrope_cos_sin_for_ttnn, get_transformation_mat

        config = Qwen3TTSTalkerConfig()

        # Prepare MROPE cos/sin
        cos_mrope = rotary_data["cos"]
        sin_mrope = rotary_data["sin"]
        cos_ttnn_format, sin_ttnn_format = compute_mrope_cos_sin_for_ttnn(cos_mrope, sin_mrope)

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

        # Run TTNN Talker
        hidden_before_norm, hidden_after_norm = run_ttnn_from_hidden_states(
            device, state_dict, layer0_input, cos_tt, sin_tt, trans_mat, config
        )

        # Compare with official (layer_27_output is BEFORE norm)
        pcc = compute_pcc(layer27_output_official, hidden_before_norm)
        print(f"\nTTNN vs Official PCC: {pcc:.6f}")

        # Get codec logits (using hidden state AFTER norm)
        print("\nGetting codec logits...")
        codec_head_key = "talker.codec_head.weight"
        codec_head_weight = state_dict[codec_head_key]  # [3072, 2048]

        # Manual linear projection
        logits = torch.matmul(hidden_after_norm.float(), codec_head_weight.T.float())  # [1, seq, 3072]
        print(f"  Codec logits shape: {logits.shape}")

        # Get last token prediction
        last_logits = logits[0, -1, :]  # [3072]
        predicted_token = torch.argmax(last_logits).item()
        print(f"  Predicted next token: {predicted_token}")

        # Compare with official if available
        if "codec_head_predicted_token" in hidden_states:
            official_token = hidden_states["codec_head_predicted_token"]
            print(f"  Official predicted token: {official_token}")
            if predicted_token == official_token:
                print("  *** Tokens MATCH! ***")
            else:
                print(f"  *** Token mismatch (diff={abs(predicted_token - official_token)}) ***")

        # Run Code Predictor
        print("\n" + "=" * 80)
        print("Running Code Predictor")
        print("=" * 80)

        from models.demos.qwen3_tts.tt.code_predictor import CodePredictor
        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig
        from models.demos.qwen3_tts.tt.rope import get_rope_tensors

        cp_config = Qwen3TTSCodePredictorConfig()

        # Prepare hidden states for code predictor (use normed output)
        hidden_4d = hidden_after_norm.unsqueeze(1)  # [1, 1, seq, 2048]
        hidden_4d_padded = F.pad(hidden_4d, (0, 0, 0, padding))

        hidden_tt_cp = ttnn.from_torch(
            hidden_4d_padded.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # Get RoPE for code predictor (standard RoPE, not MROPE)
        position_ids = torch.arange(pad_seq)
        cp_cos, cp_sin = get_rope_tensors(device, cp_config.head_dim, pad_seq, position_ids, cp_config.rope_theta)
        cp_trans_mat = get_transformation_mat(cp_config.head_dim, device)

        code_predictor = CodePredictor(
            device=device,
            config=cp_config,
            talker_hidden_size=config.hidden_size,  # 2048
            state_dict=state_dict,
        )

        start_time = time.time()
        cp_logits_list = code_predictor(hidden_tt_cp, cp_cos, cp_sin, cp_trans_mat, attention_mask=None)
        cp_elapsed = time.time() - start_time
        print(f"Code Predictor forward: {cp_elapsed*1000:.1f} ms")

        # Sample from each head
        print(f"\nSampled tokens from Code Predictor (last position):")
        all_codebook_tokens = []
        for i, logits_tt in enumerate(cp_logits_list[:5]):  # Show first 5
            logits_torch = ttnn.to_torch(logits_tt).squeeze(1)[:, seq_len - 1, :]  # [1, 2048]
            token = torch.argmax(logits_torch, dim=-1).item()
            all_codebook_tokens.append(token)
            print(f"  Codebook {i+1}: token={token}")

        # Try to decode to audio
        print("\n" + "=" * 80)
        print("Decoding to Audio")
        print("=" * 80)

        try:
            from models.demos.qwen3_tts.tt.speech_tokenizer import TtSpeechTokenizerDecoder

            # Create dummy RVQ codes for testing (would come from generation loop)
            # For now, use the predicted tokens as first codebook
            num_frames = 128  # ~10 seconds at 12.5 Hz
            rvq_codes = torch.randint(0, 2048, (1, 16, num_frames))  # [batch, codebooks, frames]

            # Use predicted token for first frame
            rvq_codes[0, 0, 0] = predicted_token if predicted_token < 2048 else 0

            print(f"  RVQ codes shape: {rvq_codes.shape}")

            speech_decoder = TtSpeechTokenizerDecoder(
                device=device,
                state_dict=speech_tokenizer_dict,
            )

            print("  Decoding RVQ codes to audio...")
            start_time = time.time()
            audio = speech_decoder.forward(rvq_codes)
            decode_elapsed = time.time() - start_time
            print(f"  Speech decoder: {decode_elapsed*1000:.1f} ms")

            print(f"  Audio shape: {audio.shape}")
            print(f"  Audio duration: {audio.shape[-1] / 24000:.2f} seconds")

            # Save audio
            import soundfile as sf

            output_path = "/tmp/ttnn_output.wav"
            audio_np = audio.squeeze().detach().cpu().float().numpy()
            sf.write(output_path, audio_np, 24000)
            print(f"\n  Saved to: {output_path}")

        except Exception as e:
            print(f"  Audio decoding failed: {e}")
            import traceback

            traceback.print_exc()

        # Summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"TTNN Talker PCC vs Official: {pcc:.6f}")
        print(f"Predicted codec token: {predicted_token}")

        if pcc > 0.97:
            print("\n*** TTNN model output is close to official! ***")
            print("Note: Full audio quality depends on autoregressive generation loop.")

    finally:
        ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser(description="TTNN Audio Generation Demo")
    parser.add_argument("--device-id", type=int, default=0, help="TT device ID")
    args = parser.parse_args()

    run_demo(device_id=args.device_id)


if __name__ == "__main__":
    main()
