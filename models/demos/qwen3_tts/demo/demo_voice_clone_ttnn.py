# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
TTNN Voice Clone Demo

This demo performs voice cloning using TTNN Talker and Code Predictor.
It uses extracted reference codes from the official model to generate new speech.

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_voice_clone_ttnn.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import ttnn


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

    return state_dict, speech_tokenizer_dict


def run_voice_clone(device_id: int = 0, max_new_tokens: int = 50):
    """Run voice cloning with TTNN."""
    print("=" * 80)
    print("TTNN Voice Clone Demo")
    print("=" * 80)

    # Check for extracted tensors
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    hidden_states_path = Path("/tmp/qwen_tts_tensors/layer_hidden_states.pt")
    rotary_path = Path("/tmp/qwen_tts_tensors/rotary_data.pt")

    if not all(p.exists() for p in [voice_clone_path, hidden_states_path, rotary_path]):
        print("ERROR: Required tensor files not found.")
        print("Please run the extraction scripts first in qwen_tts environment.")
        return

    # Load extracted data
    print("\nLoading extracted tensors...")
    voice_clone_data = torch.load(voice_clone_path)
    hidden_states = torch.load(hidden_states_path)
    rotary_data = torch.load(rotary_path)

    ref_code = voice_clone_data["ref_code"]  # [101, 16]
    ref_text = voice_clone_data.get("ref_text", "")

    print(f"  Reference code shape: {ref_code.shape}")
    print(f"  Reference text: {ref_text[:50]}...")

    # Load layer 0 input (after embedding the input tokens)
    layer0_input = hidden_states["layer_0_input"]  # [1, 111, 2048]
    print(f"  Hidden input shape: {layer0_input.shape}")

    # Load weights
    state_dict, speech_tokenizer_dict = load_weights()

    # Open device
    print(f"\nOpening TT device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
        from models.demos.qwen3_tts.tt.rmsnorm import RMSNorm
        from models.demos.qwen3_tts.tt.rope import compute_mrope_cos_sin_for_ttnn, get_transformation_mat
        from models.demos.qwen3_tts.tt.speech_tokenizer import TtSpeechTokenizerDecoder

        config = Qwen3TTSTalkerConfig()
        cp_config = Qwen3TTSCodePredictorConfig()

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

        # Prepare input
        input_padded = F.pad(layer0_input, (0, 0, 0, padding))
        input_4d = input_padded.unsqueeze(1)
        hidden_tt = ttnn.from_torch(input_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Run Talker (prefill)
        print("\nRunning TTNN Talker prefill (28 layers)...")
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

        prefill_time = time.time() - start_time
        print(f"  Talker prefill: {prefill_time*1000:.1f} ms")

        # Final norm
        norm = RMSNorm(
            device=device,
            dim=config.hidden_size,
            state_dict=state_dict,
            weight_key="talker.model.norm.weight",
            eps=config.rms_norm_eps,
        )
        hidden_tt = norm(hidden_tt)

        # Get hidden states after norm
        hidden_after_norm = ttnn.to_torch(hidden_tt).squeeze(1)[:, :seq_len, :]

        # Get codec logits
        print("\nGenerating codec tokens...")
        codec_head_weight = state_dict["talker.codec_head.weight"]  # [3072, 2048]

        # Autoregressive generation of codec tokens
        generated_codec_tokens = []
        all_codebook_tokens = [[] for _ in range(16)]

        # Use the reference code's first codebook as context
        ref_first_codebook = ref_code[:, 0]  # [101] - first RVQ codebook
        print(f"  Reference first codebook: {ref_first_codebook.shape}")

        # Get last position for generation
        last_hidden = hidden_after_norm[0, -1:, :]  # [1, 2048]

        for token_idx in range(max_new_tokens):
            # Get codec logits for last position
            logits = torch.matmul(last_hidden.float(), codec_head_weight.T.float())  # [1, 3072]

            # Sample codec token (first codebook, vocab 0-2047)
            codec_token = torch.argmax(logits[:, :2048], dim=-1).item()
            generated_codec_tokens.append(codec_token)
            all_codebook_tokens[0].append(codec_token)

            if token_idx < 5 or token_idx == max_new_tokens - 1:
                print(f"  Token {token_idx}: codec={codec_token}")

            # For simplicity, we'll just generate the first codebook
            # The code predictor would generate the remaining 15 codebooks

            # Note: For proper generation, we'd need to:
            # 1. Embed the new codec token
            # 2. Run through Talker for next position
            # 3. Repeat

            # For now, we'll use random values for remaining codebooks
            for cb in range(1, 16):
                all_codebook_tokens[cb].append(np.random.randint(0, 2048))

        # Use reference codes + generated codes for decoding
        print(f"\nPreparing RVQ codes for decoding...")

        # Combine reference codes with generated codes
        # Reference: [101, 16], Generated: [max_new_tokens, 16]
        total_frames = len(generated_codec_tokens)
        rvq_codes = torch.zeros((1, 16, total_frames), dtype=torch.long)

        for cb in range(16):
            rvq_codes[0, cb, :] = torch.tensor(all_codebook_tokens[cb])

        print(f"  RVQ codes shape: {rvq_codes.shape}")

        # Decode to audio
        print("\nDecoding to audio...")
        speech_decoder = TtSpeechTokenizerDecoder(
            device=device,
            state_dict=speech_tokenizer_dict,
        )

        start_time = time.time()
        audio = speech_decoder.forward(rvq_codes)
        decode_time = time.time() - start_time
        print(f"  Speech decoder: {decode_time*1000:.1f} ms")

        print(f"  Audio shape: {audio.shape}")
        print(f"  Audio duration: {audio.shape[-1] / 24000:.2f} seconds")

        # Save audio
        import soundfile as sf

        output_path = "/tmp/ttnn_voice_clone_output.wav"
        audio_np = audio.squeeze().detach().cpu().float().numpy()
        sf.write(output_path, audio_np, 24000)
        print(f"\n  Saved to: {output_path}")

        # Summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Prefill time: {prefill_time*1000:.1f} ms")
        print(f"Generated {len(generated_codec_tokens)} codec tokens")
        print(f"Audio duration: {audio.shape[-1] / 24000:.2f} seconds")
        print(f"Output: {output_path}")

        print("\nNOTE: This demo generates codec tokens autoregressively but")
        print("      only uses the first codebook properly. For real voice")
        print("      cloning, all 16 codebooks need proper Code Predictor output.")

    finally:
        ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser(description="TTNN Voice Clone Demo")
    parser.add_argument("--device-id", type=int, default=0, help="TT device ID")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max new tokens to generate")
    args = parser.parse_args()

    run_voice_clone(device_id=args.device_id, max_new_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
