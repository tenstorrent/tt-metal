# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Pure PyTorch Speech Tokenizer Decoder Demo

This uses our reference implementation (not qwen_tts package) to decode
RVQ codes and verify the decoder works correctly on CPU.

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_pytorch_decoder.py
"""

import time
from pathlib import Path

import torch


def load_speech_tokenizer_weights():
    """Load speech tokenizer weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("Loading speech tokenizer weights...")
    model_path = snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)

    # Strip "decoder." prefix from keys
    speech_tokenizer_dict = {}
    for k, v in raw_dict.items():
        if k.startswith("decoder."):
            speech_tokenizer_dict[k[8:]] = v
        else:
            speech_tokenizer_dict[k] = v

    print(f"  Loaded {len(speech_tokenizer_dict)} weight tensors")
    return speech_tokenizer_dict


def main():
    """Run pure PyTorch decoder test."""
    print("=" * 80)
    print("Pure PyTorch Speech Tokenizer Decoder Demo")
    print("=" * 80)

    # Load reference codes
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if not voice_clone_path.exists():
        print("ERROR: Reference codes not found.")
        return

    print("\nLoading reference codes...")
    data = torch.load(voice_clone_path)
    ref_code = data["ref_code"]  # [101, 16]
    ref_text = data.get("ref_text", "")

    print(f"  Reference code shape: {ref_code.shape}")
    print(f"  Reference text: {ref_text}")

    # Load weights
    state_dict = load_speech_tokenizer_weights()

    # Import reference implementation
    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        speech_tokenizer_decoder_forward,
    )

    # Prepare input: [batch, num_quantizers, seq_len]
    token_ids = ref_code.T.unsqueeze(0)  # [1, 16, 101]
    print(f"\nInput shape for decoder: {token_ids.shape}")

    # Create config
    config = SpeechTokenizerDecoderConfig()

    # Run decoder
    print("\nRunning PyTorch reference decoder...")
    start_time = time.time()

    try:
        audio = speech_tokenizer_decoder_forward(
            token_ids=token_ids,
            weights=state_dict,
            config=config,
        )
        decode_time = time.time() - start_time

        print(f"  Decode time: {decode_time*1000:.1f} ms")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Audio duration: {audio.shape[-1] / 24000:.2f} seconds")

        # Check audio properties
        audio_np = audio.squeeze().detach().cpu().float().numpy()
        print(f"  Audio range: [{audio_np.min():.4f}, {audio_np.max():.4f}]")
        print(f"  Audio std: {audio_np.std():.4f}")

        # Save audio
        import soundfile as sf

        output_path = "/tmp/pytorch_reference_decoder.wav"
        sf.write(output_path, audio_np, 24000)
        print(f"\n  Saved to: {output_path}")

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Reference text: {ref_text}")
        print(f"Audio duration: {audio.shape[-1] / 24000:.2f} seconds")
        print(f"Output: {output_path}")
        print()
        print("Listen to this file - if it matches the reference text,")
        print("then the PyTorch reference decoder is working correctly.")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
