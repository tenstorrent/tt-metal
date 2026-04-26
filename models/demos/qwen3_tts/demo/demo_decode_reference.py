# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Decode Reference Audio Demo

This demo decodes the extracted reference RVQ codes to verify the
speech tokenizer decoder works correctly. If this produces good
audio, then the decoder is working and the issue is in generation.

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_decode_reference.py
"""

import argparse
import time
from pathlib import Path

import torch

import ttnn


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


def run_demo(device_id: int = 0):
    """Decode reference RVQ codes to verify decoder works."""
    print("=" * 80)
    print("Decode Reference Audio Demo")
    print("=" * 80)

    # Load reference codes
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if not voice_clone_path.exists():
        print("ERROR: Reference codes not found.")
        print("Please run the extraction scripts first.")
        return

    print("\nLoading reference codes...")
    data = torch.load(voice_clone_path)
    ref_code = data["ref_code"]  # [101, 16]
    ref_text = data.get("ref_text", "")

    print(f"  Reference code shape: {ref_code.shape}")
    print(f"  Reference text: {ref_text[:60]}...")

    # Prepare RVQ codes for decoder: [batch, num_quantizers, seq_len]
    rvq_codes = ref_code.T.unsqueeze(0)  # [1, 16, 101]
    print(f"  RVQ codes for decoder: {rvq_codes.shape}")

    # Load weights
    speech_tokenizer_dict = load_speech_tokenizer_weights()

    # Open device
    print(f"\nOpening TT device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        from models.demos.qwen3_tts.tt.speech_tokenizer import TtSpeechTokenizerDecoder

        print("\nInitializing speech decoder...")
        speech_decoder = TtSpeechTokenizerDecoder(
            device=device,
            state_dict=speech_tokenizer_dict,
        )

        print("\nDecoding reference RVQ codes...")
        start_time = time.time()
        audio = speech_decoder.forward(rvq_codes)
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

        output_path = "/tmp/ttnn_reference_decoded.wav"
        sf.write(output_path, audio_np, 24000)
        print(f"\n  Saved to: {output_path}")

        # Summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Decoded {ref_code.shape[0]} frames of reference audio")
        print(f"Audio duration: {audio.shape[-1] / 24000:.2f} seconds")
        print(f"Output: {output_path}")
        print()
        print("If this audio sounds like the reference text:")
        print(f'  "{ref_text}"')
        print("Then the decoder is working and the issue is in token generation.")

    finally:
        ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser(description="Decode Reference Audio Demo")
    parser.add_argument("--device-id", type=int, default=0, help="TT device ID")
    args = parser.parse_args()

    run_demo(device_id=args.device_id)


if __name__ == "__main__":
    main()
