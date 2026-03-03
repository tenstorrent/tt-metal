# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Simple demo: Generate audio from saved RVQ codes.

Uses the reference decoder to generate audio from official qwen_tts codes.
Listen to the output to verify quality.
"""

from pathlib import Path

import soundfile as sf
import torch


def main():
    print("=" * 80)
    print("Audio Generation Demo")
    print("=" * 80)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        speech_tokenizer_decoder_forward,
    )

    # Load saved codes from official qwen_tts encoder
    print("\n1. Loading codes...")
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if not voice_clone_path.exists():
        print(f"ERROR: {voice_clone_path} not found!")
        return

    data = torch.load(voice_clone_path)
    codes = data["ref_code"].T.unsqueeze(0)  # [1, 16, 101]
    print(f"   Codes shape: {codes.shape}")
    print(f"   Code range: [{codes.min()}, {codes.max()}]")

    # Load decoder weights
    print("\n2. Loading decoder weights...")
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)
    decoder_weights = {k[8:]: v for k, v in raw_dict.items() if k.startswith("decoder.")}
    print(f"   Loaded {len(decoder_weights)} weights")

    # Generate audio
    print("\n3. Generating audio...")
    config = SpeechTokenizerDecoderConfig()
    with torch.no_grad():
        audio = speech_tokenizer_decoder_forward(codes, decoder_weights, config)

    # Audio stats
    audio_np = audio.squeeze().cpu().float().numpy()
    duration = len(audio_np) / 24000
    print(f"   Audio shape: {audio.shape}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Range: [{audio.min():.4f}, {audio.max():.4f}]")
    print(f"   Std: {audio.std():.4f}")

    # Save
    output_path = "/tmp/reference_generated_audio.wav"
    sf.write(output_path, audio_np, 24000)
    print(f"\n4. Saved: {output_path}")

    print("\n" + "=" * 80)
    print("Listen to the audio file to verify quality:")
    print(f"  {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
