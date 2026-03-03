# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS Audio Generation Script

This script generates audio from text using the qwen-tts package.
It serves as a reference implementation and produces WAV files for testing.

Usage:
    # Simple text-to-speech (uses default voice)
    python models/demos/qwen3_tts/demo/generate_audio.py --text "Hello, world!"

    # Voice cloning with reference audio
    python models/demos/qwen3_tts/demo/generate_audio.py --text "Hello" --ref-audio sample.wav --ref-text "Reference text"

    # Specify output file
    python models/demos/qwen3_tts/demo/generate_audio.py --text "Hello" --output hello.wav

Requirements:
    pip install -U qwen-tts
"""

import argparse
import time

import soundfile as sf
import torch


def generate_audio_qwen_tts(
    text: str,
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    output_path: str = "output.wav",
    device: str = "cpu",
    ref_audio: str = None,
    ref_text: str = None,
    language: str = "English",
):
    """
    Generate audio from text using qwen-tts package.

    Args:
        text: Input text to synthesize
        model_id: HuggingFace model ID
        output_path: Path to save the output WAV file
        device: Device to run inference on (cpu or cuda:0)
        ref_audio: Optional reference audio for voice cloning
        ref_text: Transcript of the reference audio (required if ref_audio is provided)
        language: Target language (English, Chinese, etc.)
    """
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        raise ImportError("Please install qwen-tts: pip install -U qwen-tts")

    print(f"=" * 60)
    print(f"Qwen3-TTS Audio Generation")
    print(f"=" * 60)
    print(f"Model: {model_id}")
    print(f"Text: {text}")
    print(f"Language: {language}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    if ref_audio:
        print(f"Reference Audio: {ref_audio}")
        print(f"Reference Text: {ref_text}")
    print(f"=" * 60)

    # Load the model
    print("\nLoading model...")
    start_time = time.time()

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
    )
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")

    # Generate audio
    print("\nGenerating audio...")
    gen_start = time.time()

    if ref_audio and ref_text:
        # Voice cloning mode
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
    else:
        # Simple TTS mode (uses model's default voice if available)
        # Note: The Base model requires voice cloning, so we'll use a sample
        print("Note: Base model requires reference audio for voice cloning.")
        print("Using sample reference audio from Qwen...")

        # Use the sample reference from Qwen's documentation
        sample_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
        sample_ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=sample_ref_audio,
            ref_text=sample_ref_text,
        )

    gen_time = time.time() - gen_start
    print(f"Audio generated in {gen_time:.2f}s")

    # Save audio
    audio = wavs[0] if isinstance(wavs, list) else wavs
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    sf.write(output_path, audio, sr)
    duration = len(audio) / sr

    print(f"\n{'=' * 60}")
    print(f"Audio saved to: {output_path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration:.2f}s")
    print(f"{'=' * 60}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate audio with Qwen3-TTS")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the Qwen 3 text to speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output WAV file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda:0)",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio file for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of the reference audio",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Target language (English, Chinese, Japanese, etc.)",
    )

    args = parser.parse_args()

    generate_audio_qwen_tts(
        text=args.text,
        model_id=args.model_id,
        output_path=args.output,
        device=args.device,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        language=args.language,
    )


if __name__ == "__main__":
    main()
