# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS Demo using the official **qwen-tts** PyPI package (`qwen_tts`).

This is **not** the same stack as:
  - `demo_pure_reference_tts.py` — loads `*.safetensors` and runs our in-repo
    `reference/functional.py` reimplementation (manual PyTorch forwards).

The official package still runs on **PyTorch** (like Hugging Face models); there is
no separate “non-PyTorch CPU binary” for this model. For a CPU baseline that matches
Qwen’s own code paths, use this script + `pip install qwen-tts` in a **clean** venv
with matching `torch` / `torchaudio` (and system `sox` if the package requires it).

Usage:
    # Prefer a dedicated venv: pip install qwen-tts
    python models/demos/qwen3_tts/demo/demo_qwen_tts.py --cpu \\
        --text "Hello, this is a test." --audio-output /tmp/out.wav

    python models/demos/qwen3_tts/demo/demo_qwen_tts.py \\
        --ref-audio /path/to/ref.wav --ref-text "what they said" ...
"""

import argparse

import soundfile as sf
import torch


def run_demo(
    text: str = "Hello, this is a test of the text to speech system.",
    audio_output: str = "output_official.wav",
    ref_audio_url: str = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
    ref_text: str = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    force_cpu: bool = False,
):
    """Run Qwen3-TTS using official package."""
    print("=" * 80)
    print("Qwen3-TTS Official Demo (qwen-tts package)")
    print("=" * 80)

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        raise ImportError(
            "Install the official package in a clean environment, e.g.\n"
            "  pip install qwen-tts\n"
            "Ensure torch/torchaudio versions are compatible; install system `sox` if import fails.\n"
            f"Original error: {e}"
        ) from e

    use_cuda = torch.cuda.is_available() and not force_cpu
    device_map = "cuda:0" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    print(f"\nLoading model (device_map={device_map}, dtype={dtype})...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device_map,
        dtype=dtype,
    )

    print(f"\nReference audio: {ref_audio_url}")
    print(f"Reference text: {ref_text}")
    print(f"\nText to synthesize: {text}")

    print("\nGenerating audio...")
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio_url,
        ref_text=ref_text,
    )

    print(f"\nAudio generated:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(wavs[0]) / sr:.2f} seconds")
    print(f"  Shape: {wavs[0].shape}")

    sf.write(audio_output, wavs[0], sr)
    print(f"\nAudio saved to: {audio_output}")

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Official Demo")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the text to speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--audio-output",
        type=str,
        default="output_official.wav",
        help="Output path for generated audio",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
        help="Reference audio URL or file path",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
        help="Transcript of reference audio",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU and float32 (ignore CUDA even if available)",
    )

    args = parser.parse_args()

    run_demo(
        text=args.text,
        audio_output=args.audio_output,
        ref_audio_url=args.ref_audio,
        ref_text=args.ref_text,
        force_cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
