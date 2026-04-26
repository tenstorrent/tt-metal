# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Hybrid Demo: Official qwen_tts infrastructure + TTNN model components

This script:
1. Uses official qwen_tts for voice cloning prompt creation
2. Captures intermediate tensors for comparison
3. Can run model components through TTNN for PCC testing

Usage (run from /tmp/qwen_tts_env):
    source /tmp/qwen_tts_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_hybrid.py --extract-tensors

Then run in tt-metal env:
    source python_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_hybrid.py --compare-ttnn
"""

import argparse
from pathlib import Path

import torch


def extract_tensors_from_official():
    """Extract intermediate tensors from official qwen_tts implementation."""
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    print("=" * 80)
    print("Extracting tensors from official qwen_tts")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    print("Model loaded!")

    # Reference audio
    ref_audio = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    text = "Hello, this is a test of the text to speech system."

    # Create voice clone prompt to get intermediate tensors
    print("\nCreating voice clone prompt...")
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    # Extract what we can from the prompt
    print(f"Prompt items type: {type(prompt_items)}")
    if hasattr(prompt_items, "__dict__"):
        print(f"Prompt items attributes: {prompt_items.__dict__.keys()}")

    # Generate audio to capture intermediate states
    print("\nGenerating audio...")
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    print(f"Audio shape: {wavs[0].shape}, SR: {sr}")

    # Save the audio
    sf.write("/tmp/hybrid_official_output.wav", wavs[0], sr)
    print("Saved to /tmp/hybrid_official_output.wav")

    # Try to extract model internals
    print("\n" + "=" * 80)
    print("Model structure inspection")
    print("=" * 80)

    # Check the model's internal components
    if hasattr(model, "model"):
        internal = model.model
        print(f"Internal model type: {type(internal)}")
        if hasattr(internal, "named_modules"):
            print("\nNamed modules (first 20):")
            for i, (name, module) in enumerate(internal.named_modules()):
                if i < 20:
                    print(f"  {name}: {type(module).__name__}")

    # Save model config for reference
    output_dir = Path("/tmp/qwen_tts_tensors")
    output_dir.mkdir(exist_ok=True)

    if hasattr(model, "config"):
        print(f"\nModel config: {model.config}")
        torch.save({"config": str(model.config)}, output_dir / "config.pt")

    print(f"\nTensors saved to {output_dir}")


def compare_with_ttnn():
    """Compare TTNN implementation with reference tensors."""
    print("=" * 80)
    print("Comparing TTNN implementation")
    print("=" * 80)

    # Load saved tensors
    tensor_dir = Path("/tmp/qwen_tts_tensors")
    if not tensor_dir.exists():
        print(f"ERROR: {tensor_dir} not found. Run --extract-tensors first.")
        return

    # TODO: Load and compare with TTNN
    print("TODO: Implement TTNN comparison")


def run_full_pipeline():
    """Run the full pipeline using official qwen_tts."""
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    print("=" * 80)
    print("Full Pipeline with Official qwen_tts")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    print("Model loaded!")

    # Reference audio
    ref_audio = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    text = "Hello, this is a test of the text to speech system."

    print(f"\nReference audio: {ref_audio}")
    print(f"Reference text: {ref_text}")
    print(f"Text to synthesize: {text}")

    # Generate
    print("\nGenerating audio...")
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    print(f"\nAudio generated!")
    print(f"  Shape: {wavs[0].shape}")
    print(f"  Sample rate: {sr}")
    print(f"  Duration: {len(wavs[0]) / sr:.2f} seconds")

    output_path = "/tmp/hybrid_output.wav"
    sf.write(output_path, wavs[0], sr)
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid qwen_tts demo")
    parser.add_argument("--extract-tensors", action="store_true", help="Extract tensors from official implementation")
    parser.add_argument("--compare-ttnn", action="store_true", help="Compare with TTNN implementation")
    parser.add_argument("--run", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    if args.extract_tensors:
        extract_tensors_from_official()
    elif args.compare_ttnn:
        compare_with_ttnn()
    elif args.run:
        run_full_pipeline()
    else:
        # Default: run full pipeline
        run_full_pipeline()


if __name__ == "__main__":
    main()
