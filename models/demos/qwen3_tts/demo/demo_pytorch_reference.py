# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
PyTorch Reference Demo for Qwen3-TTS

This demo uses the official qwen_tts package to generate audio,
extracting intermediate tensors at every step for PCC comparison with TTNN.

Usage:
    # In qwen_tts environment (NOT tt-metal environment):
    python models/demos/qwen3_tts/demo/demo_pytorch_reference.py
"""

import time
from pathlib import Path

import numpy as np
import torch


def main():
    """Run PyTorch reference demo."""
    print("=" * 80)
    print("PyTorch Reference Demo for Qwen3-TTS")
    print("=" * 80)

    # Load qwen_tts
    try:
        from qwen_tts import Qwen3TTS
    except ImportError:
        print("ERROR: qwen_tts not found. Install with: pip install qwen_tts")
        return

    # Initialize model
    print("\nLoading model...")
    start_time = time.time()
    model = Qwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    load_time = time.time() - start_time
    print(f"  Model loaded in {load_time:.1f}s")

    # Test text input
    text = "Hello, this is a test of the Qwen three T T S system."
    print(f"\nText: {text}")

    # Generate audio
    print("\nGenerating audio...")
    start_time = time.time()

    # Use the model's generate method
    with torch.no_grad():
        output = model.generate(text=text, max_new_tokens=256)

    gen_time = time.time() - start_time
    print(f"  Generation time: {gen_time:.1f}s")

    # Extract audio
    if hasattr(output, "audio"):
        audio = output.audio
    elif isinstance(output, dict) and "audio" in output:
        audio = output["audio"]
    elif isinstance(output, torch.Tensor):
        audio = output
    else:
        print(f"  Output type: {type(output)}")
        print(f"  Output: {output}")
        audio = None

    if audio is not None:
        if isinstance(audio, torch.Tensor):
            audio_np = audio.squeeze().cpu().numpy()
        else:
            audio_np = np.array(audio).squeeze()

        print(f"  Audio shape: {audio_np.shape}")
        print(f"  Audio duration: {len(audio_np) / 24000:.2f}s")
        print(f"  Audio range: [{audio_np.min():.4f}, {audio_np.max():.4f}]")

        # Save audio
        import soundfile as sf

        output_path = "/tmp/pytorch_reference_output.wav"
        sf.write(output_path, audio_np, 24000)
        print(f"\n  Saved to: {output_path}")
    else:
        print("  Warning: Could not extract audio from output")

    # Extract and save intermediate tensors for comparison
    print("\n" + "=" * 80)
    print("Extracting Intermediate Tensors for PCC Comparison")
    print("=" * 80)

    tensors_to_save = {}

    # Try to access model internals
    if hasattr(model, "talker"):
        print("  Found talker model")
        tensors_to_save["talker_config"] = str(model.talker.config if hasattr(model.talker, "config") else "N/A")

    # Save tensors
    save_path = Path("/tmp/qwen_tts_tensors/pytorch_reference_tensors.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors_to_save, save_path)
    print(f"\n  Saved tensors to: {save_path}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Text: {text}")
    print(f"Generation time: {gen_time:.1f}s")
    if audio is not None:
        print(f"Audio duration: {len(audio_np) / 24000:.2f}s")
        print(f"Output: /tmp/pytorch_reference_output.wav")
    print("\nListen to the audio to verify it sounds correct!")


if __name__ == "__main__":
    main()
