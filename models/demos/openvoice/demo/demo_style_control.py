#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Style Control Demo for OpenVoice V2

Demonstrates all style control parameters:
- tau: Voice characteristic transfer (0.1-0.8)
- speed: Speaking rate (0.5-2.0)
- noise_scale: Expressiveness/emotion variation (0.0-1.0)
- noise_scale_w: Pause/duration variation (0.0-1.0)

Usage:
    python models/demos/openvoice/demo/demo_style_control.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from melo.api import TTS

    MELO_AVAILABLE = True
except ImportError:
    MELO_AVAILABLE = False
    print("MeloTTS not available. Install with: pip install git+https://github.com/myshell-ai/MeloTTS.git")

try:
    import soundfile as sf
except ImportError:
    print("soundfile not available. Install with: pip install soundfile")
    sys.exit(1)


def create_reference_audio(duration: float = 3.0, freq: float = 330.0, sample_rate: int = 22050) -> str:
    """Create a synthetic reference audio file."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        return f.name


def demo_speed_control(output_dir: Path):
    """
    Demo: Speed/Rhythm Control

    The 'speed' parameter controls speaking rate:
    - speed=0.7: Slow, deliberate speech
    - speed=1.0: Normal speed
    - speed=1.5: Fast speech
    """
    print("\n" + "=" * 60)
    print("Demo 1: Speed/Rhythm Control")
    print("=" * 60)

    if not MELO_AVAILABLE:
        print("Skipping - MeloTTS not available")
        return

    tts = TTS(language="EN", device="cpu")
    text = "The speed parameter controls how fast I speak."

    speeds = [0.7, 1.0, 1.5]

    for speed in speeds:
        output_path = output_dir / f"speed_{speed}.wav"
        print(f"\n  Generating speed={speed}...")
        tts.tts_to_file(text, 0, output_path=str(output_path), speed=speed, quiet=True)
        print(f"  Saved: {output_path}")

    print("\n  Listen to hear: slow → normal → fast speech")


def demo_expressiveness(output_dir: Path):
    """
    Demo: Expressiveness/Emotion Variation

    The 'noise_scale' parameter controls variation:
    - noise_scale=0.1: Flat, monotone delivery
    - noise_scale=0.667: Normal expressiveness (default)
    - noise_scale=1.0: Very expressive, more emotional
    """
    print("\n" + "=" * 60)
    print("Demo 2: Expressiveness Control (noise_scale)")
    print("=" * 60)

    if not MELO_AVAILABLE:
        print("Skipping - MeloTTS not available")
        return

    # MeloTTS doesn't expose noise_scale directly in tts_to_file
    # We need to access the model internals
    print("  Note: noise_scale is controlled at model inference level")
    print("  Default noise_scale=0.667 provides balanced expressiveness")
    print("  Higher values (0.8-1.0) = more emotional variation")
    print("  Lower values (0.1-0.3) = more monotone delivery")


def demo_tau_control(output_dir: Path):
    """
    Demo: Voice Transfer Control (tau)

    The 'tau' parameter controls how much target voice is applied:
    - tau=0.1: Mostly source characteristics (prosody preserved)
    - tau=0.3: Balanced (recommended)
    - tau=0.7: Strong target voice (more of reference speaker)
    """
    print("\n" + "=" * 60)
    print("Demo 3: Voice Transfer Control (tau)")
    print("=" * 60)

    try:
        import ttnn

        TTNN_AVAILABLE = True
    except ImportError:
        TTNN_AVAILABLE = False

    # Check for checkpoint
    checkpoint_dir = Path("checkpoints/openvoice/converter")
    if not checkpoint_dir.exists():
        print("  Skipping - checkpoint not found")
        print('  Download with: python -c "from huggingface_hub import hf_hub_download; ..."')
        return

    if not MELO_AVAILABLE:
        print("  Skipping - MeloTTS not available")
        return

    from models.demos.openvoice.tt.tone_color_converter import TTNNToneColorConverter

    # Setup
    device = None
    if TTNN_AVAILABLE:
        try:
            device = ttnn.open_device(device_id=0)
        except:
            pass

    try:
        converter = TTNNToneColorConverter(checkpoint_dir / "config.json", device=device)
        converter.load_checkpoint(checkpoint_dir / "checkpoint.pth")

        # Generate base TTS
        tts = TTS(language="EN", device="cpu")
        text = "This demonstrates how tau affects voice cloning."

        source_path = output_dir / "tau_source.wav"
        tts.tts_to_file(text, 0, output_path=str(source_path), speed=1.0, quiet=True)

        # Create reference
        ref_path = create_reference_audio()

        # Extract embeddings
        src_se = converter.extract_se([str(source_path)])
        tgt_se = converter.extract_se([ref_path])

        # Generate with different tau values
        tau_values = [0.1, 0.3, 0.7]

        for tau in tau_values:
            output_path = output_dir / f"tau_{tau}.wav"
            print(f"\n  Generating tau={tau}...")

            converter.convert(
                source_audio=str(source_path),
                src_se=src_se,
                tgt_se=tgt_se,
                output_path=str(output_path),
                tau=tau,
            )
            print(f"  Saved: {output_path}")

        print("\n  Listen to hear:")
        print("    tau=0.1: More of original voice/prosody")
        print("    tau=0.3: Balanced blend")
        print("    tau=0.7: More of target voice characteristics")

        os.unlink(ref_path)

    finally:
        if device is not None and TTNN_AVAILABLE:
            ttnn.close_device(device)


def demo_pause_variation(output_dir: Path):
    """
    Demo: Pause/Duration Variation

    The 'noise_scale_w' and 'sdp_ratio' control timing:
    - noise_scale_w: Duration variation (pauses between words)
    - sdp_ratio: Stochastic vs deterministic duration prediction
    """
    print("\n" + "=" * 60)
    print("Demo 4: Pause/Duration Control")
    print("=" * 60)

    print("  Parameters that control pauses and timing:")
    print("    noise_scale_w (0.0-1.0): Duration variation")
    print("      - 0.0: Very consistent timing")
    print("      - 0.8: Natural variation (default)")
    print("      - 1.0: More dramatic pauses")
    print("")
    print("    sdp_ratio (0.0-1.0): Stochastic duration predictor ratio")
    print("      - 0.0: Deterministic timing (consistent)")
    print("      - 0.2: Slight variation")
    print("      - 1.0: Fully stochastic (unpredictable)")


def demo_emotion_from_reference(output_dir: Path):
    """
    Demo: Emotion Cloning from Reference

    OpenVoice clones emotion/style from the reference audio.
    To get angry speech: provide an angry reference.
    To get calm speech: provide a calm reference.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Emotion from Reference Audio")
    print("=" * 60)

    print("  OpenVoice clones emotion/style from the REFERENCE AUDIO.")
    print("")
    print("  To control emotion:")
    print("    1. Record/find a reference with the desired emotion")
    print("    2. Use that as the target speaker")
    print("    3. Adjust tau to control transfer strength")
    print("")
    print("  Example workflow for angry speech:")
    print("    reference = 'angry_speaker.wav'  # Contains angry speech")
    print("    tgt_se = converter.extract_se([reference])")
    print("    converter.convert(source, src_se, tgt_se, tau=0.5)")
    print("")
    print("  The model learns: pitch contours, speaking rate,")
    print("  emphasis patterns, and tonal qualities from the reference.")


def main():
    """Run all style control demos."""
    print("\n" + "=" * 60)
    print("OpenVoice V2 - Style Control Demo")
    print("=" * 60)

    # Create output directory
    output_dir = Path("output/style_control_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Run demos
    demo_speed_control(output_dir)
    demo_expressiveness(output_dir)
    demo_tau_control(output_dir)
    demo_pause_variation(output_dir)
    demo_emotion_from_reference(output_dir)

    print("\n" + "=" * 60)
    print("Style Control Summary")
    print("=" * 60)
    print("""
| Control | Parameter | How It Works |
|---------|-----------|--------------|
| Rhythm/Speed | speed | Direct rate multiplier (0.5-2.0) |
| Emotion | reference audio | Clone from reference + tau |
| Accent | reference audio | Clone from reference speaker |
| Expressiveness | noise_scale | Variation in pitch/energy |
| Pauses | noise_scale_w | Duration variation |
| Intonation | reference audio | Clone prosody patterns |

Key insight: OpenVoice's style control is primarily through
REFERENCE AUDIO SELECTION, not explicit parameters.

The tau parameter controls HOW MUCH of the reference style
is applied to the output.
""")

    print(f"\nGenerated files in: {output_dir}/")


if __name__ == "__main__":
    main()
