#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN SpeechT5 TTS Demo using Wrapper Class

This demo showcases the simplified TTNN SpeechT5 pipeline using the TTNNSpeechT5TTS
wrapper class. Generates high-quality speech from text input with minimal code.

Sample Commands:
    # Basic usage - single text
    python demo_wrapper.py "Hello world"

    # Multiple texts
    python demo_wrapper.py "Hello world" "How are you today?" "The weather is nice"

    # Custom output directory and max steps
    python demo_wrapper.py "Hello world" --output_dir ./audio_output --max_steps 50

Features:
- Uses TTNNSpeechT5TTS wrapper class for simplified API
- Full TTNN implementation (Encoder + Decoder + Postnet)
- KV caching and cross-attention optimizations
- L1 memory optimizations for maximum performance
- Automatic model loading and initialization
- Clean, production-ready code
"""

import sys
import os
import ttnn
import soundfile as sf

# Add path for imports
sys.path.append("/home/ttuser/ssinghal/PR-fix/speecht5_tts/tt-metal")

# Import the wrapper class
from models.experimental.speecht5_tts.demo.demo_utils import TTNNSpeechT5TTS


def main():
    """Main demo function using the wrapper class."""

    import argparse

    parser = argparse.ArgumentParser(description="TTNN SpeechT5 TTS Demo (Wrapper Class)")
    parser.add_argument(
        "texts", nargs="+", help="Input text(s) to convert to speech. Each text will generate a separate audio file."
    )
    parser.add_argument(
        "--output_dir", default=".", help="Output directory for audio files (default: current directory)"
    )
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of generation steps (default: 100)")

    args = parser.parse_args()

    print("🎵 TTNN SpeechT5 Wrapper Class Demo")
    print("=" * 45)
    print(f"📝 Processing {len(args.texts)} input text(s)")
    print(f"🎯 Max steps per generation: {args.max_steps}")
    print(f"📁 Output directory: {args.output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Enable persistent kernel cache for faster subsequent runs
        ttnn.device.EnablePersistentKernelCache()

        # Initialize device
        device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=10000000)

        # Enable program cache for faster inference
        device.enable_program_cache()

        # Create TTS system (loads models automatically)
        tts = TTNNSpeechT5TTS(device, max_steps=args.max_steps)

        # Warm-up phase (compiles TTNN kernels)
        warmup_duration = tts.warmup(num_steps=args.max_steps)
        print(f"🚀 Ready for generation! Warm-up took {warmup_duration:.1f}s")

        # Generate speech for each input text
        results = []
        for i, text in enumerate(args.texts, 1):
            print(f"\n🎵 [{i}/{len(args.texts)}] Generating speech for: '{text}'")

            # Generate filename from text (sanitize for filesystem)
            safe_text = "".join(c for c in text[:50] if c.isalnum() or c in (" ", "-", "_")).rstrip()
            if not safe_text:
                safe_text = f"speech_{i}"
            safe_text = safe_text.replace(" ", "_")
            output_file = os.path.join(args.output_dir, f"speech_wrapper_{safe_text}.wav")

            # Time the generation
            import time

            generation_start = time.time()
            speech, generation_stats = tts.generate(text, max_steps=args.max_steps, return_stats=True)
            generation_time = time.time() - generation_start

            # Save audio
            sf.write(output_file, speech.squeeze().detach().numpy(), samplerate=16000)
            audio_duration = len(speech.squeeze()) / 16000.0

            # Calculate tokens/sec
            tokens_generated = generation_stats.get("steps_completed", 0)
            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0

            # Store results
            result = {
                "text": text,
                "output_file": output_file,
                "generation_time": generation_time,
                "audio_duration": audio_duration,
                "tokens_generated": tokens_generated,
                "tokens_per_sec": tokens_per_sec,
                "sequence_length": generation_stats.get("final_seq_len", 0),
            }
            results.append(result)

            print(f"✅ Audio saved to {output_file}")
            print(".3f")
            print(".2f")

        # Display summary table
        print("\n" + "=" * 120)
        print("📊 INFERENCE SUMMARY")
        print("=" * 120)
        print(f"{'#':<4} {'Text':<30} {'Tokens':<8} {'Time(s)':<10} {'Audio(s)':<8} {'T/s':<8}")
        print("-" * 104)

        total_generation_time = 0
        total_tokens = 0
        total_audio_duration = 0

        for i, result in enumerate(results, 1):
            truncated_text = result["text"][:25] + "..." if len(result["text"]) > 28 else result["text"]
            print(
                f"{i:<4} {truncated_text:<30} {result['tokens_generated']:<8} {result['generation_time']:<10.3f} {result['audio_duration']:<8.1f} {result['tokens_per_sec']:<8.2f}"
            )
            total_generation_time += result["generation_time"]
            total_tokens += result["tokens_generated"]
            total_audio_duration += result["audio_duration"]

        print("-" * 104)
        print(
            f"{'TOTAL':<4} {'':<30} {total_tokens:<8} {total_generation_time:<10.3f} {total_audio_duration:<8.1f} {total_tokens/total_generation_time:<8.2f}"
        )

        print("\n📈 Overall Statistics:")
        print(f"   • Total inference time: {total_generation_time:.3f}s")
        print(f"   • Total tokens generated: {total_tokens}")
        print(f"   • Total audio duration: {total_audio_duration:.3f}s")
        print(f"   • Average tokens/sec: {total_tokens/total_generation_time:.2f} (across all texts)")
        print(f"   • Average audio duration: {total_audio_duration/len(results):.3f}s per text")

        # Cleanup
        ttnn.close_device(device)
        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
