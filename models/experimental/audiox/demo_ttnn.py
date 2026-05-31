#!/usr/bin/env python3
import argparse
import time
import torch
import ttnn
import soundfile as sf

from models.experimental.audiox.reference.audiox_model import AudioXModel
from models.experimental.audiox.tt.ttnn_audiox import TtnnAudioXModel


def main():
    parser = argparse.ArgumentParser(description="AudioX TTNN Demo")
    parser.add_argument("--prompt", type=str, default="gentle rain falling on leaves")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--length", type=int, default=160)
    parser.add_argument("--mode", type=str, choices=["text-to-audio", "text-to-music"],
                        default="text-to-audio")
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    device = ttnn.open_device(device_id=args.device_id)
    print(f"Running AudioX {args.mode} on device {device.id()}")

    ref_model = AudioXModel()
    ref_model.eval()
    print(f"Reference model loaded ({sum(p.numel() for p in ref_model.parameters()):,} params)")

    ttnn_model = TtnnAudioXModel(ref_model, device)
    print("TTNN model initialized")

    print(f"Generating audio from prompt: '{args.prompt}'")
    start = time.perf_counter()
    audio = ttnn_model.generate(num_steps=args.num_steps, length=args.length)
    elapsed = time.perf_counter() - start
    sf.write(args.output, audio.squeeze().cpu().numpy(), 16000)
    print(f"Generated {args.output} in {elapsed:.2f}s ({args.num_steps / elapsed:.1f} steps/s)")

    print(f"\nReference (PyTorch) run for comparison:")
    start = time.perf_counter()
    ref_audio = ref_model.generate(num_steps=args.num_steps, length=args.length)
    ref_elapsed = time.perf_counter() - start
    ref_output = args.output.replace(".wav", "_ref.wav")
    sf.write(ref_output, ref_audio.squeeze().cpu().numpy(), 16000)
    print(f"Generated {ref_output} in {ref_elapsed:.2f}s")

    diff = (audio - ref_audio).abs().mean().item()
    print(f"Mean absolute difference vs reference: {diff:.6f}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
