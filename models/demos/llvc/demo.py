#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("Warning: Torchaudio not available")

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    print("Error: TTNN not available")
    sys.exit(1)

from ttnn_llvc import TtLLVCModel, load_llvc_model, preprocess_audio, infer_llvc


def load_audio(audio_path, sample_rate):
    """Load audio file"""
    if not TORCHAUDIO_AVAILABLE:
        raise ImportError("Torchaudio not available")
    audio, sr = torchaudio.load(audio_path)
    audio = audio.mean(0, keepdim=False)  # Convert to mono
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    return audio


def save_audio(audio, audio_path, sample_rate):
    """Save audio file"""
    if not TORCHAUDIO_AVAILABLE:
        raise ImportError("Torchaudio not available")
    torchaudio.save(audio_path, audio, sample_rate)


def main():
    parser = argparse.ArgumentParser(description="LLVC Voice Conversion Demo")
    parser.add_argument('--checkpoint_path', '-p', type=str,
                       default='llvc_models/models/checkpoints/llvc/G_500000.pth',
                       help='Path to LLVC checkpoint file')
    parser.add_argument('--config_path', '-c', type=str,
                       default='experiments/llvc/config.json',
                       help='Path to LLVC config file')
    parser.add_argument('--input_file', '-i', type=str,
                       default='test_wavs/example.wav',
                       help='Path to input audio file')
    parser.add_argument('--output_file', '-o', type=str,
                       default='converted_output.wav',
                       help='Path to output audio file')
    parser.add_argument('--streaming', '-s', action='store_true',
                       help='Use streaming inference')
    parser.add_argument('--chunk_factor', '-n', type=int, default=1,
                       help='Chunk factor for streaming inference')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Input file {args.input_file} not found!")
        return

    # Check if checkpoint and config exist
    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint {args.checkpoint_path} not found!")
        return

    if not os.path.exists(args.config_path):
        print(f"Config {args.config_path} not found!")
        return

    print("Loading LLVC model...")

    # Initialize TT device
    device = ttnn.open_device(0)

    try:
        # Load model
        model, sample_rate = load_llvc_model(args.checkpoint_path, args.config_path, device)

        print(f"Model loaded successfully. Sample rate: {sample_rate}")

        # Load audio
        print(f"Loading audio from {args.input_file}")
        audio = load_audio(args.input_file, sample_rate)
        print(f"Audio loaded. Shape: {audio.shape}, Duration: {len(audio)/sample_rate:.2f}s")

        # Preprocess audio
        audio_tensor = preprocess_audio(audio, sample_rate, sample_rate)
        audio_tensor = ttnn.to_device(audio_tensor, device)

        # Run inference
        print("Running inference...")
        with torch.no_grad():
            if args.streaming:
                print(f"Using streaming inference with chunk_factor={args.chunk_factor}")
                output = infer_llvc(model, audio_tensor, streaming=True, chunk_factor=args.chunk_factor)
            else:
                print("Using non-streaming inference")
                output = infer_llvc(model, audio_tensor, streaming=False)

        # Convert back to torch tensor
        output_audio = ttnn.to_torch(output).squeeze()

        # Save output
        print(f"Saving output to {args.output_file}")
        save_audio(output_audio, args.output_file, sample_rate)

        print("Inference completed successfully!")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Close device
        ttnn.close_device(device)


if __name__ == '__main__':
    main()</content>
<parameter name="filePath">/home/mahmudsudo/tt-metal/models/demos/llvc/demo.py