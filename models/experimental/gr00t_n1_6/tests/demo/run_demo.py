#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GR00T N1.6 Interactive Demo on Tenstorrent Blackhole.

Runs the full VLA pipeline:
1. SigLIP2 vision encoding (27 layers)
2. Pixel shuffle + connector MLP
3. Flow matching with AlternateVLDiT (32 layers, 4 Euler steps)

Uses random or sample images as input (Qwen3 backbone not yet ported).

Usage:
    cd /home/ttuser/experiments/pi0/tt-metal
    export TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole
    python models/experimental/gr00t_n1_6/tests/demo/run_demo.py
    python models/experimental/gr00t_n1_6/tests/demo/run_demo.py --image path/to/image.png
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

DEMO_DIR = Path(__file__).parent


def load_image(path: str, size: int = 224) -> torch.Tensor:
    """Load and preprocess an image to [1, 3, H, W] tensor."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5  # normalize to [-1, 1]
    return tensor.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="GR00T N1.6 Demo on Blackhole")
    parser.add_argument("--image", type=str, default=None, help="Path to input image (224x224 RGB)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of inference runs for timing")
    parser.add_argument("--embodiment-id", type=int, default=0, help="Embodiment ID (0-31)")
    parser.add_argument("--prompt", type=str, default="pick up the object", help="Language instruction")
    parser.add_argument("--no-backbone", action="store_true", help="Skip Qwen3 backbone (use dummy features)")
    args = parser.parse_args()

    print("=" * 60)
    print("  GR00T N1.6-3B Demo on Tenstorrent Blackhole")
    print("=" * 60)
    print()

    # Load model
    print("Loading model...")
    import ttnn
    from models.experimental.gr00t_n1_6.common.configs import Gr00tN16Config
    from models.experimental.gr00t_n1_6.common.weight_loader import Gr00tN16WeightLoader
    from models.experimental.gr00t_n1_6.tt.ttnn_groot_n16_model import Gr00tN16ModelTTNN
    from models.experimental.gr00t_n1_6.tt.ttnn_common import to_tt_tensor

    config = Gr00tN16Config.default()
    loader = Gr00tN16WeightLoader()
    loader.load()

    device = ttnn.open_device(device_id=0)

    try:
        model = Gr00tN16ModelTTNN(config, loader, device)
        print("  Model loaded successfully")
        print()

        # Prepare input
        if args.image:
            print(f"Loading image: {args.image}")
            pixel_values = load_image(args.image, config.backbone.vision.image_size)
        else:
            # Check for sample images
            sample_dir = DEMO_DIR / "sample_images"
            sample_imgs = list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
            if sample_imgs:
                img_path = sample_imgs[0]
                print(f"Using sample image: {img_path.name}")
                pixel_values = load_image(str(img_path), config.backbone.vision.image_size)
            else:
                print("Using random input (no sample images found)")
                torch.manual_seed(args.seed)
                pixel_values = torch.randn(1, 3, 224, 224)

        state = torch.randn(1, config.embodiment.max_state_dim)

        # Simple tokenization: encode prompt as byte-level token IDs
        text_tokens = torch.tensor(
            [[ord(c) % 151936 for c in args.prompt]],
            dtype=torch.long,
        )

        print(f"  Image shape: {pixel_values.shape}")
        print(f"  State shape: {state.shape}")
        print(f'  Prompt: "{args.prompt}" ({text_tokens.shape[1]} tokens)')
        print(f"  Embodiment ID: {args.embodiment_id}")
        print(f"  Mode: {'Full E2E (with Qwen3)' if not args.no_backbone else 'Vision + Action Head only'}")
        print()

        # Step 1: Vision encoding
        print("Step 1: Vision encoding (SigLIP2 + connector)...")
        t0 = time.time()
        image_tokens = model.encode_vision(pixel_values)
        vision_ms = (time.time() - t0) * 1000
        image_tokens_cpu = ttnn.to_torch(image_tokens)
        print(f"  Output: {image_tokens_cpu.shape}")
        print(f"  Range: [{image_tokens_cpu.min():.4f}, {image_tokens_cpu.max():.4f}]")
        print(f"  Latency: {vision_ms:.1f}ms")
        print()

        if args.no_backbone:
            # Legacy mode: skip Qwen3, use vision features directly
            print("Step 2: Skipping Qwen3 backbone (--no-backbone)")
            backbone_features = to_tt_tensor(image_tokens_cpu, device)
            backbone_ms = 0.0
        else:
            # Full E2E: run Qwen3 backbone
            print("Step 2: Qwen3 backbone (16 layers, 2048 dim)...")
            t0 = time.time()
            backbone_features = model.encode_backbone(image_tokens, text_tokens)
            backbone_ms = (time.time() - t0) * 1000
            bb_cpu = ttnn.to_torch(backbone_features)
            print(f"  Output: {bb_cpu.shape}")
            print(f"  Range: [{bb_cpu.min():.4f}, {bb_cpu.max():.4f}]")
            print(f"  Latency: {backbone_ms:.1f}ms")
        ttnn.deallocate(image_tokens)
        print()

        # Step 3: Flow matching
        print("Step 3: Flow matching (4 Euler steps, 32-layer DiT)...")
        t0 = time.time()
        actions = model.run_flow_matching(backbone_features, state, embodiment_id=args.embodiment_id)
        flow_ms = (time.time() - t0) * 1000
        print(f"  Output: {actions.shape}")
        print(f"  Range: [{actions.min():.4f}, {actions.max():.4f}]")
        print(f"  Latency: {flow_ms:.1f}ms")
        print()

        # Performance measurement
        print(f"Running {args.num_runs} timed iterations...")
        times = []
        for i in range(args.num_runs):
            t0 = time.time()
            img = model.encode_vision(pixel_values)
            if args.no_backbone:
                bb = to_tt_tensor(ttnn.to_torch(img), device)
            else:
                bb = model.encode_backbone(img, text_tokens)
            ttnn.deallocate(img)
            actions = model.run_flow_matching(bb, state, embodiment_id=args.embodiment_id)
            elapsed = (time.time() - t0) * 1000
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.1f}ms")

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        hz = 1000.0 / avg_ms

        # Summary
        print()
        print("=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print(f"  Vision encoding:    {vision_ms:.1f}ms")
        if not args.no_backbone:
            print(f"  Qwen3 backbone:     {backbone_ms:.1f}ms")
        print(f"  Flow matching:      {flow_ms:.1f}ms")
        print(f"  Total (avg):        {avg_ms:.1f}ms")
        print(f"  Total (best):       {min_ms:.1f}ms")
        print(f"  Throughput:         {hz:.1f} Hz")
        print()
        print(f"  Action output:      {actions.shape}")
        print(f"    Horizon:          {actions.shape[1]} steps")
        print(f"    Dimensions:       {actions.shape[2]}")
        print(f"    Finite:           {not (actions.isnan().any() or actions.isinf().any())}")
        print("=" * 60)

    finally:
        ttnn.close_device(device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
