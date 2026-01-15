#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Extract sample images from LIBERO dataset for PI0 demo testing.

This script downloads a few samples from the HuggingFaceVLA/libero
dataset and saves them as images for use in demo tests.

Usage:
    python extract_libero_samples.py
"""

import sys
from pathlib import Path

import numpy as np

# Output directory - local to demo folder
DEMO_DIR = Path(__file__).parent
OUTPUT_DIR = DEMO_DIR / "sample_images" / "libero"

# Safety limit for dataset iteration (prevents infinite loops on streaming datasets)
MAX_ITERATION_LIMIT = 10000


def main():
    print("=" * 60)
    print("  LIBERO Sample Extractor")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("\n❌ datasets not installed. Please install: pip install datasets")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")

    print("\n📥 Loading HuggingFaceVLA/libero dataset (streaming)...")
    ds = load_dataset("HuggingFaceVLA/libero", split="train", streaming=True)

    print("\n🖼️  Extracting samples...")

    # Get samples from different episodes
    sample_count = 0
    episode_samples = {}

    for i, sample in enumerate(ds):
        ep_idx = sample["episode_index"]

        # Get one sample per episode, up to 5 episodes
        if ep_idx not in episode_samples and len(episode_samples) < 5:
            episode_samples[ep_idx] = sample

            # Save images
            img1 = sample["observation.images.image"]
            img2 = sample["observation.images.image2"]

            # Save main image
            filename1 = f"sample_{sample_count}_main.png"
            img1.save(OUTPUT_DIR / filename1)
            print(f"   ✅ Saved: {filename1} (size: {img1.size})")

            # Save wrist image
            filename2 = f"sample_{sample_count}_wrist.png"
            img2.save(OUTPUT_DIR / filename2)
            print(f"   ✅ Saved: {filename2} (size: {img2.size})")

            # Print task info
            task_idx = sample.get("task_index", "N/A")
            state = np.array(sample["observation.state"])
            print(f"      Episode: {ep_idx}, Task index: {task_idx}, State dim: {len(state)}")

            sample_count += 1

        if len(episode_samples) >= 5:
            break

        # Safety limit
        if i > MAX_ITERATION_LIMIT:
            print(f"Reached iteration limit ({MAX_ITERATION_LIMIT})")
            break

    # Save metadata
    metadata_file = OUTPUT_DIR / "metadata.txt"
    with open(metadata_file, "w") as f:
        f.write("LIBERO Sample Images\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: HuggingFaceVLA/libero\n")
        f.write(f"Extracted samples: {sample_count}\n")
        f.write(f"Episodes: {list(episode_samples.keys())}\n")

    print(f"\n✅ Extraction complete!")
    print(f"   Saved {sample_count * 2} images to: {OUTPUT_DIR}")

    print(f"\n📂 Saved files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   {f.name}")


if __name__ == "__main__":
    main()
