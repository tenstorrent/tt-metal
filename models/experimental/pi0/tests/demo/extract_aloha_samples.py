#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Extract sample images from ALOHA sim dataset for PI0 demo testing.

This script downloads video frames from the lerobot/aloha_sim_transfer_cube_human
dataset and saves them as images for use in demo tests.

Usage:
    python extract_aloha_samples.py
"""

import sys
from pathlib import Path

# Output directory - local to demo folder
DEMO_DIR = Path(__file__).parent
OUTPUT_DIR = DEMO_DIR / "sample_images" / "aloha_sim"


def main():
    print("=" * 60)
    print("  ALOHA Sim Sample Extractor")
    print("=" * 60)

    try:
        from huggingface_hub import hf_hub_download
        import imageio.v3 as iio
    except ImportError as e:
        print(f"\n❌ Required packages not installed: {e}")
        print("   Please install: pip install huggingface_hub imageio[pyav]")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")

    print("\n📥 Downloading ALOHA sim video from HuggingFace...")
    print("   Dataset: lerobot/aloha_sim_transfer_cube_human")
    print("   This may take a moment on first run...")

    try:
        video_path = hf_hub_download(
            "lerobot/aloha_sim_transfer_cube_human",
            "videos/observation.images.top/chunk-000/file-000.mp4",
            repo_type="dataset",
        )
        print(f"   ✅ Downloaded video")
    except Exception as e:
        print(f"\n❌ Failed to download: {e}")
        sys.exit(1)

    print("\n🖼️  Extracting frames...")

    try:
        frames = iio.imread(video_path, plugin="pyav")
        print(f"   Video: {frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]} resolution")
    except Exception as e:
        print(f"\n❌ Failed to read video: {e}")
        sys.exit(1)

    # Extract frames at different points
    sample_indices = [0, 500, 1000, 2000, 4000]

    for i, frame_idx in enumerate(sample_indices):
        if frame_idx < len(frames):
            frame = frames[frame_idx]
            img_path = OUTPUT_DIR / f"sample_{i}_top.png"
            iio.imwrite(img_path, frame)
            print(f"   ✅ Saved: sample_{i}_top.png (frame {frame_idx}, {frame.shape[0]}x{frame.shape[1]})")

    # Save metadata
    metadata_file = OUTPUT_DIR / "metadata.txt"
    with open(metadata_file, "w") as f:
        f.write("ALOHA Sim Sample Images\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: lerobot/aloha_sim_transfer_cube_human\n")
        f.write(f"Video frames: {frames.shape[0]}\n")
        f.write(f"Resolution: {frames.shape[1]}x{frames.shape[2]}\n")
        f.write(f"Extracted frames: {sample_indices}\n")
        f.write(f"Task: Transfer cube\n")

    print(f"\n✅ Extraction complete!")
    print(f"   Saved to: {OUTPUT_DIR}")

    print(f"\n📂 Saved files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   {f.name}")


if __name__ == "__main__":
    main()
