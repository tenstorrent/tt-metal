#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Download GR00T N1.6-3B weights from HuggingFace.

Model: nvidia/GR00T-N1.6-3B
Size: ~6.5 GB (2 safetensors shards)

Usage:
    python download_weights.py
    python download_weights.py --model-id nvidia/GR00T-N1.6-3B
    python download_weights.py --output-dir /custom/path
"""

import argparse
import sys
from pathlib import Path


def install_dependencies():
    """Install required dependencies if not present."""
    import subprocess

    for package in ["huggingface_hub", "safetensors"]:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package, "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def download_weights(model_id: str, output_dir: Path) -> bool:
    """Download model weights from HuggingFace."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id}...")
    print(f"  Output: {output_dir}")
    print()

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(output_dir),
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        return True
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Manual download:")
        print(f"  pip install huggingface_hub")
        print(f"  huggingface-cli download {model_id} --local-dir {output_dir}")
        return False


def list_weights(output_dir: Path) -> int:
    """List downloaded weights. Returns total size in bytes."""
    total = 0
    for f in sorted(output_dir.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            size_mb = f.stat().st_size / 1024 / 1024
            total += f.stat().st_size
            print(f"  {f.relative_to(output_dir)} ({size_mb:.1f} MB)")
    return total


def main():
    parser = argparse.ArgumentParser(description="Download GR00T N1.6-3B weights")
    parser.add_argument(
        "--model-id",
        default="nvidia/GR00T-N1.6-3B",
        help="HuggingFace model ID (default: nvidia/GR00T-N1.6-3B)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: HuggingFace cache)",
    )
    args = parser.parse_args()

    install_dependencies()

    print("=" * 60)
    print("  GR00T N1.6-3B Weight Downloader")
    print("=" * 60)
    print()

    output_dir = args.output_dir
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not download_weights(args.model_id, output_dir):
            return 1

        print()
        print("Downloaded files:")
        total = list_weights(output_dir)
        print(f"\n  Total: {total / 1024 / 1024 / 1024:.2f} GB")
    else:
        # Use HuggingFace cache (default behavior of weight_loader)
        from huggingface_hub import snapshot_download

        print(f"Downloading {args.model_id} to HuggingFace cache...")
        cache_dir = snapshot_download(
            repo_id=args.model_id,
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print(f"\n  Cached at: {cache_dir}")

    print()
    print("=" * 60)
    print("  Download complete!")
    print("  Weights will be auto-loaded by Gr00tN16WeightLoader()")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
