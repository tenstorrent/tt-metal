#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Download PI0 pretrained weights from Google Drive.

This script downloads the pi0_base checkpoint folder from Google Drive
to the models/experimental/pi0/weights directory.

Usage:
    python download_pretrained_weights.py
    python download_pretrained_weights.py --output-dir /custom/path

Google Drive folder:
    https://drive.google.com/drive/folders/1qfY0EBGh_-6Zz-omKPQW6nBcc1Cp2_WN
"""

import argparse
import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required dependencies if not present."""
    required_packages = ["gdown"]

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package, "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  ✅ {package} installed")


# Install dependencies before importing
install_dependencies()

import gdown  # noqa: E402


# =============================================================================
# CONFIGURATION
# =============================================================================

# Google Drive folder ID
# From: https://drive.google.com/drive/folders/1qfY0EBGh_-6Zz-omKPQW6nBcc1Cp2_WN
GDRIVE_FOLDER_ID = "1qfY0EBGh_-6Zz-omKPQW6nBcc1Cp2_WN"

# Default output directory (relative to this script's parent)
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "weights"


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================


def download_folder_from_gdrive(folder_id: str, output_dir: Path) -> bool:
    """
    Download a folder from Google Drive.

    Args:
        folder_id: Google Drive folder ID
        output_dir: Directory to download files to

    Returns:
        True if successful, False otherwise
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    print(f"Downloading folder from Google Drive...")
    print(f"  URL: {url}")
    print(f"  Output: {output_dir}")
    print()

    try:
        # gdown.download_folder downloads all files in the folder
        gdown.download_folder(
            url=url,
            output=str(output_dir),
            quiet=False,
            use_cookies=False,
        )
        return True
    except Exception as e:
        print(f"Error downloading folder: {e}")
        print()
        print_manual_download_instructions(folder_id, output_dir)
        return False


def print_manual_download_instructions(folder_id: str, output_dir: Path):
    """Print instructions for manual download."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    print("=" * 60)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("If the download did not work, try one of these options:")
    print()
    print("Option 1: Command line with gdown (requires public folder)")
    print("-" * 40)
    print(f"  pip install gdown")
    print(f"  gdown --folder {url} -O $TT_METAL_HOME/models/experimental/pi0/weights/")
    print()
    print("Option 2: Command line with rclone (works with private folders)")
    print("-" * 40)
    print("  # Install: https://rclone.org/install/")
    print("  rclone config  # Setup Google Drive remote named 'gdrive'")
    print(f"  rclone copy gdrive:pi0_base $TT_METAL_HOME/models/experimental/pi0/weights/pi0_base")
    print()
    print("Option 3: Manual browser download")
    print("-" * 40)
    print(f"  1. Open: {url}")
    print("  2. Download the folder (right-click → Download)")
    print("  3. Extract to: $TT_METAL_HOME/models/experimental/pi0/weights/")
    print()
    print("After download, your directory should look like:")
    print("  $TT_METAL_HOME/models/experimental/pi0/weights/")
    print("    └── pi0_base/")
    print("        ├── model.safetensors (or .bin files)")
    print("        └── config.json")
    print()


def list_weights(output_dir: Path, indent: int = 0) -> int:
    """List downloaded weights with sizes. Returns total size in bytes."""
    total_size = 0
    prefix = "  " * indent

    for item in sorted(output_dir.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            print(f"{prefix}📁 {item.name}/")
            total_size += list_weights(item, indent + 1)
        else:
            size_mb = item.stat().st_size / 1024 / 1024
            total_size += item.stat().st_size
            print(f"{prefix}📄 {item.name} ({size_mb:.1f} MB)")

    return total_size


def main():
    parser = argparse.ArgumentParser(description="Download PI0 pretrained weights from Google Drive")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for weights (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--folder-id",
        type=str,
        default=GDRIVE_FOLDER_ID,
        help="Google Drive folder ID (overrides default)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PI0 Pretrained Weights Downloader")
    print("=" * 60)
    print()
    print(f"Google Drive folder: {args.folder_id}")
    print(f"Output directory: {output_dir}")
    print()

    # Check if weights already exist
    existing_files = list(output_dir.glob("**/*.safetensors")) + list(output_dir.glob("**/*.bin"))
    if existing_files:
        print(f"⚠️  Found {len(existing_files)} existing weight files in {output_dir}")
        response = input("Continue and overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return 0

    # Download
    print("-" * 60)
    if not download_folder_from_gdrive(args.folder_id, output_dir):
        print("\n❌ Download failed!")
        return 1

    # List contents
    print()
    print("-" * 60)
    print("Downloaded contents:")
    print("-" * 60)
    total_size = list_weights(output_dir)
    total_mb = total_size / 1024 / 1024
    total_gb = total_size / 1024 / 1024 / 1024

    print()
    print("=" * 60)
    print(f"✅ Weights downloaded successfully!")
    print(f"   Location: {output_dir}")
    print(f"   Total size: {total_gb:.2f} GB ({total_mb:.0f} MB)")
    print("=" * 60)

    # Print usage hint
    print()
    print("To use these weights in tests, set:")
    print(f'  CHECKPOINT_PATH = "{output_dir}"')
    print()
    print("Weights should now be at:")
    print(f"  {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
