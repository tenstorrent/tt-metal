#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Download weights from Google Drive using gdown
"""

import os
import sys
import subprocess
from pathlib import Path


def check_and_install_gdown():
    """Check if gdown is installed, install if not."""
    try:
        pass

        print("✓ gdown is already installed")
        return True
    except ImportError:
        print("gdown is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            print("✓ gdown installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install gdown: {e}")
            return False


def download_weights(file_id, output_path):
    """Download weights from Google Drive."""
    import gdown

    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading weights to: {output_path}")

        # Download the file
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

        if Path(output_path).exists():
            print(f"✓ Successfully downloaded weights to {output_path}")
            return True
        else:
            print("✗ Download failed - file not found after download")
            return False

    except Exception as e:
        print(f"✗ Error downloading weights: {e}")
        return False


# Default values for MapTR checkpoint
MAPTR_WEIGHTS_PATH = "models/experimental/MapTR/chkpt/downloaded_weights.pth"
GOOGLE_DRIVE_FILE_ID = "1y-UBwGBSb2xiV40AuQEBhB-xJyV7VusX"


def ensure_checkpoint_downloaded(weights_path: str = MAPTR_WEIGHTS_PATH, file_id: str = GOOGLE_DRIVE_FILE_ID):
    """Ensure checkpoint is downloaded, download if missing.

    Args:
        weights_path: Path to the checkpoint file
        file_id: Google Drive file ID for the checkpoint

    Raises:
        FileNotFoundError: If download fails
    """
    if os.path.exists(weights_path):
        return

    try:
        from loguru import logger

        logger.info(f"Checkpoint not found at {weights_path}. Downloading...")
    except ImportError:
        print(f"Checkpoint not found at {weights_path}. Downloading...")

    # Check/install gdown
    if not check_and_install_gdown():
        raise RuntimeError("Failed to install gdown. Cannot download checkpoint.")

    # Download weights
    if not download_weights(file_id, weights_path):
        raise FileNotFoundError(f"Failed to download checkpoint to {weights_path}")


def main():
    """Main function."""
    print("=== MapTR Weights Downloader ===")
    print(f"File ID: {GOOGLE_DRIVE_FILE_ID}")
    print(f"Output: {MAPTR_WEIGHTS_PATH}")
    print()

    # Check/install gdown
    if not check_and_install_gdown():
        sys.exit(1)

    # Download weights
    if download_weights(GOOGLE_DRIVE_FILE_ID, MAPTR_WEIGHTS_PATH):
        print("\n✓ Download completed successfully!")
    else:
        print("\n✗ Download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
