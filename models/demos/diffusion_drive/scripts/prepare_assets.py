#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Download and cache all assets needed for DiffusionDrive TTNN bring-up.

Downloads:
  1. Model checkpoint (diffusiondrive_navsim.pth) from HuggingFace
     hustvl/DiffusionDrive
  2. Anchor cluster file (kmeans_navsim_traj_20.npy) — extracted from the
     same HuggingFace repo

All files are placed under models/demos/diffusion_drive/data/.

Usage:
    source python_env/bin/activate
    export PYTHONPATH=/root/tt/tt-metal
    python models/demos/diffusion_drive/scripts/prepare_assets.py
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Asset registry
# ---------------------------------------------------------------------------

ASSETS = {
    "checkpoint": {
        "url": "https://huggingface.co/hustvl/DiffusionDrive/resolve/main/diffusiondrive_navsim.pth",
        "dest": "data/diffusiondrive_navsim.pth",
        "sha256": None,  # not pinned — verify manually after download
    },
    "anchors": {
        "url": "https://huggingface.co/hustvl/DiffusionDrive/resolve/main/kmeans_navsim_traj_20.npy",
        "dest": "data/kmeans_navsim_traj_20.npy",
        "sha256": None,
    },
}

# ---------------------------------------------------------------------------


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_sha256: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] already exists: {dest}")
        return

    print(f"  Downloading {url}")
    print(f"  -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:

        def _progress(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, int(count * block_size * 100 / total_size))
                sys.stdout.write(f"\r  {pct}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, tmp, _progress)
        print()
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {exc}") from exc

    if expected_sha256:
        actual = _sha256(tmp)
        if actual != expected_sha256:
            tmp.unlink()
            raise ValueError(
                f"SHA-256 mismatch for {dest.name}\n" f"  expected: {expected_sha256}\n" f"  got:      {actual}"
            )

    tmp.rename(dest)
    print(f"  Saved: {dest}")


def prepare(root: Path) -> None:
    print(f"Asset root: {root.resolve()}")
    for name, info in ASSETS.items():
        print(f"\n[{name}]")
        dest = root / info["dest"]
        _download(info["url"], dest, info.get("sha256"))

    # Quick sanity check on anchor file
    anchors_path = root / "data/kmeans_navsim_traj_20.npy"
    if anchors_path.exists():
        import numpy as np

        anchors = np.load(anchors_path)
        assert anchors.shape == (20, 8, 2), f"Unexpected anchor shape {anchors.shape}; expected (20, 8, 2)"
        print(
            f"\nAnchors OK: shape {anchors.shape}, "
            f"range x [{anchors[...,0].min():.2f}, {anchors[...,0].max():.2f}] "
            f"y [{anchors[...,1].min():.2f}, {anchors[...,1].max():.2f}]"
        )

    print("\nAll assets ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DiffusionDrive assets")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Root of the diffusion_drive demo directory (default: auto-detected)",
    )
    args = parser.parse_args()
    prepare(args.root)


if __name__ == "__main__":
    main()
