#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Download and cache all assets needed for DiffusionDrive TTNN bring-up.

Downloads:
  1. Model checkpoint (diffusiondrive_navsim_88p1_PDMS) from HuggingFace
     hustvl/DiffusionDrive — saved locally as diffusiondrive_navsim.pth
  2. Anchor cluster file (kmeans_navsim_traj_20.npy) — extracted from the
     checkpoint's state_dict (plan_anchor tensor, shape 20×8×2)

All files are placed under models/demos/diffusion_drive/data/.

Usage:
    source python_env/bin/activate
    export PYTHONPATH=/root/tt/tt-metal
    python models/demos/diffusion_drive/scripts/prepare_assets.py
"""

import argparse
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# HuggingFace asset
# ---------------------------------------------------------------------------

_HF_REPO = "hustvl/DiffusionDrive"
_HF_FILENAME = "diffusiondrive_navsim_88p1_PDMS"
_LOCAL_CKPT = "data/diffusiondrive_navsim.pth"
_LOCAL_ANCHORS = "data/kmeans_navsim_traj_20.npy"


def _download_hf(repo_id: str, filename: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] already exists: {dest}")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  huggingface_hub not installed — falling back to direct URL")
        _download_url(
            f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
            dest,
        )
        return

    print(f"  Downloading {repo_id}/{filename}")
    print(f"  -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        downloaded = hf_hub_download(repo_id, filename, local_dir=str(tmp.parent), local_dir_use_symlinks=False)
        Path(downloaded).rename(dest)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {exc}") from exc
    print(f"  Saved: {dest}")


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] already exists: {dest}")
        return
    print(f"  Downloading {url}")
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
    tmp.rename(dest)
    print(f"  Saved: {dest}")


def _extract_anchors(ckpt_path: Path, anchors_path: Path) -> None:
    anchors_path.parent.mkdir(parents=True, exist_ok=True)
    if anchors_path.exists():
        print(f"  [skip] already exists: {anchors_path}")
        return
    print(f"  Extracting anchors from {ckpt_path.name}")
    import numpy as np
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    key = "agent._transfuser_model._trajectory_head.plan_anchor"
    anchor = sd[key].numpy()
    assert anchor.shape == (20, 8, 2), f"Unexpected anchor shape {anchor.shape}"
    np.save(anchors_path, anchor)
    print(f"  Saved: {anchors_path}  shape={anchor.shape}")


def prepare(root: Path) -> None:
    print(f"Asset root: {root.resolve()}")

    ckpt_path = root / _LOCAL_CKPT
    anchors_path = root / _LOCAL_ANCHORS

    print("\n[checkpoint]")
    _download_hf(_HF_REPO, _HF_FILENAME, ckpt_path)

    print("\n[anchors]")
    _extract_anchors(ckpt_path, anchors_path)

    # Sanity check
    if anchors_path.exists():
        import numpy as np

        anchors = np.load(anchors_path)
        assert anchors.shape == (20, 8, 2), f"Unexpected anchor shape {anchors.shape}"
        print(
            f"\nAnchors OK: shape {anchors.shape}, "
            f"range x [{anchors[..., 0].min():.2f}, {anchors[..., 0].max():.2f}] "
            f"y [{anchors[..., 1].min():.2f}, {anchors[..., 1].max():.2f}]"
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
