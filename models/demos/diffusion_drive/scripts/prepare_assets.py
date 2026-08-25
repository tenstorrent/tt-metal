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
    export PYTHONPATH="${TT_METAL_HOME:-$PWD}"   # your tt-metal checkout
    python models/demos/diffusion_drive/scripts/prepare_assets.py
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# HuggingFace asset
# ---------------------------------------------------------------------------

_HF_REPO = "hustvl/DiffusionDrive"
_HF_FILENAME = "diffusiondrive_navsim_88p1_PDMS"
# Pin the artifact. ``main`` is a mutable ref, so a replaced upload would otherwise
# be fetched and trusted silently. Revision *and* content hash are both checked and
# a mismatch fails closed (the suspect file is deleted, not left on disk).
_HF_REVISION = "8e3cc29cfdb5aa1a4c0818012f9a250d5153bc71"
_CKPT_SHA256 = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
_CKPT_SIZE = 729518199
_LOCAL_CKPT = "data/diffusiondrive_navsim.pth"
_LOCAL_ANCHORS = "data/kmeans_navsim_traj_20.npy"


def _safe_subpath(root: Path, rel: str) -> Path:
    """Resolve ``root / rel`` and confirm the result stays within ``root``.

    ``root`` is operator-supplied (the ``--root`` argument), so validate the join
    rather than trusting it: a stray absolute path or ``..`` in the (otherwise
    constant) relative name would otherwise let a write escape the asset root.
    """
    root = root.resolve()
    dest = (root / rel).resolve()
    if dest != root and root not in dest.parents:
        raise ValueError(f"refusing to write outside asset root: {dest} is not under {root}")
    return dest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: Path) -> None:
    """Fail closed unless ``path`` is byte-for-byte the pinned checkpoint.

    Covers both threat models: a replaced upstream artifact, and a silently
    truncated transfer (seen in practice — a dropped HTTPS download still exits 0).
    The offending file is deleted so a re-run starts clean rather than reusing it.
    """
    size = path.stat().st_size
    digest = _sha256(path)
    if size == _CKPT_SIZE and digest == _CKPT_SHA256:
        print(f"  Verified: sha256 {digest[:16]}... ({size} bytes)")
        return
    path.unlink(missing_ok=True)
    raise RuntimeError(
        f"checkpoint verification FAILED for {path} (file deleted)\n"
        f"  expected sha256={_CKPT_SHA256} size={_CKPT_SIZE}\n"
        f"  actual   sha256={digest} size={size}\n"
        f"Refusing to use an unverified checkpoint."
    )


def _download_hf(repo_id: str, filename: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] already exists: {dest}")
        _verify_checkpoint(dest)
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if hf_hub_download is not None:
        print(f"  Downloading {repo_id}/{filename} @ {_HF_REVISION[:12]}")
        print(f"  -> {dest}")
        try:
            downloaded = hf_hub_download(
                repo_id,
                filename,
                revision=_HF_REVISION,
                local_dir=str(tmp.parent),
            )
            # The returned path is assembled from repo-controlled metadata, so confirm
            # it really landed inside the asset dir before moving it into place.
            src = Path(downloaded).resolve()
            asset_dir = tmp.parent.resolve()
            if asset_dir not in src.parents:
                raise RuntimeError(f"download landed outside the asset dir: {src}")
            src.rename(dest)
        except Exception as exc:
            # Fall through to the plain HTTPS path rather than giving up. The usual
            # cause in CI is HF_HUB_OFFLINE=1, which the model-test container sets on
            # SKUs whose weight cache is a read-only NFS mount; hf_hub_download then
            # refuses to touch the network even though the runner has one. urllib is
            # unaffected, fetches the same pinned revision, and the SHA-256 check
            # below is what actually establishes trust either way.
            tmp.unlink(missing_ok=True)
            print(f"  huggingface_hub download failed ({exc}) — falling back to direct URL")
        else:
            _verify_checkpoint(dest)
            print(f"  Saved: {dest}")
            return
    else:
        print("  huggingface_hub not installed — falling back to direct URL")

    _download_url(f"https://huggingface.co/{repo_id}/resolve/{_HF_REVISION}/{filename}", dest)
    _verify_checkpoint(dest)
    print(f"  Saved: {dest}")


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] already exists: {dest}")
        return
    print(f"  Downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        # Report only when the percentage actually changes. urlretrieve fires the
        # callback per block (~89k times for this file), and in a CI log — where \r
        # does not overwrite — that was half a megabyte of noise. Off a tty, print
        # every 10% on its own line instead of redrawing.
        interactive = sys.stdout.isatty()
        state = {"pct": -1}

        def _progress(count, block_size, total_size):
            if total_size <= 0:
                return
            pct = min(100, int(count * block_size * 100 / total_size))
            step = 1 if interactive else 10
            if pct < state["pct"] + step:
                return
            state["pct"] = pct
            if interactive:
                sys.stdout.write(f"\r  {pct}%")
            else:
                sys.stdout.write(f"  {pct}%\n")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, tmp, _progress)
        if interactive:
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

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        # weights_only=True rejects any pickle that would execute code on load.
        # If the artifact will not decode as plain tensors, treat it as hostile
        # rather than falling back to the unrestricted loader.
        raise RuntimeError(
            f"refusing to load {ckpt_path}: it did not decode as a plain tensor "
            f"checkpoint under weights_only=True ({exc})"
        ) from exc
    sd = ckpt["state_dict"]
    key = "agent._transfuser_model._trajectory_head.plan_anchor"
    anchor = sd[key].numpy()
    assert anchor.shape == (20, 8, 2), f"Unexpected anchor shape {anchor.shape}"
    np.save(anchors_path, anchor)
    print(f"  Saved: {anchors_path}  shape={anchor.shape}")


def prepare(root: Path) -> None:
    print(f"Asset root: {root.resolve()}")

    ckpt_path = _safe_subpath(root, _LOCAL_CKPT)
    anchors_path = _safe_subpath(root, _LOCAL_ANCHORS)

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
