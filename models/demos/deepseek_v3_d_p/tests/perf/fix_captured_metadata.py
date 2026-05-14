#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
One-shot fixer for already-captured combine .pt files.

Older captures stored metadata[..., 0] as the Galaxy GLOBAL LinMeshCoord
(values like {0, 4, 8, 12, 16, 20, 24, 28} on 8x4) instead of the
within-dispatch-group chip ID [0, 8). The combine kernel on an 8-chip
mesh interprets these as fabric write destinations and hangs because
chips 8..31 don't exist.

This script applies the remap `src_chip //= num_dispatch_groups` to all
.pt files under a directory and writes them back in place.

Usage:
    python fix_captured_metadata.py [<capture_dir>] [--num-dispatch-groups N]

Defaults: capture_dir = $TT_DS_COMBINE_CAPTURE_DIR or generated/combine_capture,
num_dispatch_groups = 4 (Galaxy 8x4). Idempotent: re-running on an already-fixed
file produces no change (because src_chip values are already in [0, 8) so
dividing by 4 floors them to {0,0,0,1,1,1,1,1} which is wrong — so we mark
fixed files via a "src_chip_remapped" config flag).
"""

import argparse
import os
from pathlib import Path

import torch


def fix_one(path: Path, num_dispatch_groups: int, dry_run: bool = False) -> bool:
    """Returns True if modified, False if already fixed or skipped."""
    blob = torch.load(path, weights_only=False, map_location="cpu")
    cfg = blob.get("config", {})

    if cfg.get("src_chip_remapped"):
        print(f"  [skip] {path} — already remapped")
        return False

    meta = blob.get("dispatched_metadata")
    if meta is None:
        print(f"  [skip] {path} — no dispatched_metadata key")
        return False

    src_before = meta[..., 0]
    unique_before = sorted(set(src_before.flatten().tolist()))[:10]

    # Apply remap in place. Slicing meta[..., 0] returns a view; integer div modifies storage.
    meta[..., 0] //= num_dispatch_groups

    src_after = meta[..., 0]
    unique_after = sorted(set(src_after.flatten().tolist()))[:10]

    cfg["src_chip_remapped"] = True
    cfg["src_chip_remap_divisor"] = num_dispatch_groups
    blob["config"] = cfg

    print(f"  {path}")
    print(f"    src_chip unique sample before: {unique_before}")
    print(f"    src_chip unique sample after:  {unique_after}")
    if not dry_run:
        torch.save(blob, path)
        print(f"    [saved]")
    else:
        print(f"    [dry run, not saved]")
    return True


def main():
    default_dir = os.getenv(
        "TT_DS_COMBINE_CAPTURE_DIR",
        str(Path(os.getenv("TT_METAL_HOME", ".")) / "generated" / "combine_capture"),
    )
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir", nargs="?", default=default_dir, help=f"Default: {default_dir}")
    ap.add_argument(
        "--num-dispatch-groups",
        type=int,
        default=4,
        help="Divisor for src_chip remap. 4 for Galaxy 8x4 (default).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    args = ap.parse_args()

    root = Path(args.capture_dir)
    if not root.exists():
        raise SystemExit(f"capture_dir does not exist: {root}")

    pts = sorted(root.glob("L*/col*.pt"))
    if not pts:
        raise SystemExit(f"No L*/col*.pt files found under {root}")
    print(f"Found {len(pts)} capture files under {root}")
    print(f"Applying src_chip //= {args.num_dispatch_groups} (dry_run={args.dry_run})")
    print()
    n_fixed = sum(fix_one(p, args.num_dispatch_groups, dry_run=args.dry_run) for p in pts)
    print()
    print(f"Done. Modified {n_fixed} / {len(pts)} files.")


if __name__ == "__main__":
    main()
