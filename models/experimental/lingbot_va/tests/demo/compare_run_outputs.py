# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Compare PCC and statistics between latents and actions from two inference run directories.
Uses assert_quality from models.tt_dit.utils.check (same as test_transformer_wan.py) for PCC/CCC/RMSE.
Usage:
  python compare_run_outputs.py <dir1> <dir2>
  python compare_run_outputs.py  # uses default dirs below
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure tt-metal root is on path when running from various working directories
_tt_metal_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(_tt_metal_root) not in sys.path:
    sys.path.insert(0, str(_tt_metal_root))

from models.tt_dit.utils.check import assert_quality


def stats(name: str, t: torch.Tensor) -> None:
    """Print min, max, mean, std, median and quartiles."""
    t_flat = t.double().flatten()
    q25 = float(torch.quantile(t_flat, 0.25).item())
    q50 = float(torch.quantile(t_flat, 0.50).item())
    q75 = float(torch.quantile(t_flat, 0.75).item())
    print(f"  {name}:")
    print(f"    shape: {tuple(t.shape)}")
    print(f"    min:   {t_flat.min().item():.6g}")
    print(f"    max:   {t_flat.max().item():.6g}")
    print(f"    mean:  {t_flat.mean().item():.6g}")
    print(f"    std:   {t_flat.std().item():.6g}")
    print(f"    median (q50): {q50:.6g}")
    print(f"    q25:   {q25:.6g}  q75:   {q75:.6g}")


def main() -> None:
    default_base = Path(__file__).resolve().parent / "out_inference" / "real"
    default_dirs = [
        default_base / "Lift the cup from the table_20260311_151541",
        default_base / "Lift the cup from the table_20260311_152644",
    ]
    parser = argparse.ArgumentParser(description="Compare PCC and stats between two run dirs")
    parser.add_argument(
        "dirs",
        nargs="*",
        type=Path,
        default=default_dirs,
        help="Two directories containing latents_0.pt and actions_0.pt (default: two recent demo runs)",
    )
    args = parser.parse_args()
    if len(args.dirs) != 2:
        print("Usage: compare_run_outputs.py <dir1> <dir2>", file=sys.stderr)
        sys.exit(1)
    d1, d2 = args.dirs[0], args.dirs[1]
    for d in (d1, d2):
        if not d.is_dir():
            print(f"Error: not a directory: {d}", file=sys.stderr)
            sys.exit(1)
    files = ["latents_0.pt", "actions_0.pt"]
    for f in files:
        if not (d1 / f).exists():
            print(f"Error: missing {d1 / f}", file=sys.stderr)
            sys.exit(1)
        if not (d2 / f).exists():
            print(f"Error: missing {d2 / f}", file=sys.stderr)
            sys.exit(1)

    print("=" * 60)
    print("Comparing run outputs")
    print("=" * 60)
    print(f"Dir1: {d1}")
    print(f"Dir2: {d2}")
    print()

    for f in files:
        key = f.replace(".pt", "").replace("_", " ").capitalize()
        t1 = torch.load(d1 / f, map_location="cpu", weights_only=True)
        t2 = torch.load(d2 / f, map_location="cpu", weights_only=True)
        if not isinstance(t1, torch.Tensor):
            t1 = t1 if isinstance(t1, torch.Tensor) else torch.tensor(t1)
        if not isinstance(t2, torch.Tensor):
            t2 = t2 if isinstance(t2, torch.Tensor) else torch.tensor(t2)

        print(f"--- {key} ---")
        if t1.numel() == t2.numel():
            assert_quality(t1, t2)
        else:
            print("(Shapes differ; skipping assert_quality)")
        print()
        print("Statistics Dir1:")
        stats("tensor", t1)
        print()
        print("Statistics Dir2:")
        stats("tensor", t2)
        if t1.shape == t2.shape:
            diff = (t1.double() - t2.double()).float()
            print()
            print("Statistics Difference (Dir1 - Dir2):")
            stats("diff", diff)
        else:
            print()
            print("(Shapes differ; skipping difference statistics)")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
