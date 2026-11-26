# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A/B of two ``GLM53_MHC_DUMP`` directories (prefill wraps of two mHC implementations): per wrap, rel rms of
the input streams, the sublayer input, the sublayer output and the output streams, and the first wrap/stage
that diverges. Usage: python mhc_dump_diff.py <dir_a> <dir_b> [tol]."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return ((a - b).norm() / a.norm().clamp_min(1e-30)).item()


def main() -> None:
    da, db = Path(sys.argv[1]), Path(sys.argv[2])
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
    keys = sorted({p.name.rsplit("_", 1)[0] for p in da.glob("L*_*_*.pt")})
    first = None
    print(f"| wrap | in | h (sublayer in) | y (sublayer out) | out |\n|---|---|---|---|---|")
    for k in keys:
        row = []
        for stage in ("in", "h", "y", "out"):
            fa, fb = da / f"{k}_{stage}.pt", db / f"{k}_{stage}.pt"
            if fa.exists() and fb.exists():
                r = rel(torch.load(fa), torch.load(fb))
                row.append(f"{r:.2e}")
                if first is None and r > tol:
                    first = (k, stage, r)
            else:
                row.append("-")
        print(f"| {k} | " + " | ".join(row) + " |")
    print(f"\nfirst divergence > {tol}: {first}")


if __name__ == "__main__":
    main()
