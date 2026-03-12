# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Print first 20 values and statistics (mean, min, max, std, etc.) for text embedding cache files.
Usage:
  python print_text_emb_stats.py
  python print_text_emb_stats.py path/to/cache1.pt path/to/cache2.pt
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Default paths relative to this script
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FILES = [
    _SCRIPT_DIR / "out_inference" / "text_emb_cache_ttnn.pt",
    _SCRIPT_DIR / "out_inference" / "text_emb_cache_torch.pt",
]


def _stats(t: torch.Tensor, label: str = "") -> None:
    """Print first 20 values (flattened) and mean, min, max, std, median, quartiles."""
    flat = t.double().flatten()
    n = flat.numel()
    prefix = f"  {label}: " if label else "  "
    print(f"{prefix}shape: {tuple(t.shape)}  numel: {n}")
    if n > 0:
        first20 = flat[:20].tolist()
        print(f"  first 20 values: {first20}")
        print(f"  min:    {flat.min().item():.6g}")
        print(f"  max:    {flat.max().item():.6g}")
        print(f"  mean:   {flat.mean().item():.6g}")
        print(f"  std:    {flat.std().item():.6g}")
        q25 = torch.quantile(flat, 0.25).item()
        q50 = torch.quantile(flat, 0.50).item()
        q75 = torch.quantile(flat, 0.75).item()
        print(f"  median (q50): {q50:.6g}")
        print(f"  q25: {q25:.6g}  q75: {q75:.6g}")
    print()


def _compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors (flattened). Returns nan if shapes differ or std is zero."""
    if math.prod(a.shape) != math.prod(b.shape):
        return float("nan")
    a_flat = a.detach().flatten().to(torch.float64)
    b_flat = b.detach().flatten().to(torch.float64)
    cov = torch.cov(torch.stack([a_flat, b_flat])).numpy()
    std_a = math.sqrt(cov[0, 0])
    std_b = math.sqrt(cov[1, 1])
    if std_a < 1e-12 or std_b < 1e-12:
        return float("nan")
    return float(cov[0, 1] / (std_a * std_b))


def _print_obj(obj, path: Path) -> None:
    """Print stats for a loaded .pt object (tensor or dict of tensors)."""
    if isinstance(obj, torch.Tensor):
        _stats(obj, path.name)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                _stats(v, f"{path.name}['{k}']")
            else:
                print(f"  {path.name}['{k}']: (not a tensor) {type(v).__name__}")
        return
    print(f"  {path.name}: unsupported type {type(obj).__name__}\n")


def _get_tensors(obj):
    """Return dict of key -> tensor for all tensors in obj (dict or single tensor)."""
    if isinstance(obj, torch.Tensor):
        return {"": obj}
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    return {}


def main() -> None:
    paths = [Path(p) for p in (sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_FILES)]
    loaded = []
    for p in paths:
        if not p.is_file():
            print(f"File not found: {p}\n")
            continue
        print("=" * 60)
        print(p)
        print("=" * 60)
        obj = torch.load(p, map_location="cpu", weights_only=False)
        _print_obj(obj, p)
        loaded.append((p, obj))

    if len(loaded) == 2:
        p1, obj1 = loaded[0]
        p2, obj2 = loaded[1]
        tensors1 = _get_tensors(obj1)
        tensors2 = _get_tensors(obj2)
        common_keys = sorted(set(tensors1.keys()) & set(tensors2.keys()))
        if common_keys:
            print("=" * 60)
            print("PCC (Pearson correlation coefficient)")
            print("=" * 60)
            for k in common_keys:
                t1, t2 = tensors1[k], tensors2[k]
                pcc = _compute_pcc(t1, t2)
                label = k if k else "(tensor)"
                print(f"  {label}: PCC = {pcc * 100:.4f} %")
            print()

    print("Done.")


if __name__ == "__main__":
    main()
