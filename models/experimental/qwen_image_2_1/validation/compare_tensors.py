# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare captured TT tensors with the same-stage CUDA reference tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def compare(reference: torch.Tensor, candidate: torch.Tensor, atol: float, rtol: float) -> dict[str, object]:
    if reference.shape != candidate.shape:
        return {"status": "shape_mismatch", "reference_shape": list(reference.shape), "tt_shape": list(candidate.shape)}
    ref = reference.to(torch.float32).reshape(-1)
    tt = candidate.to(torch.float32).reshape(-1)
    finite = torch.isfinite(ref) & torch.isfinite(tt)
    if not finite.all():
        return {"status": "nonfinite", "finite_fraction": float(finite.float().mean())}
    delta = (tt - ref).abs()
    tolerance = atol + rtol * ref.abs()
    centered_ref = ref.double() - ref.double().mean()
    centered_tt = tt.double() - tt.double().mean()
    denominator = torch.linalg.vector_norm(centered_ref) * torch.linalg.vector_norm(centered_tt)
    pcc = float(torch.dot(centered_ref, centered_tt) / denominator) if denominator > 0 else None
    relative_rms = float(torch.sqrt(torch.mean((tt - ref) ** 2)) / torch.sqrt(torch.mean(ref**2)).clamp_min(1e-30))
    return {
        "status": "pass" if bool((delta <= tolerance).all()) else "fail",
        "elements": ref.numel(),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "relative_rms_error": relative_rms,
        "pcc": pcc,
        "fraction_within_tolerance": float((delta <= tolerance).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--tt-dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-all", action="store_true", help="also fail for CUDA stages without a TT capture")
    args = parser.parse_args()
    manifest = json.loads((args.cuda_dir / "manifest.json").read_text())
    reference_paths = {entry["path"] for entry in manifest["captures"]}
    candidate_paths = {path.relative_to(args.tt_dir).as_posix() for path in args.tt_dir.rglob("*.pt")}
    if not candidate_paths:
        raise ValueError(f"no TT captures found in {args.tt_dir}")
    rows = []
    selected_paths = candidate_paths | (reference_paths if args.require_all else set())
    for relative in sorted(selected_paths):
        if relative not in reference_paths:
            rows.append({"path": relative, "status": "unknown_reference"})
            continue
        tt_path = args.tt_dir / relative
        if not tt_path.is_file():
            rows.append({"path": relative, "status": "missing"})
            continue
        reference = torch.load(args.cuda_dir / relative, map_location="cpu", weights_only=True)
        candidate = torch.load(tt_path, map_location="cpu", weights_only=True)
        rows.append({"path": relative, **compare(reference, candidate, args.atol, args.rtol)})
    result = {
        "cuda_manifest": str(args.cuda_dir / "manifest.json"),
        "tt_dir": str(args.tt_dir),
        "counts": {status: sum(row["status"] == status for row in rows) for status in {row["status"] for row in rows}},
        "rows": rows,
    }
    report = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    print(report)
    if any(row["status"] != "pass" for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
