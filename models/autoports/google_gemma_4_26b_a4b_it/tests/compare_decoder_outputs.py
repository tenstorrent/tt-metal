# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare saved decoder outputs on CPU without importing TTNN or opening a device."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

THRESHOLD = 0.995


def load(prefix):
    report_path, tensor_path = prefix.with_suffix(".json"), prefix.with_suffix(".pt")
    report = json.loads(report_path.read_text())
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    checks = {}
    for row in report["decode"]["checks"]:
        position = row["position"]
        if position in checks and checks[position] != row:
            raise ValueError(f"Conflicting duplicate position {position} in {report_path}")
        checks[position] = row
    first, last = report["decode"]["positions"]
    positions = list(range(first, last + 1))
    if sorted(checks) != positions or len(tensors["decode"]) != len(positions):
        raise ValueError(f"Incomplete per-position output/check coverage in {report_path}")
    if report["decode"]["steps"] != len(positions):
        raise ValueError(f"Step count disagrees with recorded positions in {report_path}")
    provenance = {
        "report": str(report_path),
        "tensors": str(tensor_path),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "runtime_sha256": report.get("runtime_sha256"),
        "precision_policy": report.get("precision_policy"),
    }
    return report, tensors, checks, positions, provenance


def compare_tensor(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError(f"Output shapes differ: {reference.shape} versus {actual.shape}")
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    if not finite:
        return {"pcc": None, "passed": False, "finite": False, "max_abs_error": None}
    a, b = reference.flatten().double(), actual.flatten().double()
    max_error = float((a - b).abs().max())
    a, b = a - a.mean(), b - b.mean()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    value = float(torch.dot(a, b) / denominator) if denominator else float(torch.equal(reference, actual))
    return {"pcc": value, "passed": value >= THRESHOLD, "finite": True, "max_abs_error": max_error}


def compare(baseline, candidate):
    base_report, base_tensors, base_checks, positions, base_provenance = load(baseline)
    report, tensors, checks, candidate_positions, provenance = load(candidate)
    for key in ("layer_type", "length", "prefix_length", "real_weights"):
        if base_report.get(key) != report.get(key):
            raise ValueError(f"Workloads differ at {key}: {baseline} versus {candidate}")
    if positions != candidate_positions:
        raise ValueError(f"Decode positions differ: {baseline} versus {candidate}")
    rows = []
    for index, position in enumerate(positions):
        rows.append(
            {
                "position": position,
                **compare_tensor(base_tensors["decode"][index], tensors["decode"][index]),
                "baseline_hf_pcc": base_checks[position]["pcc"],
                "candidate_hf_pcc": checks[position]["pcc"],
            }
        )
    base_failed = {p for p in positions if not base_checks[p]["passed"]}
    failed = {p for p in positions if not checks[p]["passed"]}
    finite_pccs = [row["pcc"] for row in rows if row["pcc"] is not None]
    prefill = compare_tensor(base_tensors["prefill"], tensors["prefill"])
    return {
        "baseline": base_provenance,
        "candidate": provenance,
        "layer_type": report["layer_type"],
        "length": report["length"],
        "steps": len(positions),
        "prefill": prefill,
        "baseline_hf_failed_positions": sorted(base_failed),
        "candidate_hf_failed_positions": sorted(failed),
        "shared_hf_failed_positions": sorted(base_failed & failed),
        "new_hf_failed_positions": sorted(failed - base_failed),
        "recovered_hf_positions": sorted(base_failed - failed),
        "minimum_direct_decode_pcc": min(finite_pccs) if finite_pccs else None,
        "direct_failed_positions": [row["position"] for row in rows if not row["passed"]],
        "direct_passed": prefill["passed"] and all(row["passed"] for row in rows),
        "decode": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Common stem of baseline .json and .pt files")
    parser.add_argument("--candidate", type=Path, action="append", required=True, help="Candidate stem; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = [compare(args.baseline, candidate) for candidate in args.candidate]
    output = {
        "threshold": THRESHOLD,
        "method": "CPU float64 centered Pearson correlation of every saved output; no TTNN import",
        "input_identity": "Matching workload metadata is checked; saved output artifacts do not contain input hashes",
        "interpretation": "Shared HF failures are observed overlap, not proof of a shared numerical cause",
        "comparisons": results,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for candidate, result in zip(args.candidate, results):
        print(
            json.dumps(
                {
                    "candidate": str(candidate),
                    **{k: v for k, v in result.items() if k not in ("baseline", "candidate", "decode")},
                }
            )
        )


if __name__ == "__main__":
    main()
