# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare a TT candidate capture against the recorded native CUDA oracle."""
import argparse
import json
from pathlib import Path
from .manifest import load_manifest, load_tensor
from .metrics import compare_tensors


def validate_capture(
    reference_directory,
    candidate_directory,
    *,
    min_pcc=0.99,
    max_relative_rms=None,
    max_abs=None,
    tensor_names=None,
):
    reference = load_manifest(reference_directory)
    candidate = load_manifest(candidate_directory, require_cuda=False)
    for field in ("checkpoint", "generation"):
        if reference.get(field) != candidate.get(field):
            raise ValueError(f"Capture {field} differs; replay the oracle inputs")
    if candidate.get("backend") != "ttnn":
        raise ValueError("Candidate must declare backend='ttnn'")
    names = list(reference["tensors"]) if tensor_names is None else list(tensor_names)
    if not names:
        raise ValueError("No tensors selected")
    unknown = set(names) - reference["tensors"].keys()
    if unknown:
        raise ValueError(f"Unknown reference tensors: {sorted(unknown)}")
    results = {}
    for name in names:
        if name not in candidate["tensors"]:
            results[name] = {"passed": False, "reason": "missing candidate tensor"}
            continue
        results[name] = compare_tensors(
            load_tensor(reference_directory, reference, name),
            load_tensor(candidate_directory, candidate, name),
            min_pcc=min_pcc,
            max_relative_rms=max_relative_rms,
            max_abs=max_abs,
        ).to_dict()
    extra = (
        sorted(set(candidate["tensors"]) - reference["tensors"].keys())
        if tensor_names is None
        else []
    )
    return {
        "passed": all(r["passed"] for r in results.values()) and not extra,
        "thresholds": {
            "min_pcc": min_pcc,
            "max_relative_rms": max_relative_rms,
            "max_abs": max_abs,
        },
        "tensors": results,
        "unexpected_tensors": extra,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--min-pcc", type=float, default=0.99)
    parser.add_argument("--max-relative-rms", type=float)
    parser.add_argument("--max-abs", type=float)
    parser.add_argument(
        "--tensor",
        action="append",
        help="Exact tensor key; repeat to select component outputs",
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = validate_capture(
        args.reference,
        args.candidate,
        min_pcc=args.min_pcc,
        max_relative_rms=args.max_relative_rms,
        max_abs=args.max_abs,
        tensor_names=args.tensor,
    )
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
