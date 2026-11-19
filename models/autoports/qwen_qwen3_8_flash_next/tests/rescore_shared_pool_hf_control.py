"""Rescore saved CPU HF tensors without rerunning a model or altering raw evidence."""

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import torch

from .probe_shared_pool_generator import decode_geometry_comparison

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair(runs, comparison):
    parts = comparison["name"].split("/")
    label, step, requests = parts[0], comparison["step"], comparison["requests"]
    if label == "repeat":
        expected = runs[f"{parts[1]}/{parts[2]}"][step]
        actual = runs[f"{parts[1]}/repeat{parts[3]}"][step]
    else:
        repeat = parts[1]
        if label.startswith("combined_B1_") or label.startswith("B1_"):
            family = label.removeprefix("combined_").removeprefix("B1_").removesuffix("_vs_mixed")
            expected = {r: runs[f"{family}{r}/{repeat}"][step][r] for r in requests}
            actual = runs[f"mixed/{repeat}"][step]
        elif label == "padding_only":
            expected = {r: runs[f"solo{r}/{repeat}"][step][r] for r in requests}
            actual = {r: runs[f"padded_solo{r}/{repeat}"][step][r] for r in requests}
        elif label in ("reversed_vs_mixed", "migrated_vs_mixed"):
            expected = runs[f"mixed/{repeat}"][step]
            actual = runs[f"{label.removesuffix('_vs_mixed')}/{repeat}"][step]
        elif label.startswith("survivor_vs_"):
            expected = runs[f"{label.removeprefix('survivor_vs_')}/{repeat}"][step]
            actual = runs[f"survivor/{repeat}"][step]
        else:
            raise ValueError(f"Unrecognized comparison {comparison['name']}")
    return tuple(torch.stack([rows[r]["logits"] for r in requests]) for rows in (expected, actual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a rescore artifact")
    metadata = json.loads(args.input.read_text())
    tensors = Path(metadata["tensor_artifact"]["path"])
    assert digest(tensors) == metadata["tensor_artifact"]["sha256"]
    torch.set_num_threads(metadata["backend"]["threads"])
    torch.set_num_interop_threads(metadata["backend"]["interop_threads"])
    runs = torch.load(tensors, map_location="cpu", weights_only=False)["runs"]
    archived = ROOT / "doc/full_model/sources/shared_pool_geometry_scorer_v1.py.txt"
    tree = ast.parse(archived.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode_geometry_comparison")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(archived), "exec"), namespace)
    old_scorer = namespace["decode_geometry_comparison"]
    comparisons = []
    for comparison in metadata["comparisons"]:
        expected, actual = pair(runs, comparison)
        raw = comparison["frozen_geometry_gate"]
        assert old_scorer(expected, actual) == raw, comparison["name"]
        corrected = decode_geometry_comparison(expected, actual)
        changed = [key for key in raw if raw[key] != corrected[key]]
        assert set(changed) <= {"top1_agreement", "passed"}, (comparison["name"], changed)
        comparisons.append(
            {
                "name": comparison["name"],
                "phase": comparison["phase"],
                "step": comparison["step"],
                "requests": comparison["requests"],
                "raw_frozen_score": raw,
                "corrected_score": corrected,
                "changed_fields": changed,
                "reference_argmax": expected.argmax(-1).tolist(),
                "actual_argmax": actual.argmax(-1).tolist(),
                "actual_global_max_tie_ids": [(row == row.max()).nonzero().flatten().tolist() for row in actual],
            }
        )
    report = {
        "scope": "CPU rescore only; no HF or TT execution and no gate waiver",
        "input": str(args.input),
        "input_sha256": digest(args.input),
        "tensor_path": str(tensors),
        "tensor_sha256": digest(tensors),
        "old_scorer_sha256": digest(archived),
        "new_scorer_sha256": digest(ROOT / "tests/probe_shared_pool_generator.py"),
        "rescore_source_sha256": digest(__file__),
        "command": [sys.executable, *sys.argv],
        "threads": torch.get_num_threads(),
        "comparison_count": len(comparisons),
        "all_raw_scores_reproduced": True,
        "raw_failures": sum(not c["raw_frozen_score"]["passed"] for c in comparisons),
        "corrected_failures": sum(not c["corrected_score"]["passed"] for c in comparisons),
        "changed_comparisons": sum(bool(c["changed_fields"]) for c in comparisons),
        "allowed_changes_only_top1_and_derived_pass": True,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}))


if __name__ == "__main__":
    main()
