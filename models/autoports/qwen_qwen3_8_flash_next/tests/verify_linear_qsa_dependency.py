# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Narrow CPU-only source closure for unaffected linear decoder records.

This proves one exact local QSA constructor substitution, never numerical or
whole-stage acceptance. Original full-file provenance remains unchanged.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from .verify_format19_dependency import FormatSourceClosure

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "tests/diagnostic_sources/qsa.py.pre_ordered_projection.txt"
BASELINE_SHA = "f0c57a4a700da794662d6888e0907a3b50989a3bc18a7082b891e92cc776673b"
RUNTIME_BASELINE = "tests/diagnostic_sources/linear_runtime_before_ordered.json"
RUNTIME_BASELINE_SHA = "08f69155bc2c1ae32862da5a11f70e17d1e4c74352597ac462a7df14e3417440"
ADDED_RUNTIME = {
    "tt/ordered_projection.py": "a952d765473f0cb3e68205d9b588a2344726fabae5b30a2107035c13b255e4e4",
    "tt/kernels/ordered_projection_compute.cpp": "c9701ee9c1acb1c209463cf6cc19b6acc4e4fa397a0ba87f1ff62c45f2df9c77",
    "tt/kernels/ordered_projection_reader.cpp": "f04e14870d5a81a6fd1b7b6922fbbe1eee9c8b196b77bcac53291fb0e45f992b",
    "tt/kernels/ordered_projection_writer.cpp": "72c000b5775221692fda5422db0f746d1c9ed1b436d36cd75a9269dc1b1e73a9",
}
CONFIG_SHA = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
CHANGES = (
    (
        "from .precise_projection import PreciseIndexerProjection",
        "from .ordered_projection import OrderedIndexerProjection",
    ),
    (
        "self.project_index_qk = PreciseIndexerProjection(loader)",
        "self.project_index_qk = OrderedIndexerProjection(loader)",
    ),
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_transition(original, current):
    require(digest(original) == BASELINE_SHA, "Unexpected historical QSA source")
    before = ast.parse(original)
    cls = next(node for node in before.body if isinstance(node, ast.ClassDef) and node.name == "QSA")
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    expected = original.decode()
    for old, new in CHANGES:
        require(expected.count(old) == 1, "Historical change site is not unique")
        require(old in ast.get_source_segment(original.decode(), constructor), "Change is outside QSA constructor")
        expected = expected.replace(old, new)
    require(current == expected.encode(), "Current QSA differs beyond the two authorized constructor lines")
    return {
        "historical_sha256": digest(original),
        "current_sha256": digest(current),
        "exact_replacements": [{"old": old, "new": new} for old, new in CHANGES],
        "module_import_time_code_unchanged": True,
        "new_import_is_function_local": True,
    }


def verify_branch(source):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FunctionalDecoder")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    constructor = ast.get_source_segment(source, methods["from_state_dict"])
    chunk = ast.get_source_segment(source, methods["_chunk"])
    require("kind = config.layer_types[layer_idx]" in constructor, "Layer-kind selection changed")
    require(
        'result.attn = GatedDeltaNet(loader, config) if kind == "linear_attention" else QSA(loader, config)'
        in constructor,
        "Attention constructor branch changed",
    )
    require(
        'result.chunked_attn = ChunkedQSA(result.attn) if kind == "indexed_attention" else None' in constructor,
        "Chunked-attention constructor branch changed",
    )
    require(
        'if self.kind == "linear_attention":\n            block = self.attn(mixed, conv_state=state.conv, recurrent_state=state.recurrent)'
        in chunk,
        "Linear execution branch changed",
    )
    return {"linear_kind_selects_gated_delta_net": True, "linear_kind_never_constructs_qsa": True}


def verify_record(root, artifact_path, provenance_path, format_binding=None):
    formatting = FormatSourceClosure(root, format_binding)
    verify_transition((ROOT / BASELINE).read_bytes(), (root / "tt/qsa.py").read_bytes())
    artifact = root / artifact_path
    provenance_file = root / provenance_path
    report = json.loads(artifact.read_text())
    provenance = json.loads(provenance_file.read_text())
    require(
        provenance.get("status") == "finished" and provenance.get("exit_code") == 0, "Run did not finish successfully"
    )
    layer = report.get("layer_idx", report.get("layer"))
    require(type(layer) is int and layer in (0, 1), "Record is not explicitly layer 0 or 1")
    require(report.get("passed") is True, "Record did not pass its declared gate")
    cwd = Path(provenance["cwd"])
    require(
        any((cwd / arg).resolve() == artifact.resolve() for arg in provenance["argv"]), "Artifact is not command-bound"
    )
    log = cwd / provenance["log"]
    require(digest(log.read_bytes()) == provenance["log_sha256"], "Recorded log identity changed")
    sources = provenance["source_sha256"]
    require(sources.get("tt/qsa.py") == BASELINE_SHA, "Recorded QSA source is not the preserved baseline")
    baseline_raw = (ROOT / RUNTIME_BASELINE).read_bytes()
    require(digest(baseline_raw) == RUNTIME_BASELINE_SHA, "Runtime baseline inventory changed")
    baseline = json.loads(baseline_raw)
    current_runtime = {
        str(path.relative_to(root))
        for path in (root / "tt").rglob("*")
        if path.is_file() and path.suffix in (".py", ".cpp", ".hpp", ".h")
    }
    require(current_runtime == set(baseline) | set(ADDED_RUNTIME), "Unexpected current runtime inventory")
    recorded_runtime = {name for name in sources if name.startswith("tt/")}
    require(
        set(baseline) <= recorded_runtime <= set(baseline) | set(ADDED_RUNTIME),
        "Unexpected historical runtime inventory",
    )
    for name, expected in ADDED_RUNTIME.items():
        formatting.verify(name, expected)
    argv = provenance["argv"]
    require(argv.count("--snapshot") == 1, "Expected exactly one recorded target snapshot")
    config_path = cwd / argv[argv.index("--snapshot") + 1] / "config.json"
    require(digest(config_path.read_bytes()) == CONFIG_SHA, "Target config identity changed")
    require(
        json.loads(config_path.read_text())["text_config"]["layer_types"][layer] == "linear_attention",
        "Target layer is not linear",
    )
    unchanged = {}
    for name, historical_hash in baseline.items():
        require(sources.get(name) == historical_hash, f"Recorded runtime does not match baseline: {name}")
        if name == "tt/qsa.py":
            continue
        formatting.verify(name, historical_hash)
        unchanged[name] = historical_hash
    verify_branch((root / "tt/functional_decoder.py").read_text())
    return {
        "artifact": str(artifact_path),
        "artifact_sha256": digest(artifact.read_bytes()),
        "provenance": str(provenance_path),
        "provenance_sha256": digest(provenance_file.read_bytes()),
        "log_sha256": provenance["log_sha256"],
        "layer": layer,
        "reported_passed": True,
        "literal_hf_strict_passed": report.get("literal_hf_strict_passed"),
        "historical_shared_runtime_sha256": unchanged,
        "historical_unreachable_added_runtime_sha256": ADDED_RUNTIME,
        "historical_unreachable_candidate_sha256": {
            name: sources[name] for name in sorted(recorded_runtime - set(baseline))
        },
        "config_sha256": CONFIG_SHA,
        "ordered_candidate_not_imported_by_linear_constructor": True,
        "format_source_closure": formatting.proof,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", nargs=2, action="append", required=True, metavar=("ARTIFACT", "PROVENANCE"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format-source-closure", type=Path)
    args = parser.parse_args()
    transition = verify_transition((ROOT / BASELINE).read_bytes(), (ROOT / "tt/qsa.py").read_bytes())
    branch = verify_branch((ROOT / "tt/functional_decoder.py").read_text())
    format_binding = json.loads(args.format_source_closure.read_text()) if args.format_source_closure else None
    records = [verify_record(ROOT, *pair, format_binding=format_binding) for pair in args.record]
    output = {
        "schema_version": 1,
        "scope": "Exact QSA-local constructor substitution is unreachable for recorded linear layers 0/1",
        "source_closure_passed": True,
        "stage_acceptance": "not_evaluated_review_required",
        "baseline": BASELINE,
        "runtime_baseline": RUNTIME_BASELINE,
        "runtime_baseline_sha256": RUNTIME_BASELINE_SHA,
        "transition": transition,
        "branch": branch,
        "records": records,
        "limitations": [
            "Does not waive numerical failures or reclassify controlled findings.",
            "Does not validate QSA execution or any shared computational change; optional exact formatting is separately proved.",
            "Record-specific harness and numerical gates remain separate requirements.",
        ],
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"source_closure_passed": True, "records": len(records)}))


if __name__ == "__main__":
    main()
