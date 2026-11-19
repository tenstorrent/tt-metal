# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Refresh the host-only runtime audit and unchanged-linear-path provenance."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).parents[1]
    doc = root / "doc/functional_decoder"
    source = root / "tt/functional_decoder.py"
    audit = json.loads((doc / "fallback_audit.json").read_text())
    tree = ast.parse(source.read_text())
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    forbidden = []
    for name in audit["runtime_functions"]:
        for node in ast.walk(functions[name]):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                forbidden.append((name, ast.unparse(node)))
            if isinstance(node, ast.Name) and node.id == "torch":
                forbidden.append((name, "torch"))
            if isinstance(node, ast.Attribute) and node.attr in ("from_torch", "to_torch", "cpu", "numpy"):
                forbidden.append((name, ast.unparse(node)))
    assert not forbidden, forbidden
    audit.update(
        forbidden_calls=forbidden,
        python_audit_pass=True,
        decoder_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        audit_command="python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_source",
    )
    (doc / "fallback_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    baseline = "a407e7d3ec6a00693e1be369d8a395f63c9c2c62"
    relative = "models/autoports/qwen_qwen3_6_35b_a3b/tt/functional_decoder.py"
    old = ast.parse(subprocess.check_output(["git", "show", baseline + ":" + relative], text=True))
    old_functions = {n.name: n for n in ast.walk(old) if isinstance(n, ast.FunctionDef)}
    changed = sorted(n for n in old_functions if ast.dump(old_functions[n]) != ast.dump(functions[n]))
    assert changed == ["_full_attention", "new_kv_state", "prepare_prefill"], changed
    # Compare actual planning for both kinds, including every short start and
    # awkward lengths. Prepare_positions is replaced with a host-only stub.
    namespace = {}
    for label, fn in (("old", old_functions["prepare_prefill"]), ("new", functions["prepare_prefill"])):
        fn.name = label
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "planner", "exec"), namespace)
    from types import SimpleNamespace

    linear = SimpleNamespace(kind="linear_attention", config=SimpleNamespace(max_position_embeddings=262144))
    cases = 0
    for start in (*range(514), 261888, 262111, 262112, 262143):
        for length in (1, 7, 31, 32, 33, 255, 256, 257, 511, 512, 513):
            for chunk in (32, 64, 128, 256, 512):
                if start + length <= 262144:
                    args = dict(seq_len=length, start_pos=start, chunk_size=chunk)
                    assert namespace["old"](linear, **args) == namespace["new"](linear, **args)
                    cases += 1
    result = dict(
        baseline_commit=baseline,
        current_decoder_sha256=audit["decoder_sha256"],
        changed_functions=changed,
        unchanged_linear_runtime=True,
        linear_planning_equal_cases=cases,
        historical_linear_long_evidence="linear_long.json and logs/linear_long_01.log.gz",
        note="Baseline is the recorded incomplete checkpoint, not a fabricated historical run sidecar.",
    )
    (doc / "source_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SOURCE_AUDIT_PASS", cases, "linear planning cases")


if __name__ == "__main__":
    main()
