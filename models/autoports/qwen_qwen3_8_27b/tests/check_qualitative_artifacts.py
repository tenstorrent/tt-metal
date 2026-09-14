# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only shared-output checks, including the generated Fibonacci example."""

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=Path, required=True)
    args = parser.parse_args()
    checker = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/check_degenerate_output.py"
    spec = importlib.util.spec_from_file_location("stage7_degeneracy", checker)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report = module.Report()
    rows, sources = {}, {}
    names = ["tt_qualitative.json", "tt_qualitative_extended.json"]
    if (args.doc / "tt_qualitative_story_2048.json").exists():
        names.append("tt_qualitative_story_2048.json")
    for name in names:
        path = args.doc / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text())
        assert data["chat_template"] and data["prompt_mode"] == "chat"
        for row in data["outputs"]:
            for label, key in (("TT", "text"), ("HF", "hf_text")):
                module.check_completion(report, artifact=path, label=f"{label} prompt{row['prompt_id']}", text=row[key])
            rows[row["prompt_id"]] = row
    assert set(rows) == set(range(6))
    code = re.search(r"```python\n(.*?)```", rows[5]["text"].split("</think>")[-1], re.S).group(1)
    tree = ast.parse(code)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert not tree.body[0].decorator_list and not tree.body[0].args.defaults
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal))
        if isinstance(node, ast.Attribute):
            assert isinstance(node.value, ast.Name) and node.value.id == "fib" and node.attr == "append"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id == "range"
    namespace = {"__builtins__": {"range": range}}
    exec(compile(tree, "generated_fibonacci", "exec"), namespace)
    cases = {-1: [], 0: [], 1: [0], 2: [0, 1], 8: [0, 1, 1, 2, 3, 5, 8, 13]}
    for count, expected in cases.items():
        assert namespace["fibonacci"](count) == expected
    story_extension = None
    if "tt_qualitative_story_2048.json" in names:
        shorter = json.loads((args.doc / "tt_qualitative_extended.json").read_text())
        shorter = next(row for row in shorter["outputs"] if row["prompt_id"] == 2)
        assert rows[2]["tokens"][: len(shorter["tokens"])] == shorter["tokens"]
        assert rows[2]["text"].endswith("<|im_end|>")
        story_extension = dict(prefix_tokens_equal=len(shorter["tokens"]), completed_tokens=len(rows[2]["tokens"]))
    result = dict(
        checker_source=str(checker),
        checker_sha256=hashlib.sha256(checker.read_bytes()).hexdigest(),
        sources_sha256=sources,
        exit_code=report.exit_code,
        findings=[dataclasses.asdict(row) for row in report.findings],
        measured=report.measured,
        fibonacci_generated_code_tests=dict(passed=True, inputs=list(cases)),
        completion_lengths={str(key): len(row["tokens"]) for key, row in rows.items()},
        completion_eos={str(key): row["text"].endswith("<|im_end|>") for key, row in rows.items()},
        story_extension=story_extension,
        story_budget_note="HF and TT1024-token controls both truncate. Any2048 TT extension is completion evidence, not a matched-budget comparison.",
        command=" ".join(sys.argv),
    )
    (args.doc / "qualitative_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(exit_code=report.exit_code, findings=result["findings"])))
    assert report.exit_code == 0, result["findings"]


if __name__ == "__main__":
    main()
