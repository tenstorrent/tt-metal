# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Static reachable-method audit; runtime sentinels supplement this check."""

import ast
import hashlib
import json
import os
from pathlib import Path


def audit():
    stage = os.environ.get("QWEN_DECODER", "functional") + "_decoder"
    source = Path(__file__).parents[1] / ("tt/" + stage + ".py")
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name
        == {
            "fused_decoder": "FusedDecoder",
            "optimized_decoder": "OptimizedDecoder",
            "functional_decoder": "FunctionalDecoder",
        }[stage]
    )
    methods = {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    helpers = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    functions = {**helpers, **methods}
    pending = ["prefill_forward", "decode_forward"]
    seen = set()
    violations = []
    ops = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                violations.append({"method": name, "line": node.lineno, "reason": "runtime import"})
            if isinstance(node, ast.Call):
                fn = ast.unparse(node.func)
                if fn.startswith("self.") and fn[5:] in methods:
                    pending.append(fn[5:])
                elif fn in helpers:
                    pending.append(fn)
                if fn.startswith("torch.") or fn in ["ttnn.from_torch", "ttnn.to_torch", "ttnn.as_tensor", "ttnn.full"]:
                    violations.append({"method": name, "line": node.lineno, "call": fn})
                if fn.startswith("ttnn."):
                    ops.add(fn)
    result = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reachable_methods": sorted(seen),
        "ttnn_calls": sorted(ops),
        "violations": violations,
    }
    path = source.parents[1] / ("doc/" + stage + "/runtime_audit.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("RUNTIME_AUDIT", "PASS" if not violations else "FAIL", len(seen), "methods", len(ops), "TTNN operations")
    assert not violations


if __name__ == "__main__":
    audit()
