# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Resolve inherited runtime methods and inspect direct host-conversion calls."""

import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).parents[1]
    methods, helpers, hashes = {}, {}, {}
    base_methods = {}
    for name in ("optimized_decoder", "multichip_decoder"):
        path = root / "tt" / f"{name}.py"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                helpers[node.name] = node
            if isinstance(node, ast.ClassDef) and node.name.endswith("Decoder"):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        methods[method.name] = method
                        if name == "optimized_decoder":
                            base_methods[method.name] = method
    nodes = methods | helpers | {"base." + k: v for k, v in base_methods.items()}
    pending, seen, violations = ["decode_forward", "prefill_forward"], set(), []
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(nodes[name]):
            if isinstance(node, ast.Call):
                fn = ast.unparse(node.func)
                called = (
                    fn.removeprefix("self.")
                    if fn.startswith("self.")
                    else ("base." + fn.removeprefix("super().") if fn.startswith("super().") else fn)
                )
                if called in nodes:
                    pending.append(called)
                if fn.startswith("torch.") or fn in (
                    "ttnn.from_torch",
                    "ttnn.to_torch",
                    "ttnn.as_tensor",
                    "ttnn.to_device",
                    "ttnn.full",
                    "ttnn.copy_host_to_device_tensor",
                ):
                    violations.append({"function": name, "line": node.lineno, "call": fn})
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                violations.append({"function": name, "line": node.lineno, "import": ast.unparse(node)})
    result = {
        "source_hashes": hashes,
        "reachable_functions": sorted(seen),
        "violations": violations,
        "scope": "Direct Python methods, baseline helpers, explicit super calls; CCLManager and native ops require source/runtime inspection.",
    }
    (root / "doc/multichip_decoder/runtime_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("STATIC_RUNTIME_AUDIT", len(seen), "functions", len(violations), "violations")
    assert not violations, violations


if __name__ == "__main__":
    main()
