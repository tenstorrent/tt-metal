# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Static companion to the runner's guarded measured device-only calls."""

import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "tt/fused_decoder.py"
module = ast.parse(path.read_text())
allowed_setup = {"from_state_dict", "allocate_cache"}
methods = []
violations = []
for cls in module.body:
    if not isinstance(cls, ast.ClassDef):
        continue
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name in allowed_setup:
            continue
        methods.append(fn.name)
        for node in ast.walk(fn):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                violations.append((fn.name, node.lineno, "runtime import"))
            if isinstance(node, ast.Call):
                expr = ast.unparse(node.func)
                if expr.startswith("torch.") or expr in (
                    "ttnn.from_torch",
                    "ttnn.to_torch",
                    "ttnn.as_tensor",
                    "ttnn.from_device",
                ):
                    violations.append((fn.name, node.lineno, expr))
report = {
    "file": str(path.relative_to(root)),
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "runtime_methods": methods,
    "setup_boundaries": sorted(allowed_setup),
    "violations": violations,
    "runtime_guard": "run_fused.device_only rejects TTNN host conversions and Python torch function entry during each measured forward",
    "status": "clean" if not violations else "failed",
}
(root / "doc/fused_decoder/runtime_audit.json").write_text(json.dumps(report, indent=2) + "\n")
assert not violations, violations
print(json.dumps(report, indent=2))
