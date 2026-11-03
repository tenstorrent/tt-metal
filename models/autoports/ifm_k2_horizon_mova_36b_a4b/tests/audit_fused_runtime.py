# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audit the complete decoder method closure, excluding the setup boundary."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "tt/fused_decoder.py"
tree = ast.parse(source.read_text())
methods = {
    n.name: n for cls in tree.body if isinstance(cls, ast.ClassDef) for n in cls.body if isinstance(n, ast.FunctionDef)
}
pending = ["prefill_forward", "decode_forward"]
visited = set()
ops = set()
violations = []
while pending:
    name = pending.pop()
    if name in visited:
        continue
    visited.add(name)
    for node in ast.walk(methods[name]):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append((name, node.lineno, "runtime import"))
        if isinstance(node, ast.Call):
            target = ast.unparse(node.func)
            if target.startswith("self.") and target[5:] in methods:
                pending.append(target[5:])
            if (
                target.startswith("torch.")
                or target in ("ttnn.from_torch", "ttnn.to_torch", "ttnn.as_tensor", "ttnn.to_device")
                or target.endswith((".cpu", ".numpy", ".item"))
            ):
                violations.append((name, node.lineno, target))
            if target.startswith("ttnn."):
                ops.add(target)
assert "from_state_dict" not in visited
report = dict(
    source=str(source.relative_to(ROOT)),
    sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    audited_methods=sorted(visited),
    device_ops=sorted(ops),
    violations=violations,
    boundary="from_state_dict and explicit test input/output boundaries",
    wrapper_audit="TTNN device primitives; no model helper imports other than LightweightModule. "
    "Traced decode prohibits hidden host writes/reads. Test runtime patches reject "
    "from_torch, to_torch, and as_tensor inside both forward paths.",
)
(ROOT / "doc/fused_decoder/runtime_audit.json").write_text(json.dumps(report, indent=2))
assert not violations, violations
print("RUNTIME_AUDIT_PASS", len(visited), "methods", len(ops), "TTNN operations")
