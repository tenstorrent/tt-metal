# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Static fallback audit of TP4 and its inherited single-chip runtime."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = {"from_state_dict", "allocate_cache", "_init_subdevice"}
CONVERSIONS = {"ttnn.from_torch", "ttnn.to_torch", "ttnn.as_tensor", "ttnn.from_device"}
rows = []
for name in ("multichip_decoder", "optimized_decoder", "prepared_prefill"):
    path = ROOT / "tt" / (name + ".py")
    tree = ast.parse(path.read_text())
    methods, violations, setup_hits = [], [], []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            if fn.name not in SETUP:
                methods.append(cls.name + "." + fn.name)
            for node in ast.walk(fn):
                hit = None
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    hit = "import: " + ast.unparse(node)
                elif isinstance(node, ast.Call):
                    expr = ast.unparse(node.func)
                    if (
                        expr.startswith(("torch.", "numpy.", "np."))
                        or expr in CONVERSIONS
                        or expr.endswith((".cpu", ".numpy", ".item"))
                    ):
                        hit = expr
                if hit:
                    (setup_hits if fn.name in SETUP else violations).append(
                        dict(function=fn.name, line=node.lineno, call=hit)
                    )
    rows.append(
        dict(
            path=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            setup_boundaries=sorted(SETUP),
            setup_hits=setup_hits,
            runtime_methods=methods,
            violations=violations,
            status="clean" if not violations else "failed",
        )
    )
report = dict(
    files=rows,
    runtime_guard="Required runner forwards reject Torch function entry and TTNN host conversions; setup/reference/comparison excluded.",
    status="clean" if all(r["status"] == "clean" for r in rows) else "failed",
)
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=ROOT / "doc/multichip_decoder/runtime_fallback_static_audit.json")
parser.parse_args().output.write_text(json.dumps(report, indent=2) + "\n")
assert report["status"] == "clean", report
print("MULTICHIP_RUNTIME_AUDIT_CLEAN")
