# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recheck saved acceptance vectors with the actual finite PCC gate, offline."""

import ast
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
runner = root.parents[1] / "tests/run_multichip_decoder.py"
tree = ast.parse(runner.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_pcc_values_pass")
namespace = {"math": math}
exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(runner), "exec"), namespace)
gate = namespace["_pcc_values_pass"]
for bad in (float("nan"), float("inf"), -float("inf")):
    assert not gate([0.999989, bad, 0.999980])
    assert not gate([bad, 0.999989])
assert gate([0.995, 1.0]) and not gate([0.995, 0.994999]) and not gate([])
failed = json.loads((root / "historical_k44_failure.json").read_text())
assert min(failed["cache_pcc"]) >= 0.995 and not gate(failed["cache_pcc"])
checks = []
for path in sorted(root.glob("final_*.json")):
    data = json.loads(path.read_text())
    if data.get("passed") is not True:
        continue
    vectors = {}
    for key in ("pcc", "cache_pcc", "prefill_pcc", "decode_pcc"):
        if data.get(key):
            vectors[key] = data[key]
    if data.get("comparisons"):
        vectors["comparisons"] = [item["pcc"] for item in data["comparisons"]]
    if not vectors:
        continue
    for key, values in vectors.items():
        assert gate(values), (path, key)
    checks.append(
        dict(
            artifact=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            vector_counts={key: len(values) for key, values in vectors.items()},
            finite_and_passing=True,
        )
    )
(root / "finite_gate_audit.json").write_text(
    json.dumps(
        dict(
            passed=True,
            runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
            actual_gate_source=ast.unparse(function),
            regression="A finite passing prefix followed by NaN or infinity is rejected.",
            historical_cache_nan_rejected="historical_k44_failure.json",
            accepted_artifacts=checks,
            scope="Offline acceptance-gate repair only; no change to device execution or captured measurements.",
        ),
        indent=2,
    )
    + "\n"
)
print(f"Finite gate regression and {len(checks)} accepted artifact vectors pass")
