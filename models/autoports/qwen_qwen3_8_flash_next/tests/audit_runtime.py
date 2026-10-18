"""Static host-boundary audit of model-local runtime methods.

Run alongside dynamic no_runtime_host_conversion guards in integration tests.
Setup routines are explicit exclusions, listed in the report for review.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = {"from_state_dict", "make_cache", "prepare"}
BOUNDARIES = {"from_torch", "to_torch", "as_tensor", "numpy", "cpu", "tolist", "item"}


def audit():
    methods, excluded, violations, hashes = [], [], [], {}
    for path in sorted((ROOT / "tt").glob("*.py")):
        if path.name == "checkpoint.py":
            excluded.append({"file": str(path.relative_to(ROOT)), "method": "entire setup-only checkpoint module"})
            continue
        source = path.read_text()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(source.encode()).hexdigest()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.Module)):
                continue
            for function in node.body:
                if not isinstance(function, ast.FunctionDef):
                    continue
                identity = {"file": str(path.relative_to(ROOT)), "method": function.name, "line": function.lineno}
                if function.name in SETUP or (
                    path.name == "functional_decoder.py"
                    and isinstance(node, ast.ClassDef)
                    and node.name == "_PrefixView"
                ):
                    excluded.append(identity)
                    continue
                methods.append(identity)
                for expression in ast.walk(function):
                    if isinstance(expression, ast.Import):
                        if any(a.name.split(".")[0] in {"torch", "numpy"} for a in expression.names):
                            violations.append({**identity, "expression": ast.unparse(expression)})
                    if isinstance(expression, ast.Call):
                        name = ast.unparse(expression.func)
                        if name.startswith(("torch.", "numpy.", "np.")) or name.split(".")[-1] in BOUNDARIES:
                            violations.append({**identity, "expression": name, "line": expression.lineno})
    result = {
        "status": "pass" if not violations else "fail",
        "sha256": hashes,
        "runtime_methods": methods,
        "setup_exclusions": excluded,
        "violations": violations,
        "limits": "Static named-call audit; dynamic guards and source review are also required.",
    }
    out = ROOT / "doc/functional_decoder/runtime_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "methods": len(methods), "violations": violations}))
    assert not violations


if __name__ == "__main__":
    audit()
