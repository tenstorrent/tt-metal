"""Validate a generated operation against its golden test API contract.

Performs fast, offline checks (no device needed) to catch signature mismatches
before running the full golden test suite on device.

Usage:
    python3 -m eval.validate_contract eval/golden_tests/layer_norm_rm/

Exit codes:
    0  - All checks passed (or no contract found)
    1  - Contract validation failed

Outputs validation_report.json to stdout with details.
"""

import importlib
import inspect
import json
import re
import sys
from pathlib import Path


def parse_contract(contract_path: Path) -> dict:
    """Extract structured info from api_contract.md."""
    text = contract_path.read_text()
    contract = {}

    # Extract import path
    import_match = re.search(r"from ([\w.]+) import (\w+)", text)
    if import_match:
        contract["module"] = import_match.group(1)
        contract["function"] = import_match.group(2)

    # Extract function signature — look for def line in code block
    sig_match = re.search(r"def (\w+)\((.*?)\)", text, re.DOTALL)
    if sig_match:
        contract["signature_raw"] = sig_match.group(0)
        params_raw = sig_match.group(2)

        # Parse parameter names and defaults
        params = []
        for line in params_raw.split("\n"):
            line = line.strip().rstrip(",")
            if not line or line.startswith("#") or line == "*" or line.startswith("*,"):
                continue

            param_match = re.match(r"(\*?\*?\w+)(?:\s*:\s*[\w.\[\]]+)?\s*(?:=\s*(.+))?", line)
            if param_match:
                name = param_match.group(1)
                default = param_match.group(2)
                if default:
                    default = default.split("#")[0].strip().rstrip(",")
                params.append({"name": name, "default": default})

        contract["params"] = params

    # Extract valid call patterns
    func_name = contract.get("function")
    patterns = []
    if func_name:
        for match in re.finditer(rf"{re.escape(func_name)}\(([^)]*)\)", text):
            args = match.group(1).strip()
            if args and not args.startswith("input_tensor"):  # skip the def line
                patterns.append(args)
    contract["call_patterns"] = patterns

    return contract


def validate_operation(contract: dict, contract_path: Path) -> list:
    """Validate the generated operation matches the contract. Returns list of issues."""
    issues = []
    module_name = contract.get("module")
    func_name = contract.get("function")

    if not module_name or not func_name:
        issues.append("Could not parse import path from contract")
        return issues

    # 1. Check importability
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        issues.append(f"ImportError: cannot import module '{module_name}': {e}")
        return issues

    # 2. Check function exists
    if not hasattr(mod, func_name):
        issues.append(f"Module '{module_name}' has no attribute '{func_name}'")
        return issues

    func = getattr(mod, func_name)
    if not callable(func):
        issues.append(f"'{func_name}' is not callable")
        return issues

    # 3. Check signature
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError) as e:
        issues.append(f"Cannot inspect signature of '{func_name}': {e}")
        return issues

    params = list(sig.parameters.values())
    expected_params = contract.get("params", [])

    # Check parameter names match
    actual_names = [p.name for p in params]
    expected_names = [p["name"] for p in expected_params]

    for exp_name in expected_names:
        if exp_name not in actual_names:
            issues.append(f"Missing parameter '{exp_name}' in function signature")

    # Check defaults for optional parameters
    for exp_param in expected_params:
        name = exp_param["name"]
        expected_default = exp_param.get("default")
        if expected_default is None:
            continue  # No default expected

        if name not in actual_names:
            continue  # Already reported above

        actual_param = sig.parameters[name]
        if actual_param.default is inspect.Parameter.empty:
            if expected_default == "None":
                issues.append(
                    f"Parameter '{name}' has no default but contract expects default=None "
                    f"(make it optional: {name}=None)"
                )
            elif expected_default:
                issues.append(f"Parameter '{name}' has no default but contract expects default={expected_default}")

    # Check keyword-only parameters
    raw_sig = contract.get("signature_raw", "")
    contract_text = contract_path.read_text()
    has_keyword_only_marker = "*,\n" in contract_text or "*, " in raw_sig

    if has_keyword_only_marker:
        for exp_param in expected_params:
            name = exp_param["name"]
            if name not in actual_names:
                continue

            actual_param = sig.parameters[name]
            if exp_param.get("default") is not None and actual_param.kind != inspect.Parameter.KEYWORD_ONLY:
                issues.append(
                    f"Parameter '{name}' should be keyword-only (use *, {name}=...) " f"but is {actual_param.kind.name}"
                )

    return issues


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m eval.validate_contract <golden_test_dir>", file=sys.stderr)
        sys.exit(1)

    test_dir = Path(sys.argv[1])
    contract_path = test_dir / "api_contract.md"

    if not contract_path.exists():
        # No contract = no validation (backwards compatible)
        result = {"status": "skipped", "reason": "No api_contract.md found", "issues": []}
        print(json.dumps(result, indent=2))
        sys.exit(0)

    contract = parse_contract(contract_path)

    if not contract.get("module"):
        result = {"status": "error", "reason": "Could not parse contract", "issues": ["Malformed api_contract.md"]}
        print(json.dumps(result, indent=2))
        sys.exit(1)

    issues = validate_operation(contract, contract_path)

    if issues:
        result = {
            "status": "failed",
            "module": contract.get("module"),
            "function": contract.get("function"),
            "issues": issues,
        }
        print(json.dumps(result, indent=2))
        sys.exit(1)
    else:
        result = {
            "status": "passed",
            "module": contract.get("module"),
            "function": contract.get("function"),
            "issues": [],
        }
        print(json.dumps(result, indent=2))
        sys.exit(0)


if __name__ == "__main__":
    main()
