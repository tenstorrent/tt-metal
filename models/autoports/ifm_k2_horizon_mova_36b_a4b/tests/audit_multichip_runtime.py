# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resolve runtime self/super calls and prohibit host tensor access/conversion."""

import argparse
import ast
import hashlib
import inspect
import json
import textwrap
from pathlib import Path

from ..tt.multichip_decoder import MultichipDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    mro = MultichipDecoder.__mro__

    def resolve(name, start=0):
        for index in range(start, len(mro)):
            if name in mro[index].__dict__:
                method = getattr(mro[index], name)
                if inspect.isfunction(method):
                    return index, name, method
        return None

    pending = [resolve(name) for name in ("prefill_forward", "decode_forward")]
    seen = {}
    violations = []
    prohibited = {
        "from_torch",
        "to_torch",
        "as_tensor",
        "to_device",
        "copy_host_to_device_tensor",
        "cpu",
        "numpy",
        "item",
        "tolist",
    }
    while pending:
        item = pending.pop()
        if item is None:
            continue
        index, name, method = item
        key = f"{mro[index].__name__}.{name}"
        if key in seen:
            continue
        source = textwrap.dedent(inspect.getsource(method))
        tree = ast.parse(source)
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] in ("torch", "numpy") for a in node.names):
                violations.append([key, "host import"])
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in ("torch", "numpy"):
                violations.append([key, "host import"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func
                calls.append(ast.unparse(attr))
                if attr.attr in prohibited:
                    violations.append([key, ast.unparse(attr)])
                if isinstance(attr.value, ast.Name) and attr.value.id == "self":
                    pending.append(resolve(attr.attr))
                if (
                    isinstance(attr.value, ast.Call)
                    and isinstance(attr.value.func, ast.Name)
                    and attr.value.func.id == "super"
                ):
                    pending.append(resolve(attr.attr, index + 1))
        seen[key] = {
            "file": inspect.getfile(method),
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "calls": calls,
        }
    assert any(
        name.endswith("._experts_indexed") for name in seen
    ), "Audit must reach through super() into indexed expert dispatch"
    result = {
        "status": "pass" if not violations else "fail",
        "entrypoints": ["prefill_forward", "decode_forward"],
        "resolved_methods": seen,
        "violations": violations,
        "dynamic_guard": "multichip_test_utils.device_only plus throw_exception_on_fallback=True in hardware harnesses",
    }
    path = Path(__file__).resolve().parents[1] / "doc/multichip_decoder/runtime_audit.json"
    path = args.output or path
    path.write_text(json.dumps(result, indent=2) + "\n")
    assert not violations, violations
    print("PASS", len(seen), "runtime methods including inherited super dispatch")


if __name__ == "__main__":
    main()
