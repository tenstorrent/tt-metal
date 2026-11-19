# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Format a new triangular candidate and prove its one local import permutation."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
SOURCE = E / "v6_corrected_fma_full_v5_triangular_compact_silu_integration.py.snapshot"
EXPECTED = "85644b73bc1effb516c1fc7a5092cc443ef7d8a30a67a6635767a06f3eb7f858"
OUTPUT = E / "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot"
REPORT = E / "v6_triangular_formatted_integration_build.json"
BLACK = "/home/ttuser/.cache/pre-commit/repoefelr92y/py_env-python3/bin/black"
ISORT = "/home/ttuser/.cache/pre-commit/repodrn1fio9/py_env-python3/bin/isort"
AUTOFLAKE = "/home/ttuser/.cache/pre-commit/repodje5jyzf/py_env-python3/bin/autoflake"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(SOURCE) == EXPECTED and not OUTPUT.exists() and not REPORT.exists()
    OUTPUT.write_bytes(SOURCE.read_bytes())
    commands = [
        [AUTOFLAKE, "--remove-all-unused-imports", "--in-place", str(OUTPUT)],
        [ISORT, str(OUTPUT)],
        [BLACK, str(OUTPUT)],
        [AUTOFLAKE, "--remove-all-unused-imports", "--check-diff", str(OUTPUT)],
        [ISORT, "--check-only", "--diff", str(OUTPUT)],
        [BLACK, "--check", str(OUTPUT)],
    ]
    checks = []
    for command in commands:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        checks.append(
            dict(command=command, exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        )
    old, new = ast.parse(SOURCE.read_text()), ast.parse(OUTPUT.read_text())
    restored = copy.deepcopy(new)
    before = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "chunk64_mathematical_exp_table")
    after = next(
        n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == "chunk64_mathematical_exp_table"
    )
    original = [ast.unparse(n) for n in before.body[1:4]]
    formatted = [ast.unparse(n) for n in after.body[1:4]]
    assert original == ["import torch", "import struct", "from decimal import Decimal, localcontext"]
    assert formatted == ["import struct", "from decimal import Decimal, localcontext", "import torch"]
    assert all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in before.body[1:4] + after.body[1:4])
    after.body[1:4] = copy.deepcopy(before.body[1:4])
    assert ast.dump(restored) == ast.dump(old)
    assert sha(SOURCE) == EXPECTED
    report = dict(
        status="all_pinned_format_checks_pass",
        source_sha256=sha(__file__),
        input=str(SOURCE),
        input_sha256=EXPECTED,
        output=str(OUTPUT),
        output_sha256=sha(OUTPUT),
        allowed_change="Only three local imports reorder in chunk64_mathematical_exp_table; no import addition/removal, no arithmetic change",
        before_imports=original,
        after_imports=formatted,
        ast_restoration_exact=True,
        checks=checks,
        hardware_executed=False,
    )
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["output_sha256"])


if __name__ == "__main__":
    main()
