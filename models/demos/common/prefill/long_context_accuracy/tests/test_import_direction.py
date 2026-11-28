# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The shared long-context harness imports no model code: every ``models.`` import in this folder stays inside it.
Model specifics (goldens' readers, device blocks, sizes) come in through the case. CPU only."""

from __future__ import annotations

import ast
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
PACKAGE = "models.demos.common.prefill.long_context_accuracy"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_the_harness_imports_no_model_code() -> None:
    files = sorted(HARNESS.rglob("*.py"))
    assert len(files) > 5, f"found only {files} under {HARNESS}"
    outside = [
        f"{path.relative_to(HARNESS)}: {name}"
        for path in files
        for name in _imports(path)
        if (name == "models" or name.startswith("models.")) and not (name == PACKAGE or name.startswith(PACKAGE + "."))
    ]
    assert not outside, outside
