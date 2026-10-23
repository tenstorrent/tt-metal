# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPRECATED_ALL_GATHER_ARGS = frozenset(
    {
        "num_links",
        "topology",
        "chunks_per_sync",
        "num_workers_per_link",
        "num_buffers_per_channel",
        "use_l1_small_for_semaphores",
    }
)
REQUIRED_ALL_GATHER_ARGS = frozenset({"cluster_axis", "memory_config"})


def _production_all_gather_calls() -> tuple[tuple[Path, ast.Call], ...]:
    calls: list[tuple[Path, ast.Call]] = []
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT)
        if relative.parts[0] == "tests":
            continue
        tree = ast.parse(path.read_text(), filename=str(relative))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ttnn"
                and node.func.attr == "all_gather"
            ):
                calls.append((relative, node))
    assert calls, "no production ttnn.all_gather calls found; coverage would be vacuous"
    return tuple(calls)


def _keyword_names(path: Path, call: ast.Call) -> frozenset[str]:
    unpacked = [keyword for keyword in call.keywords if keyword.arg is None]
    assert not unpacked, f"{path}:{call.lineno} uses **kwargs, so all_gather arguments cannot be checked statically"
    return frozenset(keyword.arg for keyword in call.keywords if keyword.arg is not None)


def test_production_all_gather_calls_omit_every_deprecated_argument() -> None:
    violations: list[str] = []
    for path, call in _production_all_gather_calls():
        deprecated = _keyword_names(path, call) & DEPRECATED_ALL_GATHER_ARGS
        if deprecated:
            violations.append(f"{path}:{call.lineno}: {', '.join(sorted(deprecated))}")
    assert not violations, "deprecated ttnn.all_gather arguments remain:\n" + "\n".join(violations)


def test_production_all_gather_calls_retain_axis_and_memory_config() -> None:
    violations: list[str] = []
    for path, call in _production_all_gather_calls():
        missing = REQUIRED_ALL_GATHER_ARGS - _keyword_names(path, call)
        if missing:
            violations.append(f"{path}:{call.lineno}: {', '.join(sorted(missing))}")
    assert not violations, "required ttnn.all_gather arguments are missing:\n" + "\n".join(violations)
