# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every corpus source checked against the compiled object; lit reference pairs identical.

Slow (minutes): run with `pytest -m slow`.
"""

import pathlib

import harness
import pytest
from conftest import TOOL, needs_toolchain
from hal_explorer import pipeline
from hal_explorer.toolchain import build

LIT = (
    build.LLK_ROOT
    / "tools"
    / "tests"
    / "unit"
    / "lit"
    / "hal"
    / "blackhole"
    / "codegen"
)
TOOL_DIR = pathlib.Path(TOOL)
CORPUS = (
    sorted((TOOL_DIR / "examples").glob("*.cpp"))
    + sorted((TOOL_DIR / "tests" / "cases").glob("*.cpp"))
    + sorted(LIT.rglob("*.cpp"))
)
VECTORS = 12

pytestmark = [needs_toolchain, pytest.mark.slow]


def label(path):
    return str(path.relative_to(build.LLK_ROOT))


@pytest.mark.parametrize("path", CORPUS, ids=label)
def test_matches_the_object_file(tables, path):
    source = path.read_text(encoding="utf-8")
    # The reviewer cases were written against a prelude that declared the HAL aliases.
    if path.parent.name == "cases":
        source = harness.with_aliases(source)
    report = harness.check(source, tables, VECTORS)
    assert not report.mismatches, "\n".join(report.mismatches[:5])
    assert not report.internal, "\n".join(report.internal)


@pytest.mark.parametrize("path", sorted(LIT.rglob("*.cpp")), ids=label)
def test_lit_reference_pairs_are_identical(tables, path):
    result = pipeline.run("code", path.read_text(encoding="utf-8"), tables)
    assert result.ok, result.diagnostics
    failing = [
        f"{c.name}: {c.verdict} {c.reasons} {c.counterexample}"
        for c in result.comparisons
        if c.verdict != "identical"
    ]
    assert not failing, "\n".join(failing)
