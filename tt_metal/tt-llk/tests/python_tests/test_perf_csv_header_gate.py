# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Merge gate: perf-CSV column headers must be unique by construction.

A perf-report sweep column is named after a parameter dataclass field. If two
parameter classes declare the same field name, a test that passes both produces
two columns with the same header — a duplicate the CSV pipeline cannot represent
(it ships a phantom ``<name>.1`` column and silently skips the row collapse).

This test enforces the invariant that makes that impossible: every parameter
field name is unique across all ``TemplateParameter``/``RuntimeParameter``
classes, and no field name equals a fixed sweep header. It parses the source
with ``ast``, so it needs no hardware and runs in any CI lane.

A new test or parameter that would reintroduce a duplicate header fails this
test, so the pull request cannot merge. (The same-parameter-passed-twice case is
caught at run time by the gate in ``PerfReport.append``.)
"""

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARAM_BASES = {"TemplateParameter", "RuntimeParameter"}


def _iter_source_files():
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        yield path


def _load_perf_schema():
    """Load perf_schema.py directly (it has no imports), bypassing the helpers
    package __init__ which pulls device libraries. Keeps this runnable anywhere."""
    path = ROOT / "helpers" / "perf_schema.py"
    spec = importlib.util.spec_from_file_location("_perf_schema_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixed_sweep_headers() -> set:
    return set(_load_perf_schema().FIXED_SWEEP_HEADERS)


def _collect_parameter_fields() -> dict:
    field_owners: dict[str, list] = {}
    for path in _iter_source_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(
                isinstance(base, ast.Name) and base.id in PARAM_BASES
                for base in node.bases
            ):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    field = stmt.target.id
                    if field.startswith("_"):
                        continue
                    field_owners.setdefault(field, []).append(
                        (node.name, path.relative_to(ROOT).as_posix())
                    )
    return field_owners


def test_parameter_field_names_are_globally_unique():
    field_owners = _collect_parameter_fields()
    clashes = {
        field: owners
        for field, owners in field_owners.items()
        if len({cls for cls, _ in owners}) > 1
    }

    lines = [
        "Duplicate perf-CSV headers are possible: a parameter field name is "
        "declared by more than one class. Two params that share a field name "
        "produce two columns with the same header when a test passes both.",
        "",
    ]
    for field, owners in sorted(clashes.items()):
        lines.append(f"  field '{field}':")
        for cls, rel in sorted(set(owners)):
            lines.append(f"      {cls}  ({rel})")
    lines.append("")
    lines.append("Rename one field so every parameter field name is unique.")

    assert not clashes, "\n".join(lines)


def test_no_parameter_field_equals_a_fixed_header():
    fixed = _fixed_sweep_headers()
    field_owners = _collect_parameter_fields()
    clashes = {f: owners for f, owners in field_owners.items() if f in fixed}

    assert not clashes, (
        f"Parameter field name(s) collide with fixed sweep headers "
        f"{sorted(fixed)}: {clashes}. Rename the offending field."
    )


# ── Unit coverage for the perf_schema gate and name builders ──────────────


def test_find_duplicate_columns():
    ps = _load_perf_schema()
    assert ps.find_duplicate_columns(["a", "b", "a", "c", "b"]) == ["a", "b"]
    assert ps.find_duplicate_columns(["a", "b", "c"]) == []


def test_assert_unique_columns_passes_on_unique():
    ps = _load_perf_schema()
    # No exception expected.
    ps.assert_unique_columns(["formats.input_A", "mathop", "marker", "mean(L1_TO_L1)"])


def test_assert_unique_columns_raises_on_duplicate():
    ps = _load_perf_schema()
    raised = False
    try:
        ps.assert_unique_columns(["dest_acc", "tile_cnt", "tile_cnt", "marker"], "bad")
    except ps.PerfSchemaError as exc:
        raised = True
        assert "tile_cnt" in str(exc)
    assert raised, "assert_unique_columns must raise PerfSchemaError on a duplicate"


def test_column_name_builders():
    ps = _load_perf_schema()
    assert ps.stat_column("L1_TO_L1", ps.MEAN) == "mean(L1_TO_L1)"
    assert ps.stat_prefix(ps.STD) == "std("
    assert ps.metric_column("L1_TO_L1", "fpu_utilization_pct") == (
        "L1_TO_L1_fpu_utilization_pct"
    )
    assert ps.text_size_column("L1_TO_L1") == "TEXT_SIZE(L1_TO_L1)"
    assert ps.counter_base("FPU", "FPU_COUNTER") == "FPU.FPU_COUNTER"
    assert ps.cycles_of("FPU.FPU_COUNTER") == "FPU.FPU_COUNTER.cycles"
