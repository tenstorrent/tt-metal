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
import re
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


# Checks if two or more classes share a field name
# For every parameter field it checks if it has multiple "owners" (classes)
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


# Checks if there is a parameter named the same as any of the fixed parameters
# These fixed parameters are FIXED_SWEEP_HEADERS in perf_schema.py
# For example if you added a new @dataclass
# class SOME_NEW_RANDOM_PARAM_WE_MADE_UP(TemplateParameter):
# dest_add: bool = False


# This test would fail, because it would produce duplicate columns
def test_no_parameter_field_equals_a_fixed_header():
    fixed = _fixed_sweep_headers()
    field_owners = _collect_parameter_fields()
    clashes = {f: owners for f, owners in field_owners.items() if f in fixed}

    assert not clashes, (
        f"Parameter field name(s) collide with fixed sweep headers "
        f"{sorted(fixed)}: {clashes}. Rename the offending field."
    )


# Golden-catalog drift gate
#
# The hand-maintained catalog in perf_schema.py is the single source of
# header names. These tests read the LIVE source with ast and fail if it drifts
# from the catalog, so a header rename cannot merge without a deliberate catalog edit.


def _enum_member_names(module_filename: str, enum_name: str) -> set:
    """Names assigned in an Enum class body, read statically (no import)."""
    path = ROOT / "helpers" / module_filename
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == enum_name:
            return {
                target.id
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                for target in stmt.targets
                if isinstance(target, ast.Name)
            }
    return set()


# Checks if there is added perf sweep parameters that dont align with the ones
# defined in the catalog.
def test_sweep_params_match_golden_catalog():
    ps = _load_perf_schema()
    live = set(_collect_parameter_fields())
    golden = set(ps.GOLDEN_SWEEP_PARAMS)

    added = sorted(live - golden)
    removed = sorted(golden - live)

    lines = [
        "Perf-CSV sweep headers drifted from the golden catalog "
        "(perf_schema.GOLDEN_SWEEP_PARAMS).",
        "",
        f"  New/renamed field(s) in source, NOT in catalog: {added}",
        f"  Field(s) in catalog, gone from source:          {removed}",
        "",
        "A header is a join key between the branch report and the master "
        "baseline, so a silent rename breaks the compare. If this change is "
        "intentional, update GOLDEN_SWEEP_PARAMS in the same PR. If it is a "
        "rename whose OLD name is now fully retired, also add a HEADER_ALIASES "
        "entry so old baselines still line up.",
    ]
    assert not added and not removed, "\n".join(lines)


# Same as above, just for run type params
def test_run_type_names_match_source():
    ps = _load_perf_schema()
    live = _enum_member_names("llk_params.py", "PerfRunType")
    assert live == set(ps.RUN_TYPE_NAMES), (
        f"PerfRunType members {sorted(live)} drifted from "
        f"perf_schema.RUN_TYPE_NAMES {sorted(ps.RUN_TYPE_NAMES)}. Update the "
        f"catalog: a run-type name prefixes every metric/counter header."
    )


# Same as above, just for metrics
def test_metric_bases_match_source():
    """The catalog's metric bases must equal the *_pct keys metrics.py exports."""
    source = (ROOT / "helpers" / "metrics.py").read_text()
    # metrics.py exports exactly the keys ending in "_pct" (see _exportable()).
    live = set(re.findall(r'"([a-z0-9_]+_pct)"', source))
    ps = _load_perf_schema()
    assert live == set(ps.METRIC_BASES), (
        f"Efficiency metric names drifted. In source but not catalog: "
        f"{sorted(live - set(ps.METRIC_BASES))}; in catalog but not source: "
        f"{sorted(set(ps.METRIC_BASES) - live)}. Update perf_schema.METRIC_BASES."
    )


PARAM_LIST_KWARGS = {"templates", "runtimes"}


# This tests how are classes used in tests, because even though we may not have a class with duplicate field
# We can still use that class in a wrong way and cause duplicate columns, for example:
# class RNDM_CLASS(RuntimeParameter):
# input_tile_cnt: int = 0 - this is cool, its unique
# BUT, if we do
# runtimes=(RNDM_CLASS(4), RNDM_CLASS(8)) we still produce duplicate columns of tile_cnt?
def _configs_with_duplicate_param_types():
    problems = []  # (path, lineno, [duplicated type names])
    for path in _iter_source_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            names = []
            for kw in node.keywords:
                if kw.arg in PARAM_LIST_KWARGS and isinstance(kw.value, ast.List):
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name):
                            names.append(elt.func.id)
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                problems.append((path.relative_to(ROOT).as_posix(), node.lineno, dupes))
    return problems


# Explained above
def test_no_param_type_used_twice_in_one_config():
    problems = _configs_with_duplicate_param_types()
    lines = [
        "A parameter type is used more than once in a single test config. Each "
        "use emits the same CSV header, so the report gets a duplicate column.",
        "",
    ]
    for path, lineno, dupes in problems:
        lines.append(f"  {path}:{lineno}: used twice -> {dupes}")
    lines += [
        "",
        "Use each parameter type at most once per config. If you need two of the "
        "same quantity, model them as distinct parameter types with unique field "
        "names (e.g. INPUT_TILE_CNT vs OUTPUT_TILE_CNT).",
    ]
    assert not problems, "\n".join(lines)


# Mapping old headers to new ones is valid only if the old header is not used anywhere else anymore
# Meaning if old is still owned by another param class, renaming it globally would corrupt that column
def test_header_aliases_are_sound():
    ps = _load_perf_schema()
    live_headers = (
        set(_collect_parameter_fields())
        | set(ps.FIXED_SWEEP_HEADERS)
        | set(ps.KEY_COLUMNS)
    )
    for old, new in ps.HEADER_ALIASES.items():
        assert new in live_headers, (
            f"alias target {new!r} is not a current header; an alias must map "
            f"onto a live column name."
        )
        assert old not in live_headers, (
            f"alias source {old!r} is STILL a live header (owned by another "
            f"parameter). A global rename would corrupt that column — bridge "
            f"this rename per-test instead of adding it to HEADER_ALIASES."
        )


# Unit coverage for the perf_schema gate and name builders


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
