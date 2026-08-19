# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Merge gate: per-perf-test CSV schema versioning + metric-vocabulary drift.

The per-test gate re-derives, for every WH/BH perf test, the columns its CSV
would carry (statically, from that test's ``PerfConfig`` parameters) and compares
them to the recorded schema in ``helpers/perf/test_schemas.py``. A change to one
test's columns fails with a per-test diff, so you can see exactly which test
changed and how — and, because each test carries a ``version``, two reports of
the same test are comparable by that number.

Quasar perf tests derive the same way from their sibling test_*_quasar.py source
(see perf_schema_derive), so ``PERF_TEST_SCHEMAS_QSR`` is gated exactly like WH/BH.

Two smaller global gates keep the metric-column vocabulary (run-type names and
efficiency-metric names) from drifting. Derivation helpers live in the device-free
``perf_schema_derive`` module, so the gate needs no hardware.
"""

import ast

from perf_schema_derive import (
    PARAM_BASES,
    ROOT,
    derive_perf_test_schemas,
    enum_member_names,
    iter_source_files,
    load_pure_module,
)

# ── Per-test schema catalog (WH/BH) ───────────────────────────────────


def _assert_schemas_match(catalog, live, arch):
    problems = []
    for name, cols in sorted(live.items()):
        entry = catalog.get(name)
        if entry is None:
            problems.append(
                f"  '{name}': perf test found in source but missing from the "
                f"catalog. Add an entry with its columns and version=1."
            )
            continue
        added = sorted(set(cols) - set(entry["columns"]))
        removed = sorted(set(entry["columns"]) - set(cols))
        if added or removed:
            problems.append(
                f"  '{name}' (schema v{entry['version']}): +{added} -{removed}"
            )
    for name in sorted(set(catalog) - set(live)):
        problems.append(
            f"  '{name}': in catalog but no longer a {arch} perf test "
            f"(renamed or deleted?). Remove its catalog entry."
        )

    msg = [
        f"{arch} per-perf-test CSV schema(s) drifted from "
        f"helpers/perf/test_schemas.py:",
        "",
        *problems,
        "",
        "If intentional: update that test's 'columns', bump its 'version', and "
        "for a renamed column add an 'aliases' entry (old -> new).",
    ]
    assert not problems, "\n".join(msg)


def test_perf_test_schemas_match():
    ps = load_pure_module("test_schemas.py")
    _assert_schemas_match(
        ps.PERF_TEST_SCHEMAS, derive_perf_test_schemas(quasar=False), "WH/BH"
    )


def test_perf_test_schemas_match_qsr():
    ps = load_pure_module("test_schemas.py")
    _assert_schemas_match(
        ps.PERF_TEST_SCHEMAS_QSR, derive_perf_test_schemas(quasar=True), "Quasar"
    )


# ── Metric-vocabulary drift (global) ──────────────────────────────────


def test_run_type_names_match_source():
    ps = load_pure_module("schema.py")
    live = enum_member_names("llk_params.py", "PerfRunType")
    assert live == set(ps.RUN_TYPE_NAMES), (
        f"PerfRunType members {sorted(live)} drifted from "
        f"helpers/perf/schema.py RUN_TYPE_NAMES {sorted(ps.RUN_TYPE_NAMES)}. Update the "
        f"catalog: a run-type name prefixes every metric/counter header."
    )


def test_metric_bases_match_source():
    """The catalog's metric bases must equal the *_pct dict keys metrics.py exports."""
    tree = ast.parse((ROOT / "helpers" / "metrics.py").read_text())
    # export_metrics keeps exactly the metric-dict keys ending in "_pct" (see
    # _exportable()). Read the dict keys via ast, not a text scan, so an
    # unrelated "_pct" string literal (log line, docstring, m.get() arg) can
    # neither trip nor evade the gate.
    live = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and key.value.endswith("_pct")
    }
    ps = load_pure_module("schema.py")
    assert live == set(ps.METRIC_BASES), (
        f"Efficiency metric names drifted. In source but not catalog: "
        f"{sorted(live - set(ps.METRIC_BASES))}; in catalog but not source: "
        f"{sorted(set(ps.METRIC_BASES) - live)}. Update helpers/perf/schema.py METRIC_BASES."
    )


# ── Uniqueness & duplicate-column gate ────────────────────────────────


def _load_perf_schema():
    return load_pure_module("schema.py")


def _reserved_headers() -> set:
    """Columns the pipeline injects itself, which no parameter field may shadow.

    Beyond the fixed sweep headers, this reserves ``marker`` (the merge key — a
    param named ``marker`` would be suffixed to ``marker_x``/``marker_y`` by the
    cross-merge, evade the duplicate gate, and break marker processing) and
    ``test_name``. ``loop_factor``/``tile_cnt`` are intentionally NOT reserved:
    they ARE parameter fields (``LOOP_FACTOR``/``TILE_COUNT``).
    """
    ps = _load_perf_schema()
    return set(ps.FIXED_SWEEP_HEADERS) | {ps.MARKER, ps.TEST_NAME_COLUMN}


def _collect_parameter_fields() -> dict:
    field_owners: dict[str, list] = {}
    for path in iter_source_files():
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


def _collect_parameter_declarations():
    """(class_name, frozenset(fields), rel_path) for every parameter ClassDef."""
    decls = []
    for path in iter_source_files():
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
            fields = frozenset(
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and not stmt.target.id.startswith("_")
            )
            decls.append((node.name, fields, path.relative_to(ROOT).as_posix()))
    return decls


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


# The uniqueness check above groups by class NAME, so a class declared twice
# collapses to one owner. Catch the dangerous case it misses: the same class
# name defined more than once with DIFFERENT fields, where the later definition
# silently shadows the earlier one. Identical re-declarations (same fields, e.g.
# a param copied across two test files) are benign and allowed.
def test_no_shadowed_parameter_class():
    by_name: dict[str, list] = {}
    for name, fields, rel in _collect_parameter_declarations():
        by_name.setdefault(name, []).append((fields, rel))
    shadowed = {
        name: decls
        for name, decls in by_name.items()
        if len({fields for fields, _ in decls}) > 1
    }
    lines = [
        "A parameter class is declared more than once with different fields; the "
        "later definition silently shadows the earlier one (only the last binding "
        "wins), and the name-only uniqueness check cannot see it.",
        "",
    ]
    for name, decls in sorted(shadowed.items()):
        lines.append(f"  class '{name}':")
        for fields, rel in sorted(decls, key=lambda d: d[1]):
            lines.append(f"      {rel}: fields={sorted(fields)}")
    lines.append("")
    lines.append("Consolidate to one definition, or give them distinct class names.")

    assert not shadowed, "\n".join(lines)


# This test would fail, because it would produce duplicate columns
def test_no_parameter_field_equals_a_fixed_header():
    reserved = _reserved_headers()
    field_owners = _collect_parameter_fields()
    clashes = {f: owners for f, owners in field_owners.items() if f in reserved}

    assert not clashes, (
        f"Parameter field name(s) collide with reserved headers "
        f"{sorted(reserved)}: {clashes}. Rename the offending field."
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
    for path in iter_source_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            names = []
            for kw in node.keywords:
                if kw.arg in PARAM_LIST_KWARGS and isinstance(
                    kw.value, (ast.List, ast.Tuple, ast.Set)
                ):
                    for elt in kw.value.elts:
                        if not isinstance(elt, ast.Call):
                            continue
                        # X(...) -> "X";  module.X(...) -> "X"
                        if isinstance(elt.func, ast.Name):
                            names.append(elt.func.id)
                        elif isinstance(elt.func, ast.Attribute):
                            names.append(elt.func.attr)
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
        ps.assert_unique_columns(
            ["dest_acc", "tile_cnt", "tile_cnt", "marker"], context="bad"
        )
    except ps.PerfSchemaError as exc:
        raised = True
        assert "tile_cnt" in str(exc)
    assert raised, "assert_unique_columns must raise PerfSchemaError on a duplicate"


def test_column_name_builders():
    ps = _load_perf_schema()
    rt = "L1_TO_L1"
    assert ps.stat_column(rt, ps.MEAN) == "mean(L1_TO_L1)"
    assert ps.stat_prefix(ps.STD) == "std("
    assert ps.metric_column(rt, "fpu_utilization_pct") == "L1_TO_L1_fpu_utilization_pct"
    assert ps.text_size_column(rt) == "TEXT_SIZE(L1_TO_L1)"
    assert ps.counter_base("FPU", "FPU_COUNTER") == "FPU.FPU_COUNTER"
    assert ps.cycles_of("FPU.FPU_COUNTER") == "FPU.FPU_COUNTER.cycles"
