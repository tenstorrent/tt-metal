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
    class_field_specs,
    class_post_init_copies,
    derive_perf_test_schemas,
    emitted_fields,
    enum_member_names,
    helper_returned_param_calls,
    iter_source_files,
    load_pure_module,
)

# ── Per-test schema catalog (WH/BH) ───────────────────────────────────


def _test_name_alias_problems(catalog) -> list:
    """Every catalog entry must have ``test_name_aliases`` with current -> current.

    Extra keys are previous names. Every value must equal the catalog key, so
    renaming a test in place without editing this map fails: the identity
    entry would still point at the old name.
    """
    problems = []
    claimed = {}
    for name, entry in sorted(catalog.items()):
        if "test_name_aliases" not in entry:
            problems.append(
                f"  '{name}': missing test_name_aliases. Add "
                f"{{'{name}': '{name}'}} and record any previous names."
            )
            continue
        aliases = entry["test_name_aliases"]
        if not isinstance(aliases, dict) or not aliases:
            problems.append(
                f"  '{name}': test_name_aliases must be a non-empty map "
                f"(old -> new), starting with '{name}' -> '{name}'."
            )
            continue
        if aliases.get(name) != name:
            problems.append(
                f"  '{name}': test_name_aliases must include the current "
                f"name '{name}' -> '{name}'. Update this map when the test "
                f"is renamed."
            )
        for old, new in aliases.items():
            if new != name:
                problems.append(
                    f"  '{name}': test_name_aliases maps '{old}' -> '{new}', "
                    f"but this catalog entry is '{name}'. Every new name must "
                    f"match the current test name."
                )
            if old != name and old in catalog:
                problems.append(
                    f"  '{name}': test_name_aliases old name '{old}' is still "
                    f"a catalog key. Point aliases at the surviving name only."
                )
            owner = claimed.get(old)
            if owner is not None and owner != name:
                problems.append(
                    f"  '{name}': test_name_aliases old name '{old}' already "
                    f"maps to '{owner}'."
                )
            claimed[old] = name
    return problems


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
    problems.extend(_test_name_alias_problems(catalog))

    msg = [
        f"{arch} per-perf-test CSV schema(s) drifted from "
        f"helpers/perf/test_schemas.py:",
        "",
        *problems,
        "",
        "If intentional: update that test's 'columns', bump its 'version', and "
        "for a renamed column add an 'aliases' entry (old -> new). For a renamed "
        "test, update 'test_name_aliases' so the current name maps to itself and "
        "every previous name maps to the current name.",
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


def test_test_name_aliases_align_with_catalog_key():
    catalog = {
        "perf_new": {
            "version": 1,
            "columns": ["marker"],
            "test_name_aliases": {
                "perf_new": "perf_new",
                "perf_old": "perf_new",
                "perf_older": "perf_new",
            },
        },
        "perf_untouched": {
            "version": 1,
            "columns": ["marker"],
            "test_name_aliases": {"perf_untouched": "perf_untouched"},
        },
    }
    assert _test_name_alias_problems(catalog) == []


def test_test_name_aliases_reject_misaligned_new_name():
    catalog = {
        "perf_new": {
            "version": 1,
            "columns": ["marker"],
            "test_name_aliases": {
                "perf_new": "perf_new",
                "perf_old": "perf_wrong",
            },
        },
    }
    problems = _test_name_alias_problems(catalog)
    assert len(problems) == 1
    assert "perf_old" in problems[0]
    assert "perf_wrong" in problems[0]
    assert "perf_new" in problems[0]


def test_test_name_aliases_require_current_identity():
    catalog = {
        "perf_new": {
            "version": 1,
            "columns": ["marker"],
            "test_name_aliases": {"perf_old": "perf_new"},
        },
    }
    problems = _test_name_alias_problems(catalog)
    assert any("must include the current name" in p for p in problems)


def test_test_name_aliases_reject_rename_without_updating_map():
    # Catalog key renamed in place; map still identity of the old name.
    catalog = {
        "perf_new": {
            "version": 1,
            "columns": ["marker"],
            "test_name_aliases": {"perf_old": "perf_old"},
        },
    }
    problems = _test_name_alias_problems(catalog)
    assert any("must include the current name" in p for p in problems)
    assert any("perf_old" in p and "perf_new" in p for p in problems)


def test_test_name_aliases_reject_missing_and_stale_key():
    catalog = {
        "perf_new": {"version": 1, "columns": ["marker"]},
        "perf_still_live": {
            "version": 1,
            "columns": ["marker"],
            "test_name_aliases": {
                "perf_still_live": "perf_still_live",
                "perf_other": "perf_still_live",
            },
        },
        "perf_other": {
            "version": 1,
            "columns": ["marker"],
            "test_name_aliases": {"perf_other": "perf_other"},
        },
    }
    problems = _test_name_alias_problems(catalog)
    assert any("missing test_name_aliases" in p for p in problems)
    assert any("still a catalog key" in p for p in problems)


# ── Metric-vocabulary drift (global) ──────────────────────────────────


def test_emitted_fields_see_helper_returns_and_post_init_copies():
    """The two constructs the plain None-default rule cannot see, and the one it must."""
    specs = class_field_specs()
    helpers = helper_returned_param_calls(specs)
    post_init = class_post_init_copies()

    def fields(src):
        call = ast.parse(src).body[0].value
        return set(emitted_fields(call, specs, helpers, post_init))

    # A helper that returns a param object: templates= holds the helper, not the class.
    assert {"full_rt_dim", "full_ct_dim", "block_ct_dim", "block_rt_dim"} <= fields(
        "generate_input_dim((32, 32), (32, 32))"
    )
    # __post_init__ copies: the call sets one field, the runtime emits three.
    assert {"num_blocks", "input_num_blocks", "output_num_blocks"} <= fields(
        "NUM_BLOCKS(4)"
    )
    # Still dropped: a genuinely optional None-default slot the call leaves unset.
    assert "mathop" in fields("MATH_OP(MathOperation.Elwadd)")
    assert not {"unary_extra", "pool_type"} & fields("MATH_OP(MathOperation.Elwadd)")

    # Guard paths: with neither map passed, the plain None-default rule stands and
    # a helper call resolves to nothing, so existing callers are unaffected.
    plain = ast.parse("NUM_BLOCKS(4)").body[0].value
    assert set(emitted_fields(plain, specs)) == {"num_blocks"}
    helper = ast.parse("generate_input_dim((32, 32), (32, 32))").body[0].value
    assert emitted_fields(helper, specs) == []


# Run mode, not a sweep parameter: published, deliberately never in the catalog.
NOT_A_SWEEP_COLUMN = {"speed_of_light"}


def _published_test_columns(module_filename: str) -> set:
    """Literal Column(...) names a test fills. Generated timing columns have no
    literal to read and drop out; origin="ci" provenance is skipped."""
    tree = ast.parse((ROOT / "helpers" / "perf" / module_filename).read_text())
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "Column"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and not any(
            kw.arg == "origin" and getattr(kw.value, "value", None) == "ci"
            for kw in node.keywords
        )
    }


def test_no_published_column_without_a_perf_test():
    """No dead column: a published column cannot outlive the test that filled it."""
    ps = load_pure_module("test_schemas.py")
    for module, catalog, arch in (
        ("wide_schema.py", ps.PERF_TEST_SCHEMAS, "WH/BH"),
        ("wide_schema_quasar.py", ps.PERF_TEST_SCHEMAS_QSR, "Quasar"),
    ):
        recorded = set().union(*(set(e["columns"]) for e in catalog.values()))
        dead = sorted(_published_test_columns(module) - recorded - NOT_A_SWEEP_COLUMN)
        assert not dead, (
            f"{arch}: helpers/perf/{module} declares {dead}, which no perf test "
            "produces. Drop the column, or add the test that fills it to "
            "helpers/perf/test_schemas.py."
        )


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
