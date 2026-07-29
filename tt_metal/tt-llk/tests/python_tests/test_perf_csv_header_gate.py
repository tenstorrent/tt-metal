# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Merge gate: per-perf-test CSV schema versioning + metric-vocabulary drift.

The per-test gate re-derives, for every perf test, the columns its CSV would
carry (statically, from that test's ``PerfConfig`` parameters) and compares them
to the recorded schema in ``helpers/perf_test_schemas.py``. A change to one
test's columns fails with a per-test diff, so you can see exactly which test
changed and how — and, because each test carries a ``version``, two reports of
the same test are comparable by that number.

Two smaller global gates keep the metric-column vocabulary (run-type names and
efficiency-metric names) from drifting. Everything parses the source with
``ast``, so the gate needs no hardware.
"""

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARAM_BASES = {"TemplateParameter", "RuntimeParameter"}
LIST_KWARGS = {"templates", "runtimes"}
MARKER_COLUMN = "marker"


def _iter_source_files():
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        yield path


def _load(module_filename: str):
    """Load a helpers module that has no device imports, directly by path.

    perf_schema.py / perf_test_schemas.py are import-free, so this bypasses the
    helpers package __init__ (which pulls device libraries) and keeps the gate
    runnable without hardware."""
    path = ROOT / "helpers" / module_filename
    spec = importlib.util.spec_from_file_location(f"_gate_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Per-test schema derivation ────────────────────────────────────────


def _class_field_specs() -> dict:
    """Parameter class -> ordered [(field_name, default_is_none)].

    Order matters so we can map positional constructor args to fields.
    ``default_is_none`` marks a field the report DROPS when left unset (the
    runtime emits a column only when its value is not None)."""
    specs: dict[str, list] = {}
    for path in _iter_source_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(
                isinstance(b, ast.Name) and b.id in PARAM_BASES for b in node.bases
            ):
                continue
            fields = []
            for s in node.body:
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                    name = s.target.id
                    if name.startswith("_"):
                        continue
                    default_is_none = (
                        isinstance(s.value, ast.Constant) and s.value.value is None
                    )
                    fields.append((name, default_is_none))
            specs.setdefault(node.name, fields)
    return specs


def _emitted_fields(elt: ast.Call, specs: dict) -> list:
    """Fields a param instantiation actually emits as CSV columns.

    Mirrors the runtime rule: a None-default field left unset is dropped, so we
    emit a field unless (its default is None AND the call does not set it). This
    stops us over-listing optional slots like MATH_OP.pool_type/unary_extra."""
    name = None
    if isinstance(elt.func, ast.Name):
        name = elt.func.id
    elif isinstance(elt.func, ast.Attribute):
        name = elt.func.attr
    fields = specs.get(name)
    if fields is None:
        return []
    provided = {fields[i][0] for i in range(min(len(elt.args), len(fields)))}
    provided |= {kw.arg for kw in elt.keywords if kw.arg}
    return [f for f, dflt_none in fields if not (dflt_none and f not in provided)]


def derive_perf_test_schemas() -> dict:
    """{perf_test_name -> sorted column list} derived statically from PerfConfig.

    Columns = the fixed sweep headers (formats/flags) + the sweep-parameter
    fields the test actually sets + the marker column. Metric/counter columns are
    formula-driven from run types and are covered by the drift gates below.

    NOTE: static approximation. It cannot see params built with dynamic lists
    (comprehensions/helpers), so such columns are under-listed; only the runtime
    report (or a hardware-free report test, #51244) is exact.
    """
    ps = _load("perf_schema.py")
    fixed = list(ps.FORMAT_HEADERS) + list(ps.FLAG_HEADERS)
    specs = _class_field_specs()
    schemas: dict[str, list] = {}
    for path in ROOT.rglob("perf_*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        configs = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "PerfConfig"
        ]
        if not configs:
            continue
        fields = set()
        for cfg in configs:
            for kw in cfg.keywords:
                if kw.arg in LIST_KWARGS and isinstance(
                    kw.value, (ast.List, ast.Tuple, ast.Set)
                ):
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Call):
                            fields.update(_emitted_fields(elt, specs))
        schemas[path.stem] = sorted(set(fixed) | fields | {MARKER_COLUMN})
    return schemas


def test_perf_test_schemas_match():
    catalog = _load("perf_test_schemas.py").PERF_TEST_SCHEMAS
    live = derive_perf_test_schemas()

    problems = []
    for name, cols in sorted(live.items()):
        entry = catalog.get(name)
        if entry is None:
            problems.append(f"  '{name}': new perf test, no schema entry. Add one.")
            continue
        added = sorted(set(cols) - set(entry["columns"]))
        removed = sorted(set(entry["columns"]) - set(cols))
        if added or removed:
            problems.append(
                f"  '{name}' (schema v{entry['version']}): +{added} -{removed}"
            )
    for name in sorted(set(catalog) - set(live)):
        problems.append(f"  '{name}': in catalog but no longer a perf test.")

    msg = [
        "Per-perf-test CSV schema(s) drifted from helpers/perf_test_schemas.py:",
        "",
        *problems,
        "",
        "If intentional: update that test's 'columns', bump its 'version', and "
        "for a renamed column add an 'aliases' entry (old -> new).",
    ]
    assert not problems, "\n".join(msg)


# ── Metric-vocabulary drift (global) ──────────────────────────────────


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


def test_run_type_names_match_source():
    ps = _load("perf_schema.py")
    live = _enum_member_names("llk_params.py", "PerfRunType")
    assert live == set(ps.RUN_TYPE_NAMES), (
        f"PerfRunType members {sorted(live)} drifted from "
        f"perf_schema.RUN_TYPE_NAMES {sorted(ps.RUN_TYPE_NAMES)}. Update the "
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
    ps = _load("perf_schema.py")
    assert live == set(ps.METRIC_BASES), (
        f"Efficiency metric names drifted. In source but not catalog: "
        f"{sorted(live - set(ps.METRIC_BASES))}; in catalog but not source: "
        f"{sorted(set(ps.METRIC_BASES) - live)}. Update perf_schema.METRIC_BASES."
    )
