#!/usr/bin/env python3
"""Self-test for the perf-CSV column-schema contract (lane FM, FM-F1).

The perf pipeline derives CSV column headers from the DATACLASS FIELD NAMES
of every Template/RuntimeParameter a test passes (PerfConfig
_build_sweep_frame -> _dataclass_name_and_values), then refuses to ship a
report whose columns are not a unique, homogeneous schema (PerfSchemaError,
helpers/perf/schema.py).  That contract has two commit-time invariants no
runtime check sees until a device session already burned:

  1. GLOBAL FIELD-NAME UNIQUENESS.  Two parameter classes sharing a field
     name produce duplicate columns the moment one test passes both.  FM-F1:
     COVERAGE_OP / COVERAGE_SUBOP / FRESH_CPP_IMPL all named their field
     ``value`` — every perf_sfpu_coverage node raised PerfSchemaError
     ("duplicate column header(s) ['value']") even in a solo session,
     blocking all 10 coverage ops in two weekly runs.  Field names must
     also stay clear of the fixed sweep/key headers schema.py owns.

  2. ONE SCHEMA PER MODULE FILE.  A module whose parametrizations emit
     different column SETS (a param passed for some sweep values and not
     others) stacks >=2 schemas into its combined CSV — the multi-schema
     PerfSchemaError (binarypow precedent, one-schema-per-file).  For
     perf_sfpu_coverage the guard is structural: the templates list must be
     ONE literal list with no conditional appends, so every (op, impl) node
     emits identical columns by construction.

Pure AST/static checks — no torch, no pandas, no device.  Run standalone or
from the sweep wrappers; exits nonzero on any failure.
"""

import ast
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
PYTESTS = HERE.parent / "python_tests"
PARAMS_PY = PYTESTS / "helpers" / "test_variant_parameters.py"
SCHEMA_PY = PYTESTS / "helpers" / "perf" / "schema.py"
COVERAGE_PERF_PY = PYTESTS / "perf_sfpu_coverage.py"

FAILS = []


def check(cond, msg):
    print(("PASS: " if cond else "FAIL: ") + msg)
    if not cond:
        FAILS.append(msg)


def param_class_fields(tree):
    """{class_name: [annotated field names]} for every class deriving
    TemplateParameter/RuntimeParameter (direct base spelling)."""
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
        if not bases & {"TemplateParameter", "RuntimeParameter"}:
            continue
        fields = [
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and not stmt.target.id.startswith("_")
        ]
        out[node.name] = fields
    return out


def load_schema_constants():
    """Fixed/key header names from helpers/perf/schema.py by direct import
    (stdlib-only module — safe without the tests venv)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("perf_schema", SCHEMA_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # PIPELINE-owned columns only.  loop_factor/tile_cnt are param-owned:
    # schema.py names them precisely because LOOP_FACTOR/TILE_COUNT emit
    # them — single ownership is what the uniqueness check above proves.
    reserved = set(mod.FIXED_SWEEP_HEADERS)
    reserved.add(mod.MARKER)
    reserved.add(mod.TEST_NAME_COLUMN)
    reserved.add("speed_of_light")  # _build_sweep_frame's extra fixed column
    return reserved


def main():
    params_tree = ast.parse(PARAMS_PY.read_text())
    classes = param_class_fields(params_tree)
    check(len(classes) > 20, f"parsed a real class catalog ({len(classes)} classes)")

    # --- invariant 1: global field-name uniqueness -------------------------
    by_field = defaultdict(list)
    for cls, fields in classes.items():
        for f in fields:
            by_field[f].append(cls)
    dupes = {f: cs for f, cs in by_field.items() if len(cs) > 1}
    check(
        not dupes,
        "no two parameter classes share a dataclass field name "
        f"(duplicate perf-CSV columns otherwise): {dupes or 'clean'}",
    )

    reserved = load_schema_constants()
    clashes = {f: cs for f, cs in by_field.items() if f in reserved}
    check(
        not clashes,
        "no parameter field shadows a fixed sweep/key header "
        f"from helpers/perf/schema.py: {clashes or 'clean'}",
    )

    # FM-F1 regression pins: the trio that used to collide on 'value'.
    for cls, want in (
        ("FRESH_CPP_IMPL", "fresh_cpp_impl"),
        ("COVERAGE_OP", "coverage_op"),
        ("COVERAGE_SUBOP", "coverage_subop"),
    ):
        check(
            classes.get(cls) == [want],
            f"{cls} exposes exactly the unique field ['{want}'] (got {classes.get(cls)})",
        )

    # --- invariant 2: perf_sfpu_coverage emits ONE schema ------------------
    cov_tree = ast.parse(COVERAGE_PERF_PY.read_text())

    # (a) No conditional template growth: 'templates' is assigned exactly one
    #     literal list and never .append()ed / +=ed afterwards.
    assigns, mutations = [], []
    for node in ast.walk(cov_tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "templates" for t in node.targets
        ):
            assigns.append(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("append", "extend", "insert")
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "templates"
        ):
            mutations.append(node)
        if isinstance(node, ast.AugAssign) and (
            isinstance(node.target, ast.Name) and node.target.id == "templates"
        ):
            mutations.append(node)
    check(
        len(assigns) == 1 and isinstance(assigns[0].value, ast.List),
        "perf_sfpu_coverage builds 'templates' as exactly one literal list",
    )
    check(
        not mutations,
        "perf_sfpu_coverage never grows 'templates' conditionally "
        "(no append/extend/insert/+=) — every op emits identical columns",
    )

    # (b) The literal list's parameter classes exist in the catalog and their
    #     combined field names are duplicate-free (the exact FM-F1 failure).
    if assigns and isinstance(assigns[0].value, ast.List):
        used = []
        for elt in assigns[0].value.elts:
            if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name):
                used.append(elt.func.id)
        check(
            "SFPU_UNARY_SCALAR" in used,
            "SFPU_UNARY_SCALAR is passed unconditionally (schema homogeneity "
            "across scalar and non-scalar coverage ops)",
        )
        unknown = [u for u in used if u not in classes]
        check(
            not unknown,
            f"every passed parameter class is in the catalog: {unknown or 'clean'}",
        )
        union = [f for u in used if u in classes for f in classes[u]]
        cols_dupes = sorted({f for f in union if union.count(f) > 1})
        check(
            not cols_dupes,
            "the coverage module's combined parameter columns are "
            f"duplicate-free: {cols_dupes or 'clean'}",
        )

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURES")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
