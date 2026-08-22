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

Lane FQ (FO-1/FO-2) extended the scan from test_variant_parameters.py to ALL
python_tests files and ALL field-definition forms, and added the commit-time
twin of the runtime variant-hash tripwire:

  3. VALUE-COMPLETE REPRS.  A parameter class that subclasses the @dataclass
     base WITHOUT @dataclass and stores values in a hand-written __init__
     inherits the base's EMPTY repr — the variant hash (which keys str() of
     the templates list) cannot see its values, and .build_complete reuses
     the WRONG impl's ELF within one ARTEFACTS_DIR (ReciprocalImpl /
     TypecastImpl, lane FO finding).  Every param class must be @dataclass
     with annotated fields and no hand-written __init__.  The runtime
     tripwire (helpers/param_repr_gate.py, called by generate_variant_hash)
     is proven here to FAIL LOUDLY on an injected empty-repr param.

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
REPR_GATE_PY = PYTESTS / "helpers" / "param_repr_gate.py"
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


def iter_test_sources():
    """Every python_tests .py file (tests, perf modules, helpers), venv excluded."""
    for path in sorted(PYTESTS.rglob("*.py")):
        if ".venv" in path.parts:
            continue
        yield path


def param_class_decls(tree):
    """All param ClassDef declarations in a tree, with every field-def form.

    Yields (class_name, node, ann_fields, init_self_attrs, has_dataclass,
    has_own_init):
      ann_fields      annotated class-level fields (the dataclass form)
      init_self_attrs ``self.X = ...`` targets in a hand-written __init__ /
                      __post_init__ (the form the dataclass repr cannot see)
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
        bases |= {b.attr for b in node.bases if isinstance(b, ast.Attribute)}
        if not bases & {"TemplateParameter", "RuntimeParameter"}:
            continue
        has_dataclass = any(
            (isinstance(d, ast.Name) and d.id == "dataclass")
            or (
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id == "dataclass"
            )
            for d in node.decorator_list
        )
        ann_fields = [
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and not stmt.target.id.startswith("_")
        ]
        init_self_attrs = []
        has_own_init = False
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name in (
                "__init__",
                "__post_init__",
            ):
                if stmt.name == "__init__":
                    has_own_init = True
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Assign):
                        for target in sub.targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                                and not target.attr.startswith("_")
                            ):
                                init_self_attrs.append(target.attr)
        yield node.name, node, ann_fields, init_self_attrs, has_dataclass, has_own_init


def scan_all_param_classes(trees):
    """Scan (rel_path -> ast tree) pairs; return (field_owners, offenders).

    field_owners: field -> [(class, rel_path)] over BOTH field-def forms
                  (annotated fields and __init__ self-assignments), so a
                  test-local class whose only 'fields' live in __init__ still
                  participates in the uniqueness check.
    offenders:    [(class, rel_path, reason)] for the empty-repr pattern —
                  a param class that is not @dataclass, hand-writes __init__,
                  or stores repr-invisible state without any annotated field.
    """
    field_owners = defaultdict(list)
    offenders = []
    for rel, tree in trees:
        for name, _node, ann, init_attrs, has_dc, has_init in param_class_decls(tree):
            for f in dict.fromkeys(ann + init_attrs):
                field_owners[f].append((name, rel))
            if not has_dc:
                offenders.append(
                    (
                        name,
                        rel,
                        "subclasses the @dataclass base WITHOUT @dataclass "
                        "(inherits the EMPTY repr -> value-blind variant hash)",
                    )
                )
            elif has_init:
                offenders.append(
                    (
                        name,
                        rel,
                        "hand-writes __init__ (values invisible to the "
                        "dataclass repr the variant hash keys)",
                    )
                )
            elif init_attrs and not ann:
                offenders.append(
                    (
                        name,
                        rel,
                        f"stores state {sorted(set(init_attrs))} with no "
                        "annotated field (repr-invisible)",
                    )
                )
    return field_owners, offenders


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

    # --- invariant 1b (lane FQ): the same, across ALL python_tests files ----
    trees = []
    for path in iter_test_sources():
        try:
            trees.append(
                (path.relative_to(PYTESTS).as_posix(), ast.parse(path.read_text()))
            )
        except SyntaxError as exc:
            check(False, f"{path} parses ({exc})")
    all_fields, offenders = scan_all_param_classes(trees)
    check(
        len(all_fields) > 40,
        f"scanned a real global field catalog ({len(all_fields)} fields, "
        f"{len(trees)} files)",
    )
    # Identical re-declarations of one class name (a param copied across test
    # files) are one owner; different class names sharing a field are the
    # duplicate-column hazard (mirrors test_perf_header_gate.py).
    global_dupes = {
        f: sorted(set(owners))
        for f, owners in all_fields.items()
        if len({cls for cls, _ in owners}) > 1
    }
    check(
        not global_dupes,
        "no two parameter classes ANYWHERE (test-local included, both "
        f"field-def forms) share a field name: {global_dupes or 'clean'}",
    )
    global_reserved = {
        f: sorted(set(owners)) for f, owners in all_fields.items() if f in reserved
    }
    check(
        not global_reserved,
        "no test-local parameter field shadows a pipeline-owned header: "
        f"{global_reserved or 'clean'}",
    )

    # --- invariant 3 (lane FQ, FO-2): value-complete reprs ------------------
    check(
        not offenders,
        "every parameter class is a @dataclass with annotated fields and no "
        "hand-written __init__ (empty-repr params hash value-blind and reuse "
        f"the wrong impl's ELF): {offenders or 'clean'}",
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

    # --- injection negatives: the guards must FAIL LOUDLY when seeded -------
    # (a) AST scan flags the exact ReciprocalImpl pattern (non-@dataclass
    #     subclass, hand-written __init__, self.value state).
    injected = (
        "class EvilImpl(TemplateParameter):\n"
        "    def __init__(self, value: int):\n"
        "        self.value = value\n"
        "    def convert_to_cpp(self):\n"
        "        return ''\n"
    )
    inj_fields, inj_off = scan_all_param_classes(
        [("injected_evil.py", ast.parse(injected))]
    )
    check(
        any(cls == "EvilImpl" for cls, _, _ in inj_off),
        "AST scan FLAGS an injected empty-repr param class (EvilImpl pattern)",
    )
    check(
        ("EvilImpl", "injected_evil.py") in inj_fields.get("value", []),
        "AST scan still counts the injected __init__-only field for the "
        "uniqueness check (all field-def forms)",
    )
    # (b) dup field across two DIFFERENT classes is flagged by the same scan.
    dup_src = (
        "class A(TemplateParameter):\n    value: int\n"
        "class B(RuntimeParameter):\n    value: int\n"
    )
    dup_fields, _ = scan_all_param_classes([("injected_dup.py", ast.parse(dup_src))])
    check(
        len({cls for cls, _ in dup_fields.get("value", [])}) == 2,
        "AST scan sees an injected cross-class duplicate field",
    )

    # (c) The RUNTIME tripwire (generate_variant_hash calls it) raises loudly
    #     on an empty-repr instance and stays quiet on healthy params.
    import importlib.util
    from dataclasses import dataclass

    spec = importlib.util.spec_from_file_location("param_repr_gate", REPR_GATE_PY)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    @dataclass
    class _Base:
        pass

    class _EvilImpl(_Base):  # the ReciprocalImpl defect, reconstructed
        def __init__(self, value):
            self.value = value

    @dataclass
    class _GoodImpl(_Base):
        good_impl: int = 1

    @dataclass
    class _Parameterless(_Base):  # EN_DEST_REUSE shape: no state to hide
        pass

    raised = None
    try:
        gate.assert_value_complete_reprs([_GoodImpl(2), _EvilImpl(3)], "selftest")
    except gate.ParamReprError as exc:
        raised = str(exc)
    check(
        raised is not None and "_EvilImpl" in (raised or ""),
        "runtime tripwire RAISES ParamReprError naming the injected "
        "empty-repr param (loud, not a warning)",
    )
    ok = True
    try:
        gate.assert_value_complete_reprs([_GoodImpl(2), _Parameterless()], "selftest")
    except gate.ParamReprError:
        ok = False
    check(
        ok,
        "runtime tripwire passes healthy dataclass params AND legitimately "
        "parameterless ones (EN_DEST_REUSE shape)",
    )
    check(
        REPR_GATE_PY.read_text().find("assert_value_complete_reprs") != -1
        and "assert_value_complete_reprs(self.templates"
        in (PYTESTS / "helpers" / "test_config.py").read_text(),
        "generate_variant_hash actually wires the tripwire "
        "(assert_value_complete_reprs on self.templates)",
    )

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURES")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
