# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Static derivation of per-perf-test CSV schemas (no hardware).

Helpers the merge gate uses to re-derive, from source, the columns each perf
test's CSV would carry, so a change to a test's columns is caught as drift from
the recorded catalog in helpers/perf/test_schemas.py. Everything parses source
with ``ast`` and touches no device libraries.

Two arch families, derived separately:

  WH/BH  perf_*.py at the python_tests root, derived from themselves.
         ->  PERF_TEST_SCHEMAS

  Quasar quasar/perf_*_quasar.py wrappers, derived from the sibling
         test_*_quasar.py (perf_ -> test_) that calls PerfConfig and holds the
         templates/runtimes lists.  ->  PERF_TEST_SCHEMAS_QSR

Both families pass their templates=/runtimes= lists as literals (inline in the
PerfConfig call, via a named variable, or as a dict entry), so both derive
statically the same way.

This module is import-free of device libraries on purpose, so the gate imports it
normally (unlike helpers/*, whose package __init__ pulls ttexalens).
"""

import ast
import importlib.util
from pathlib import Path

# python_tests root (this file lives directly under it).
ROOT = Path(__file__).resolve().parent

PARAM_BASES = {"TemplateParameter", "RuntimeParameter"}
LIST_KWARGS = {"templates", "runtimes"}
MARKER_COLUMN = "marker"
QUASAR_DIR = "quasar"
PERF_CONFIG_BUILDERS = {"PerfConfig", "create_test_or_perf_config"}


def iter_source_files(include_quasar: bool = True):
    """Every test/​helper .py under the root. Scans ALL files (not just perf_*.py)
    because the parameter CLASSES are defined in helpers/, not in the perf tests."""
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        if not include_quasar and QUASAR_DIR in path.parts:
            continue
        yield path


def load_pure_module(module_filename: str):
    """Load a device-free helpers module directly by path.

    schema.py / test_schemas.py hold no device imports, so loading them
    by path bypasses the helpers package __init__ (which pulls ttexalens) and
    keeps the gate runnable without hardware.
    """
    path = ROOT / "helpers" / "perf" / module_filename
    spec = importlib.util.spec_from_file_location(f"_gate_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_field_specs() -> dict:
    """Map every parameter class name -> ordered [(field_name, default_is_none)].

    Read statically from the class body: each annotated field becomes an entry,
    in declaration order (so positional constructor args can be mapped to fields).
    ``default_is_none`` marks a field whose default is None; the runtime emits a
    column for such a field only when the call sets it, so the gate drops it
    otherwise (see emitted_fields). Private (``_``-prefixed) fields are ignored.
    """
    specs: dict[str, list] = {}
    for path in iter_source_files():
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


def emitted_fields(elt: ast.Call, specs: dict) -> list:
    """Fields a param instantiation actually emits as CSV columns.

    Mirrors the runtime rule: a None-default field left unset is dropped, so we
    emit a field unless (its default is None AND the call does not set it). This
    stops us over-listing optional slots like MATH_OP.pool_type/unary_extra.
    """
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


def _has_perfconfig(tree) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in PERF_CONFIG_BUILDERS
        for n in ast.walk(tree)
    )


def _is_fuser_perf_test(tree) -> bool:
    return any(
        isinstance(n, ast.Attribute) and n.attr == "run_perf_test"
        for n in ast.walk(tree)
    )


def _param_fields_in_tree(tree, specs) -> set:
    """Emitted param fields from every templates=/runtimes= list in a file.

    Covers the three shapes the tests use to pass those lists:
      * inline in the call:      PerfConfig(templates=[...], runtimes=[...])
      * a named variable:        templates = [...]; PerfConfig(**cfg)
      * a dict entry:            cfg = {"templates": [...], "runtimes": [...]}
    """
    fields = set()

    def add_list(list_node):
        for elt in list_node.elts:
            if isinstance(elt, ast.Call):
                fields.update(emitted_fields(elt, specs))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in LIST_KWARGS
                    and isinstance(node.value, (ast.List, ast.Tuple, ast.Set))
                ):
                    add_list(node.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in PERF_CONFIG_BUILDERS
        ):
            for kw in node.keywords:
                if kw.arg in LIST_KWARGS and isinstance(
                    kw.value, (ast.List, ast.Tuple, ast.Set)
                ):
                    add_list(kw.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value in LIST_KWARGS
                    and isinstance(value, (ast.List, ast.Tuple, ast.Set))
                ):
                    add_list(value)
    return fields


def _perf_test_sources(quasar: bool):
    """Yield (catalog_key, source_path) for each perf test.

    WH/BH: perf_*.py outside quasar/, derived from itself.
    Quasar: quasar/perf_*_quasar.py wrappers, derived from the sibling
    test_*_quasar.py (perf_ -> test_) that actually calls PerfConfig and holds the
    templates/runtimes lists. Keyed by the wrapper name a developer runs.
    """
    if quasar:
        for wrapper in sorted(ROOT.glob(f"{QUASAR_DIR}/perf_*_quasar.py")):
            sibling = wrapper.parent / wrapper.name.replace("perf_", "test_", 1)
            if _is_fuser_perf_test(ast.parse(wrapper.read_text())):
                yield wrapper.stem, wrapper
            elif sibling.exists():
                yield wrapper.stem, sibling
    else:
        for path in sorted(ROOT.rglob("perf_*.py")):
            if QUASAR_DIR not in path.parts:
                yield path.stem, path


def derive_perf_test_schemas(quasar: bool = False) -> dict:
    """{perf_test_name -> sorted column list}, derived statically from PerfConfig.

    ``quasar=False`` derives the WH/BH tests; ``quasar=True`` derives the Quasar
    tests (from their sibling test_*_quasar.py source). Columns = the fixed sweep
    headers (formats/flags) + the sweep-parameter fields the test sets + marker.
    Metric/counter columns are formula-driven from run types and covered by the
    global drift gates, not here.

    Static approximation: it reads params passed as literal templates=/runtimes=
    lists (inline, via a variable, or as a dict entry). Params built purely at
    runtime (comprehensions/helpers) are not visible; only the runtime report (or
    the hardware-free report test, #51244) is exact.
    """
    ps = load_pure_module("schema.py")
    fixed = list(ps.FORMAT_HEADERS) + list(ps.FLAG_HEADERS)
    specs = class_field_specs()
    schemas: dict[str, list] = {}
    for key, path in _perf_test_sources(quasar):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        if _is_fuser_perf_test(tree):
            schemas[key] = sorted({MARKER_COLUMN, ps.LOOP_FACTOR_COLUMN})
            continue
        if not _has_perfconfig(tree):
            continue
        fields = _param_fields_in_tree(tree, specs)
        schemas[key] = sorted(set(fixed) | fields | {MARKER_COLUMN})
    return schemas


def enum_member_names(module_filename: str, enum_name: str) -> set:
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
