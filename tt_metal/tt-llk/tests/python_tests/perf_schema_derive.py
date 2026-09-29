# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Static derivation of per-perf-test CSV schemas (no hardware).

Helpers the merge gate uses to re-derive, from source, the columns each perf
test's CSV would carry, so a change to a test's columns is caught as drift from
the recorded catalog in helpers/perf/test_schemas.py. Everything parses source
with ``ast`` and touches no device libraries.

Two arch families, derived separately:

  WH/BH  perf_*.py at the python_tests root, derived from themselves
         unless they are thin wrappers and the sibling test_*.py holds
         create_test_or_perf_config (Quasar-style shared harness).
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


def _call_name(elt: ast.Call):
    return (
        elt.func.id
        if isinstance(elt.func, ast.Name)
        else getattr(elt.func, "attr", None)
    )


def _self_attr(node):
    """``self.x`` -> "x", anything else -> None."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.attr if node.value.id == "self" else None
    return None


def helper_returned_param_calls(specs: dict) -> dict:
    """Map helper-function name -> the param constructor Calls it returns.

    ``generate_input_dim(...)`` returns ``INPUT_DIMENSIONS(...)``. Tests pass the
    helper in templates=, so the derivation must follow the return.
    """
    helpers: dict[str, list] = {}
    for path in iter_source_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            # Returns from a function nested inside fn belong to that function.
            nested = {
                n
                for d in ast.walk(fn)
                if isinstance(d, ast.FunctionDef) and d is not fn
                for n in ast.walk(d)
            }
            got = [
                n.value
                for n in ast.walk(fn)
                if isinstance(n, ast.Return)
                and n not in nested
                and isinstance(n.value, ast.Call)
                and _call_name(n.value) in specs
            ]
            if not got:
                continue
            prior = helpers.get(fn.name)
            if prior is None:
                helpers[fn.name] = got
            elif sorted(map(_call_name, prior)) != sorted(map(_call_name, got)):
                # First-wins would drop the other one's columns without a word,
                # which is the blind spot this module exists to remove.
                raise ValueError(
                    f"two helpers named '{fn.name}' return different parameter "
                    f"classes ({sorted(map(_call_name, prior))} vs "
                    f"{sorted(map(_call_name, got))}); rename one, the derivation "
                    "cannot tell which a test called."
                )
    return helpers


def class_post_init_copies() -> dict:
    """Map param-class name -> {dest: src} for ``if self.dest is None: self.dest = self.src``.

    ``NUM_BLOCKS`` fills input_*/output_* this way: one field set, three emitted.
    """
    copies: dict[str, dict] = {}
    for path in iter_source_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            if not any(
                isinstance(b, ast.Name) and b.id in PARAM_BASES for b in cls.bases
            ):
                continue
            found = {}
            for fn in cls.body:
                if not (isinstance(fn, ast.FunctionDef) and fn.name == "__post_init__"):
                    continue
                for node in ast.walk(fn):
                    test = getattr(node, "test", None)
                    if not (
                        isinstance(node, ast.If)
                        and isinstance(test, ast.Compare)
                        and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.Is)
                        and isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value is None
                    ):
                        continue
                    dest = _self_attr(test.left)
                    for stmt in node.body:
                        if (
                            dest
                            and isinstance(stmt, ast.Assign)
                            and len(stmt.targets) == 1
                            and _self_attr(stmt.targets[0]) == dest
                            and _self_attr(stmt.value)
                        ):
                            found[dest] = _self_attr(stmt.value)
            if not found:
                continue
            prior = copies.get(cls.name)
            if prior is None:
                copies[cls.name] = found
            elif prior != found:
                raise ValueError(
                    f"parameter class '{cls.name}' is declared more than once with "
                    f"different __post_init__ copies ({prior} vs {found}); "
                    "consolidate them or rename one."
                )
    return copies


def emitted_fields(
    elt: ast.Call, specs: dict, helpers: dict = None, post_init: dict = None
) -> list:
    """Fields a param instantiation actually emits as CSV columns.

    Mirrors the runtime rule: a None-default field left unset is dropped, so we
    emit a field unless (its default is None AND the call does not set it). This
    stops us over-listing optional slots like MATH_OP.pool_type/unary_extra.

    ``helpers`` and ``post_init`` add the two constructs the plain rule misses: a
    helper that returns a param object, and a ``__post_init__`` copy of a field.
    """
    name = _call_name(elt)
    if name not in specs and helpers and name in helpers:
        return [
            f
            for c in helpers[name]
            for f in emitted_fields(c, specs, helpers, post_init)
        ]
    fields = specs.get(name)
    if fields is None:
        return []
    provided = {fields[i][0] for i in range(min(len(elt.args), len(fields)))}
    provided |= {kw.arg for kw in elt.keywords if kw.arg}
    emitted = [f for f, dflt_none in fields if not (dflt_none and f not in provided)]
    for dest, src in (post_init or {}).get(name, {}).items():
        if src in emitted and dest not in emitted:
            emitted.append(dest)
    return emitted


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


def _param_fields_in_tree(tree, specs, helpers=None, post_init=None) -> set:
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
                fields.update(emitted_fields(elt, specs, helpers, post_init))

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

    WH/BH: perf_*.py at the python_tests root, derived from itself unless the
    sibling test_*.py holds create_test_or_perf_config and the wrapper does not.
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
        for path in sorted(ROOT.glob("perf_*.py")):
            sibling = path.parent / path.name.replace("perf_", "test_", 1)
            source = path
            if sibling.exists():
                try:
                    sibling_tree = ast.parse(sibling.read_text())
                    perf_tree = ast.parse(path.read_text())
                except SyntaxError:
                    sibling_tree = None
                    perf_tree = None
                if (
                    sibling_tree
                    and _has_perfconfig(sibling_tree)
                    and (not perf_tree or not _has_perfconfig(perf_tree))
                ):
                    source = sibling
            yield path.stem, source


def derive_perf_test_schemas(quasar: bool = False) -> dict:
    """{perf_test_name -> sorted column list}, derived statically from PerfConfig.

    ``quasar=False`` derives the WH/BH tests; ``quasar=True`` derives the Quasar
    tests (from their sibling test_*_quasar.py source). Columns = the fixed sweep
    headers (formats/flags) + the sweep-parameter fields the test sets + marker.
    Metric/counter columns are formula-driven from run types and covered by the
    global drift gates, not here.

    Static approximation: it reads params passed as literal templates=/runtimes=
    lists (inline, via a variable, or as a dict entry), follows a helper that
    returns a param object, and includes ``__post_init__`` copies of None-default
    fields. Params built purely at runtime (comprehensions) are still invisible,
    which is why ``_assert_matches_catalog`` re-checks the catalog against the CSV
    a run actually produced.
    """
    ps = load_pure_module("schema.py")
    fixed = list(ps.FORMAT_HEADERS) + list(ps.FLAG_HEADERS)
    specs = class_field_specs()
    helpers = helper_returned_param_calls(specs)
    post_init = class_post_init_copies()
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
        fields = _param_fields_in_tree(tree, specs, helpers, post_init)
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
