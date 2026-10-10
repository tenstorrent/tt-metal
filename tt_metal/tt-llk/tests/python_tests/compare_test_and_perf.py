# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare @parametrize sweeps. Not collected by pytest.

``--arch`` is functional vs perf on one architecture. ``--cross-arch LEFT RIGHT``
is perf/perf or func/func; a directory sweep of that mode needs ``--kind``.
Wormhole and Blackhole share files and are imported twice. Quasar files live in
``quasar/`` and drop a trailing ``_quasar``. ``CHIP_ARCH`` selects the architecture
unless ``--arch`` is passed. Comparison is the union of values on each axis.

``math_op`` joins ``mathop`` only when the sides disagree. A tuple axis whose name
is the other side's axes (``dest_sync_dest_acc``) splits when slot types match.
``CROSS_ARCH_FUNCTION_EXCEPTIONS`` pairs a function that lives in another module.
Format lists print shared values first, then side-only values, each alphabetical.
``iterations``, ``loop_factor``, ``run_types``, ``is_perf``, and
``implied_math_format`` are shown and ignored. Composite axes are repeated one
field at a time. ``--csv DIR`` writes that table.

    python compare_test_and_perf.py
    python compare_test_and_perf.py --full
    python compare_test_and_perf.py --dir quasar --arch quasar
    python compare_test_and_perf.py --csv reports/
    python compare_test_and_perf.py <functional.py> <perf.py>
    python compare_test_and_perf.py --cross-arch blackhole quasar --kind perf
    python compare_test_and_perf.py --cross-arch blackhole quasar --kind func
    python compare_test_and_perf.py --cross-arch blackhole quasar perf_op.py quasar/perf_op_quasar.py
"""
from __future__ import annotations

import argparse
import csv
import enum
import importlib
import os
import sys
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SELF = Path(__file__).resolve()
_HERE = _SELF.parent
_BANNER_WIDTH = 88
_VALUE_PREVIEW_LIMIT = 8
_ARCH_CHOICES = ("wormhole", "blackhole", "quasar")
_ARCH_LABELS = {"wormhole": "W", "blackhole": "B", "quasar": "Q"}

# pytest.param() is a ParameterSet, a 3-tuple, so it must not be read as a row.
_PARAMETER_SET = type(pytest.param(None))


def find_python_tests_root(sample: Path) -> Path:
    for parent in [sample.resolve(), *sample.resolve().parents]:
        if (parent / "helpers").is_dir() and (parent / "pytest.ini").exists():
            return parent
    raise RuntimeError(
        f"Could not find python_tests root (helpers/ + pytest.ini) above {sample}"
    )


def _clear_chip_arch_cache() -> None:
    chip = sys.modules.get("helpers.chip_architecture")
    if chip is not None:
        chip._cached_chip_architecture = None


def _reload_local_dependencies(module: ModuleType, root: Path) -> None:
    root = root.resolve()
    for value in list(vars(module).values()):
        if not isinstance(value, ModuleType) or value is module:
            continue
        source = getattr(value, "__file__", None)
        if not source:
            continue
        try:
            Path(source).resolve().relative_to(root)
        except ValueError:
            continue
        _clear_chip_arch_cache()
        importlib.reload(value)


def import_test_module(path: Path, root: Path, arch: str) -> ModuleType:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("LLK_HOME", str(root.parent.parent))
    os.environ["CHIP_ARCH"] = arch
    _clear_chip_arch_cache()
    dotted = ".".join(path.resolve().relative_to(root).with_suffix("").parts)
    loaded = sys.modules.get(dotted)
    if loaded is not None:
        # WH and BH share a file, so reload it and the local modules it imported.
        _reload_local_dependencies(loaded, root)
        _clear_chip_arch_cache()
        return importlib.reload(loaded)
    return importlib.import_module(dotted)


_IGNORED_LEAF_NAMES = frozenset({"implied_math_format", "ImpliedMathFormat"})


def _is_ignored_leaf(name: str, leaf: Any) -> bool:
    tail = name.rsplit(".", 1)[-1].split("#", 1)[0]
    if tail in _IGNORED_LEAF_NAMES:
        return True
    return isinstance(leaf, enum.Enum) and type(leaf).__name__ == "ImpliedMathFormat"


def coverage_canon(value: Any) -> str | None:
    """Canon used for coverage. Drops packed implied_math_format fields."""
    if isinstance(value, enum.Enum) and type(value).__name__ == "ImpliedMathFormat":
        return None
    if is_dataclass(value) and not isinstance(value, type):
        members = []
        for field in fields(value):
            if field.name in _IGNORED_LEAF_NAMES:
                continue
            piece = coverage_canon(getattr(value, field.name))
            if piece is not None:
                members.append(f"{field.name}={piece}")
        return f"{type(value).__name__}({', '.join(members)})"
    if (
        isinstance(value, (list, tuple))
        and value
        and not all(isinstance(item, _SCALAR_TYPES) for item in value)
    ):
        parts = [coverage_canon(item) for item in value]
        return "[" + ", ".join(part for part in parts if part is not None) + "]"
    return canon(value)


def canon(value: Any) -> str:
    if is_dataclass(value) and not isinstance(value, type):
        members = ", ".join(
            f"{field.name}={canon(getattr(value, field.name))}"
            for field in fields(value)
        )
        return f"{type(value).__name__}({members})"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(canon(v) for v in value) + "]"
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if hasattr(value, "name"):
        return str(value.name)
    if hasattr(value, "value"):
        return str(value.value)
    return repr(value)


def parametrized_functions(module: ModuleType) -> dict[str, list]:
    out: dict[str, list] = {}
    for name, obj in vars(module).items():
        if not callable(obj) or getattr(obj, "__module__", None) != module.__name__:
            continue
        pmarks = [
            m
            for m in getattr(obj, "pytestmark", [])
            if getattr(m, "name", None) == "parametrize"
        ]
        if pmarks:
            out[name] = pmarks
    return out


def as_tuple(v: Any, n: int) -> tuple:
    if isinstance(v, _PARAMETER_SET):
        v = tuple(v.values)  # drop the marks and the hand-authored id
    if isinstance(v, tuple) and len(v) == n:
        return v
    return (v,) if n == 1 else tuple(v)


def axis_names_of(argnames: Any) -> list[str]:
    if isinstance(argnames, str):
        argnames = argnames.split(",")
    return [str(name).strip() for name in argnames]


@dataclass
class Sweep:
    axes: dict[str, list[str]]
    rows: list[dict[str, str]]
    raw: dict[str, list[Any]]
    raw_rows: list[dict[str, Any]]


def axis_value_sets(pmarks: list) -> Sweep:
    axis_names: list[str] = []
    rows: list[tuple] | None = None
    for mark in pmarks:
        names = axis_names_of(mark.args[0])
        mark_rows = [as_tuple(v, len(names)) for v in mark.args[1]]
        if rows is None:
            axis_names, rows = names, mark_rows
        else:
            axis_names += names
            rows = [a + b for a in rows for b in mark_rows]
    rows = rows or []

    per_axis: dict[str, list[str]] = {n: [] for n in axis_names}
    raw_per_axis: dict[str, list[Any]] = {n: [] for n in axis_names}
    seen: dict[str, set] = {n: set() for n in axis_names}
    canonical_rows = []
    raw_rows = []
    for row in rows:
        canonical_row = {}
        raw_row = {}
        for name, val in zip(axis_names, row):
            key = coverage_canon(val) or canon(val)
            canonical_row[name] = key
            raw_row[name] = val
            if key not in seen[name]:
                seen[name].add(key)
                per_axis[name].append(key)
                raw_per_axis[name].append(val)
        canonical_rows.append(canonical_row)
        raw_rows.append(raw_row)
    return Sweep(
        axes=per_axis, rows=canonical_rows, raw=raw_per_axis, raw_rows=raw_rows
    )


# Applied only when the two sides use different names.
AXIS_ALIASES = {"math_op": "mathop"}


def _rename_key(mapping: dict, old: str, new: str) -> dict:
    return {(new if key == old else key): value for key, value in mapping.items()}


def _rename_axis(sweep: Sweep, old: str, new: str) -> None:
    sweep.axes = _rename_key(sweep.axes, old, new)
    sweep.raw = _rename_key(sweep.raw, old, new)
    for row in sweep.rows:
        if old in row:
            row[new] = row.pop(old)
    for row in sweep.raw_rows:
        if old in row:
            row[new] = row.pop(old)


def _apply_join_aliases(left: Sweep, right: Sweep) -> None:
    for old, new in AXIS_ALIASES.items():
        if old in left.axes and new not in left.axes and new in right.axes:
            _rename_axis(left, old, new)
        if old in right.axes and new not in right.axes and new in left.axes:
            _rename_axis(right, old, new)


def _split_bundle_name(name: str, atoms: set[str]) -> list[str] | None:
    """Exact cover of ``name`` by two or more ``atoms``, longest first."""
    ordered = sorted(atoms, key=len, reverse=True)

    def cover(rest: str) -> list[str] | None:
        if rest == "":
            return []
        if rest.startswith("_"):
            rest = rest[1:]
            if rest == "":
                return None
        for atom in ordered:
            if rest == atom or rest.startswith(atom + "_"):
                found = cover(rest[len(atom) :])
                if found is not None:
                    return [atom, *found]
        return None

    parts = cover(name)
    if parts is None or len(parts) < 2:
        return None
    return parts


def _joinable_component(value: Any) -> bool:
    return isinstance(value, enum.Enum) or (
        is_dataclass(value) and not isinstance(value, type)
    )


def _expand_bundles(sweep: Sweep, other: Sweep) -> None:
    for axis in list(sweep.axes):
        parts = _split_bundle_name(axis, set(other.axes))
        if parts is None or any(part in sweep.axes for part in parts):
            continue
        values = sweep.raw.get(axis, [])
        if not values or not all(
            isinstance(value, tuple) and len(value) == len(parts) for value in values
        ):
            continue
        columns = list(zip(*values))
        if any(not other.raw.get(part) for part in parts):
            continue
        if not all(
            _joinable_component(item) and type(item) is type(other.raw[part][0])
            for part, column in zip(parts, columns)
            for item in column
        ):
            continue
        spliced_axes: dict[str, list[str]] = {}
        spliced_raw: dict[str, list[Any]] = {}
        for part, column in zip(parts, columns):
            seen: set[str] = set()
            axis_values: list[str] = []
            raw_values: list[Any] = []
            for item in column:
                key = canon(item)
                if key not in seen:
                    seen.add(key)
                    axis_values.append(key)
                    raw_values.append(item)
            spliced_axes[part] = axis_values
            spliced_raw[part] = raw_values

        def splice(mapping: dict, replacement: dict) -> dict:
            out = {}
            for key, value in mapping.items():
                if key == axis:
                    out.update(replacement)
                else:
                    out[key] = value
            return out

        sweep.axes = splice(sweep.axes, spliced_axes)
        sweep.raw = splice(sweep.raw, spliced_raw)
        for row, raw_row in zip(sweep.rows, sweep.raw_rows):
            packed = raw_row.pop(axis)
            row.pop(axis, None)
            for part, item in zip(parts, packed):
                raw_row[part] = item
                row[part] = canon(item)


def align_sweeps(left: Sweep, right: Sweep) -> None:
    _apply_join_aliases(left, right)
    _expand_bundles(left, right)
    _expand_bundles(right, left)


def projected_variant_count(
    rows: list[dict[str, str]], ignored_axes: frozenset[str]
) -> int:
    return len(
        {
            tuple(
                (name, value) for name, value in row.items() if name not in ignored_axes
            )
            for row in rows
        }
    )


_SCALAR_TYPES = (str, bytes, int, float, bool, type(None))


def flatten_param(value: Any, label: str) -> list[tuple[str, Any]]:
    """Dataclass fields keep their names. Non-scalar tuples use the value type.

    A list of plain scalars stays one value: ``[32, 32]`` is one dimension pair.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return [
            leaf
            for field in fields(value)
            for leaf in flatten_param(
                getattr(value, field.name), f"{label}.{field.name}"
            )
        ]
    if isinstance(value, (list, tuple)) and not all(
        isinstance(item, _SCALAR_TYPES) for item in value
    ):
        return [
            leaf for item in value for leaf in flatten_param(item, type(item).__name__)
        ]
    return [(label, value)]


def flatten_axis_value(value: Any, axis: str) -> list[tuple[str, Any]]:
    leaves = flatten_param(value, axis)
    repeated = {
        name for name, count in Counter(n for n, _ in leaves).items() if count > 1
    }
    ordinal: Counter = Counter()
    numbered = []
    for name, leaf in leaves:
        if name in repeated:
            numbered.append((f"{name}#{ordinal[name]}", leaf))
            ordinal[name] += 1
        else:
            numbered.append((name, leaf))
    return numbered


@dataclass
class Parameters:
    values: dict[str, list[str]]
    composite_axes: set[str]
    from_composite: set[str]


def parameter_values(sweep: Sweep, ignored_axes: frozenset[str]) -> Parameters:
    per_param: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    composite_axes: set[str] = set()
    from_composite: set[str] = set()
    for axis, values in sweep.raw.items():
        if axis in ignored_axes:
            continue
        axis_params: set[str] = set()
        kept_names: list[str] = []
        for value in values:
            leaves = flatten_axis_value(value, axis)
            for name, leaf in leaves:
                if _is_ignored_leaf(name, leaf):
                    continue
                kept_names.append(name)
                axis_params.add(name)
                key = canon(leaf)
                if key not in seen.setdefault(name, set()):
                    seen[name].add(key)
                    per_param.setdefault(name, []).append(key)
        if kept_names and kept_names != [axis] * len(values):
            composite_axes.add(axis)
            from_composite |= axis_params
    return Parameters(per_param, composite_axes, from_composite)


def _drop_quasar(name: str) -> str:
    return name[: -len("_quasar")] if name.endswith("_quasar") else name


def normalize(name: str) -> str:
    normalized = name
    for pre in ("test_perf_", "perf_test_", "test_", "perf_"):
        if normalized.startswith(pre):
            normalized = normalized[len(pre) :]
            break
    normalized = _drop_quasar(normalized)
    for suffix in ("_perf", "_test"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


# (left stem, left function) -> (right stem, right function), both normalized.
CROSS_ARCH_FUNCTION_EXCEPTIONS: dict[tuple[str, str], tuple[str, str]] = {
    ("eltwise_binary", "eltwise_binary_dest_reuse"): (
        "eltwise_binary_reuse_dest",
        "eltwise_binary_reuse_dest",
    ),
}


def module_key(path: Path) -> str:
    stem = _drop_quasar(path.stem)
    for prefix in ("perf_", "test_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def apply_cross_arch_exceptions(
    left_by_stem: dict[str, Path], right_by_stem: dict[str, Path]
) -> tuple[dict[str, list[tuple[Path, str, str, bool]]], set[str]]:
    """Map a matched stem to ``(extra path, home function, extra function, extra_on_right)``.

    The home stem must exist on both sides. The extra module is consumed only
    then, so an unmatched architecture file stays in the without-counterpart list.
    ``extra_on_right`` is false when the extra module lives on the left architecture.
    """
    matched = set(left_by_stem) & set(right_by_stem)
    extras: dict[str, list[tuple[Path, str, str, bool]]] = {}
    consumed: set[str] = set()
    for (home_stem, home_function), (
        extra_stem,
        extra_function,
    ) in CROSS_ARCH_FUNCTION_EXCEPTIONS.items():
        if home_stem not in matched or extra_stem in matched:
            continue
        if extra_stem in right_by_stem:
            path, on_right = right_by_stem[extra_stem], True
        elif extra_stem in left_by_stem:
            path, on_right = left_by_stem[extra_stem], False
        else:
            continue
        extras.setdefault(home_stem, []).append(
            (path, home_function, extra_function, on_right)
        )
        consumed.add(extra_stem)
    return extras, consumed


def extras_for_paths(
    left_path: Path, right_path: Path, left_arch: str, right_arch: str
) -> list[tuple[Path, str, str, bool]]:
    """Exception modules for one explicit pair, in either architecture order."""
    kind = module_kind(left_path)
    if kind is None or module_key(left_path) != module_key(right_path):
        return []
    root = find_python_tests_root(left_path)
    left_map = dict(_collect_kind(arch_directory(root, left_arch), kind))
    right_map = dict(_collect_kind(arch_directory(root, right_arch), kind))
    left_map[module_key(left_path)] = left_path
    right_map[module_key(right_path)] = right_path
    extras, _consumed = apply_cross_arch_exceptions(left_map, right_map)
    return extras.get(module_key(left_path), [])


def pair_functions(
    test_funcs: dict, perf_funcs: dict
) -> list[tuple[str | None, str | None, str | None]]:
    if len(test_funcs) == 1 and len(perf_funcs) == 1:
        tname, pname = next(iter(test_funcs)), next(iter(perf_funcs))
        method = "name" if normalize(tname) == normalize(pname) else "position"
        return [(tname, pname, method)]
    perf_by_norm = {normalize(n): n for n in perf_funcs}
    pairs, used = [], set()
    for tname in test_funcs:
        pname = perf_by_norm.get(normalize(tname))
        pairs.append((tname, pname, "name" if pname else None))
        if pname:
            used.add(pname)
    pairs += [(None, pname, None) for pname in perf_funcs if pname not in used]
    return pairs


MEASUREMENT_AXES = frozenset({"iterations", "loop_factor", "run_types", "is_perf"})
IGNORED_AXES = MEASUREMENT_AXES | frozenset({"implied_math_format"})


@dataclass(frozen=True)
class SideNames:
    left: str
    right: str
    left_mark: str
    right_mark: str


FUNCTIONAL_PERF = SideNames("functional", "perf", "T", "P")


def arch_label(arch: str) -> str:
    return _ARCH_LABELS.get(arch, arch)


def _print_names(title: str, paths: list[Path]) -> None:
    if paths:
        print(f"{title}: {', '.join(path.name for path in paths)}")


def arch_sides(left: str, right: str) -> SideNames:
    left_label, right_label = arch_label(left), arch_label(right)
    return SideNames(left_label, right_label, left_label, right_label)


def ignored_reason(axis: str) -> str:
    if axis in MEASUREMENT_AXES:
        return "ignored measurement axis"
    return "ignored"


def _is_format_name(name: str) -> bool:
    return (
        name == "formats"
        or name.startswith(("formats.", "format."))
        or "Format." in name
    )


def order_format_values(
    left: list[str], right: list[str]
) -> tuple[list[str], list[str]]:
    shared = sorted(set(left) & set(right))
    return shared + sorted(set(left) - set(right)), shared + sorted(
        set(right) - set(left)
    )


def _display_values(
    name: str, left: list[str], right: list[str]
) -> tuple[list[str], list[str]]:
    if _is_format_name(name):
        return order_format_values(left, right)
    return left, right


def fmt_values(values: list[str], full: bool) -> str:
    if full or len(values) <= _VALUE_PREVIEW_LIMIT:
        return ", ".join(values) if values else "-"
    return (
        ", ".join(values[:_VALUE_PREVIEW_LIMIT])
        + f", ... (+{len(values) - _VALUE_PREVIEW_LIMIT} more)"
    )


def axis_value_relation(test_values: list[str], perf_values: list[str]) -> str:
    if not test_values or not perf_values:
        return "unreadable"
    set_t, set_p = set(test_values), set(perf_values)
    if set_t == set_p:
        return "identical"
    if set_p <= set_t:
        return "perf_subset"
    if set_t <= set_p:
        return "functional_subset"
    return "different"


def _only_headline(mark: str, owner: str, name: str, kind: str) -> str:
    if mark == "P":
        return f"[P] {name}: PERF-ONLY {kind}"
    if mark == "T":
        return f"[T] {name}: FUNCTIONAL-ONLY {kind}"
    return f"[{mark}] {name}: {owner}-only {kind}"


# verdict, compare, compare_parameters, and write_parameter_csv take ``sides``.
def verdict(
    name: str,
    t: list[str],
    p: list[str],
    in_t: bool,
    in_p: bool,
    kind: str,
    sides: SideNames = FUNCTIONAL_PERF,
) -> tuple[str, str]:
    if not in_t:
        return "diff", _only_headline(sides.right_mark, sides.right, name, kind)
    if not in_p:
        return "diff", _only_headline(sides.left_mark, sides.left, name, kind)
    relation = axis_value_relation(t, p)
    if relation == "identical":
        return "same", f"[=] {name}: identical ({len(t)} value(s))"
    if relation == "unreadable":
        return (
            "unreadable",
            f"[!] {name}: UNREADABLE - no values parsed "
            f"({sides.left}={len(t)}, {sides.right}={len(p)})",
        )
    if relation in ("perf_subset", "functional_subset"):
        smaller, larger, narrow, wide = (
            (sides.right, sides.left, p, t)
            if relation == "perf_subset"
            else (sides.left, sides.right, t, p)
        )
        return (
            "diff",
            f"[~] {name}: {smaller} subset of {larger} "
            f"({len(narrow)}/{len(wide)} value(s))",
        )
    return "diff", f"[x] {name}: DIFFERENT"


def _print_side(
    label: str, values: list[str], full: bool, width: int, indent: str = "        "
) -> None:
    print(f"{indent}{label:<{width}} : {fmt_values(values, full)}")


def _print_both(
    sides: SideNames,
    left: list[str],
    right: list[str],
    full: bool,
    width: int,
    have_left: bool,
    have_right: bool,
    indent: str = "        ",
) -> None:
    if have_left:
        _print_side(sides.left, left, full, width, indent)
    if have_right:
        _print_side(sides.right, right, full, width, indent)


def compare(
    test_axes: dict,
    perf_axes: dict,
    ignored_axes: frozenset[str],
    composite_axes: set[str],
    full: bool,
    sides: SideNames = FUNCTIONAL_PERF,
) -> None:
    same, diff, ignored, unreadable = [], [], [], []
    buckets = {"same": same, "diff": diff, "unreadable": unreadable}
    label_width = max(len(sides.left), len(sides.right))
    for axis in dict.fromkeys([*test_axes, *perf_axes]):
        in_t, in_p = axis in test_axes, axis in perf_axes
        t, p = test_axes.get(axis, []), perf_axes.get(axis, [])

        shown_t, shown_p = _display_values(axis, t, p)
        if axis in ignored_axes:
            ignored.append(axis)
            print(f"  [i] {axis}: {ignored_reason(axis)}")
            _print_both(
                sides,
                shown_t,
                shown_p,
                full,
                width=label_width,
                have_left=in_t,
                have_right=in_p,
            )
            continue

        bucket, headline = verdict(axis, t, p, in_t, in_p, "axis", sides)
        buckets[bucket].append(axis)
        print(f"  {headline}")
        if bucket == "same":
            if full:
                print(f"        values : {fmt_values(shown_t, full)}")
        elif axis in composite_axes and not full:
            print("        values : split per parameter below (--full for tuples)")
        else:
            _print_both(
                sides,
                shown_t,
                shown_p,
                full,
                width=label_width,
                have_left=in_t,
                have_right=in_p,
            )
    print(f"\n  Summary: {len(same)} identical axis/axes, {len(diff)} differing.")
    if same:
        print(f"    identical : {', '.join(same)}")
    if diff:
        print(f"    differing : {', '.join(diff)}")
    measurement = [axis for axis in ignored if axis in MEASUREMENT_AXES]
    other = [axis for axis in ignored if axis not in MEASUREMENT_AXES]
    if measurement:
        print(f"    ignored   : {', '.join(measurement)} (measurement controls)")
    if other:
        print(f"    ignored   : {', '.join(other)}")
    if unreadable:
        print(f"    UNREADABLE: {', '.join(unreadable)} (verdict withheld)")


def compare_parameters(
    functional: Parameters,
    perf: Parameters,
    full: bool,
    sides: SideNames = FUNCTIONAL_PERF,
) -> None:
    test_params, perf_params = functional.values, perf.values
    split = functional.from_composite | perf.from_composite
    names = [n for n in dict.fromkeys([*test_params, *perf_params]) if n in split]
    same, diff, unreadable = [], [], []
    buckets = {"same": same, "diff": diff, "unreadable": unreadable}
    print(f"\n  Composite axes split into {len(names)} parameter(s):")
    for name in names:
        in_t, in_p = name in test_params, name in perf_params
        t, p = test_params.get(name, []), perf_params.get(name, [])
        shown_t, shown_p = _display_values(name, t, p)
        bucket, headline = verdict(name, t, p, in_t, in_p, "parameter", sides)
        buckets[bucket].append(name)
        print(f"    {headline}")
        if bucket != "same" or full:
            _print_both(
                sides,
                shown_t,
                shown_p,
                full,
                width=max(len(sides.left), len(sides.right)),
                have_left=in_t,
                have_right=in_p,
                indent="          ",
            )
    print(
        f"\n  Parameter summary: {len(same)} identical parameter(s), "
        f"{len(diff)} differing."
    )
    if diff:
        print(f"    differing : {', '.join(diff)}")
    if unreadable:
        print(f"    UNREADABLE: {', '.join(unreadable)} (verdict withheld)")


def write_parameter_csv(
    path: Path,
    test_params: dict[str, list[str]],
    perf_params: dict[str, list[str]],
    sides: SideNames = FUNCTIONAL_PERF,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", sides.left, sides.right])
        for name in dict.fromkeys([*test_params, *perf_params]):
            t, p = test_params.get(name, []), perf_params.get(name, [])
            for value in dict.fromkeys([*t, *p]):
                writer.writerow([name, value, int(value in t), int(value in p)])


def discover_pairs(
    directory: Path,
) -> tuple[list[tuple[str, Path, Path]], list[Path], list[Path]]:
    tests: dict[str, Path] = {}
    perfs: dict[str, Path] = {}
    for path in sorted(directory.glob("*.py")):
        if path.resolve() == _SELF:
            continue
        stem = path.stem
        if stem.startswith("test_"):
            tests[stem[len("test_") :]] = path
        elif stem.startswith("perf_"):
            perfs[stem[len("perf_") :]] = path

    matched = [(key, tests[key], perfs[key]) for key in tests if key in perfs]
    matched.sort(key=lambda item: item[0])
    tests_without_perf = [tests[key] for key in sorted(tests) if key not in perfs]
    perfs_without_test = [perfs[key] for key in sorted(perfs) if key not in tests]
    return matched, tests_without_perf, perfs_without_test


def module_kind(path: Path) -> str | None:
    stem = _drop_quasar(path.stem)
    if stem.startswith("perf_"):
        return "perf"
    if stem.startswith("test_"):
        return "func"
    return None


def arch_directory(base: Path, arch: str) -> Path:
    if arch == "quasar":
        return base if base.name == "quasar" else base / "quasar"
    return base.parent if base.name == "quasar" else base


def _collect_kind(directory: Path, kind: str) -> dict[str, Path]:
    prefix = "perf_" if kind == "perf" else "test_"
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.py")):
        if path.resolve() == _SELF or module_kind(path) != kind:
            continue
        found[_drop_quasar(path.stem)[len(prefix) :]] = path
    return found


def discover_cross_arch(
    directory: Path, left_arch: str, right_arch: str, kind: str
) -> tuple[list[tuple[str, Path, Path]], list[Path], list[Path]]:
    left_dir = arch_directory(directory, left_arch)
    right_dir = arch_directory(directory, right_arch)
    left = _collect_kind(left_dir, kind)
    right = _collect_kind(right_dir, kind)
    if left_dir.resolve() == right_dir.resolve():
        matched = [(key, left[key], left[key]) for key in sorted(left)]
        return matched, [], []
    keys = sorted(set(left) & set(right))
    matched = [(key, left[key], right[key]) for key in keys]
    left_only = [left[key] for key in sorted(left) if key not in right]
    right_only = [right[key] for key in sorted(right) if key not in left]
    return matched, left_only, right_only


def compare_pair(
    left_path: Path,
    right_path: Path,
    root: Path,
    left_arch: str,
    full: bool,
    csv_dir: Path | None = None,
    right_arch: str | None = None,
    sides: SideNames = FUNCTIONAL_PERF,
    extra_right: list[tuple[Path, str, str, bool]] | None = None,
) -> bool:
    """True when at least one function pair was compared, or neither side is parametrized.

    ``sides`` labels the two sweeps. Snapshot the left module before the right
    import: a same-file Wormhole/Blackhole pair reloads one module object.
    """
    right_arch = left_arch if right_arch is None else right_arch
    label_width = max(len(sides.left), len(sides.right))
    print("#" * _BANNER_WIDTH)
    print(f"# {left_path.name}  vs  {right_path.name}")
    print("#" * _BANNER_WIDTH)
    try:
        left_mod = import_test_module(left_path, root, left_arch)
        # Hold the left functions before reload mutates a shared module.
        left_funcs = parametrized_functions(left_mod)
        right_mod = import_test_module(right_path, root, right_arch)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        # BaseException, not Exception: sys.exit / pytest.exit at import must not
        # kill the rest of the sweep.
        print(f"  ! skipped: failed to import ({type(exc).__name__}: {exc})\n")
        return False

    right_funcs = parametrized_functions(right_mod)
    # home function -> (extra function name, module, marks). Right means the extra
    # file is the right architecture and fills a missing right function.
    exception_right: dict[str, tuple[str, ModuleType, list]] = {}
    exception_left: dict[str, tuple[str, ModuleType, list]] = {}
    for extra_path, home_function, extra_function, extra_on_right in extra_right or []:
        extra_arch = right_arch if extra_on_right else left_arch
        try:
            extra_mod = import_test_module(extra_path, root, extra_arch)
            extra_funcs = parametrized_functions(extra_mod)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            print(
                "  ! skipped exception "
                f"{home_function} -> {extra_path.name} "
                f"({type(exc).__name__}: {exc})\n"
            )
            continue
        match = next(
            (name for name in extra_funcs if normalize(name) == extra_function),
            None,
        )
        if match is None:
            print(
                f"  ! skipped exception {home_function}: "
                f"{extra_function} not found in {extra_path.name}\n"
            )
            continue
        target = exception_right if extra_on_right else exception_left
        target[home_function] = (match, extra_mod, extra_funcs[match], extra_path)
    if not left_funcs and not right_funcs:
        print("  ! skipped: no parametrized functions on either side\n")
        return True
    if not left_funcs:
        print(f"  ! no parametrized functions found in {left_path.name}\n")
        return False
    if not right_funcs:
        print(f"  ! no parametrized functions found in {right_path.name}\n")
        return False

    compared = False
    used_exceptions: set[str] = set()
    for left_name, right_name, match_method in pair_functions(left_funcs, right_funcs):
        left_mod_for_fn = left_mod
        right_mod_for_fn = right_mod
        left_marks = left_funcs.get(left_name) if left_name else None
        right_marks = right_funcs.get(right_name) if right_name else None
        if (
            right_name is None
            and left_name is not None
            and normalize(left_name) in exception_right
        ):
            right_name, right_mod_for_fn, right_marks, _extra_path = exception_right[
                normalize(left_name)
            ]
            match_method = "exception"
            used_exceptions.add(normalize(left_name))
        if (
            left_name is None
            and right_name is not None
            and normalize(right_name) in exception_left
        ):
            left_name, left_mod_for_fn, left_marks, _extra_path = exception_left[
                normalize(right_name)
            ]
            match_method = "exception"
            used_exceptions.add(normalize(right_name))
        print("=" * _BANNER_WIDTH)
        print(
            f"{sides.left:<{label_width}}: {left_mod_for_fn.__name__}.{left_name or '<none>'}"
        )
        print(
            f"{sides.right:<{label_width}}: "
            f"{right_mod_for_fn.__name__}.{right_name or '<none>'}"
        )
        print("=" * _BANNER_WIDTH)
        if left_name is None or right_name is None:
            print("  (unmatched - no counterpart found)\n")
            continue
        if match_method == "position":
            print("  ! paired by position because normalized function names differ")
        if match_method == "exception":
            print("  ! paired by cross-file exception")
        try:
            left_sweep = axis_value_sets(left_marks)
            right_sweep = axis_value_sets(right_marks)
        except Exception as exc:
            print(
                "  ! skipped: cannot read parametrize marks "
                f"({type(exc).__name__}: {exc})\n"
            )
            continue
        if not left_sweep.rows or not right_sweep.rows:
            print(
                "  ! empty sweep: "
                f"{sides.left}={len(left_sweep.rows)} row(s), "
                f"{sides.right}={len(right_sweep.rows)} row(s)\n"
            )
            continue
        align_sweeps(left_sweep, right_sweep)
        ignored_axes = frozenset(
            axis
            for axis in IGNORED_AXES
            if axis in right_sweep.axes or axis in left_sweep.axes
        )
        left_n = projected_variant_count(left_sweep.rows, ignored_axes)
        right_n = projected_variant_count(right_sweep.rows, ignored_axes)
        print(f"  variants: {sides.left}={left_n}, {sides.right}={right_n}\n")

        left_params = parameter_values(left_sweep, ignored_axes)
        right_params = parameter_values(right_sweep, ignored_axes)
        composite_axes = left_params.composite_axes | right_params.composite_axes
        compare(
            left_sweep.axes,
            right_sweep.axes,
            ignored_axes,
            composite_axes,
            full,
            sides,
        )
        if composite_axes:
            compare_parameters(left_params, right_params, full, sides)
        if csv_dir is not None:
            target = csv_dir / f"{left_path.stem}.{left_name}.csv"
            write_parameter_csv(target, left_params.values, right_params.values, sides)
            print(f"\n  parameter table written to {target}")
        print()
        compared = True
    for home, entry in (*exception_right.items(), *exception_left.items()):
        if home not in used_exceptions:
            print(f"  ! unused cross-file exception: {home} ({entry[3].name})\n")
    return compared


def _validate_arches(arches: list[str], parser: argparse.ArgumentParser) -> None:
    for arch in arches:
        if arch not in _ARCH_CHOICES:
            parser.error(
                f"unknown architecture {arch!r}; choose from {', '.join(_ARCH_CHOICES)}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "functional",
        type=Path,
        nargs="?",
        help="left module (functional, or either side of a cross-arch pair)",
    )
    ap.add_argument(
        "perf",
        type=Path,
        nargs="?",
        help="right module (perf, or the other side of a cross-arch pair)",
    )
    ap.add_argument(
        "--dir",
        type=Path,
        default=_HERE,
        help="folder to sweep for test_*/perf_* pairs (default: this script's folder)",
    )
    ap.add_argument(
        "--full", action="store_true", help="print full value lists (no truncation)"
    )
    ap.add_argument(
        "--csv",
        type=Path,
        metavar="DIR",
        help="also write the untruncated parameter/value table of each pair there",
    )
    env_arch = os.environ.get("CHIP_ARCH", "").lower()
    ap.add_argument(
        "--arch",
        choices=_ARCH_CHOICES,
        default=env_arch if env_arch in _ARCH_CHOICES else "wormhole",
        help="CHIP_ARCH used to resolve a same-arch sweep (default: $CHIP_ARCH or wormhole)",
    )
    ap.add_argument(
        "--cross-arch",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        help="compare perf/perf or func/func across two architectures",
    )
    ap.add_argument(
        "--kind",
        choices=("perf", "func"),
        help="module kind for a --cross-arch directory sweep",
    )
    args = ap.parse_args()

    if bool(args.functional) ^ bool(args.perf):
        ap.error("provide both paths for single-pair mode, or neither to sweep")
    if args.kind and not args.cross_arch:
        ap.error("--kind is only used with --cross-arch")

    if args.cross_arch:
        _validate_arches(args.cross_arch, ap)
        left_arch, right_arch = args.cross_arch
        sides = arch_sides(left_arch, right_arch)
        if args.functional and args.perf:
            left_kind = module_kind(args.functional)
            right_kind = module_kind(args.perf)
            if left_kind is None or left_kind != right_kind:
                ap.error(
                    "cross-arch compares perf/perf or func/func; "
                    "use --arch for functional vs perf"
                )
            if args.kind and args.kind != left_kind:
                ap.error(
                    f"--kind {args.kind} does not match the {left_kind} modules given"
                )
            root = find_python_tests_root(args.functional)
            print(
                f"Resolving sweeps for {arch_label(left_arch)} vs {arch_label(right_arch)}"
            )
            return (
                0
                if compare_pair(
                    args.functional,
                    args.perf,
                    root,
                    left_arch,
                    args.full,
                    args.csv,
                    right_arch,
                    sides,
                    extras_for_paths(args.functional, args.perf, left_arch, right_arch),
                )
                else 1
            )
        if not args.kind:
            ap.error(
                "a --cross-arch directory sweep requires --kind perf or --kind func"
            )
        directory = args.dir.resolve()
        matched, left_only, right_only = discover_cross_arch(
            directory, left_arch, right_arch, args.kind
        )
        left_by_stem = {key: left_path for key, left_path, _right in matched}
        left_by_stem.update({module_key(path): path for path in left_only})
        right_by_stem = {key: right_path for key, _left, right_path in matched}
        right_by_stem.update({module_key(path): path for path in right_only})
        extras, consumed = apply_cross_arch_exceptions(left_by_stem, right_by_stem)
        left_only = [path for path in left_only if module_key(path) not in consumed]
        right_only = [path for path in right_only if module_key(path) not in consumed]
        print(
            f"Sweeping {directory} for {arch_label(left_arch)} vs "
            f"{arch_label(right_arch)} ({args.kind})"
        )
        print(
            f"Matched {len(matched)} pair(s): "
            f"{', '.join(key for key, _, _ in matched) or '-'}"
        )
        _print_names(
            f"{arch_label(left_arch)} without a {arch_label(right_arch)} counterpart",
            left_only,
        )
        _print_names(
            f"{arch_label(right_arch)} without a {arch_label(left_arch)} counterpart",
            right_only,
        )
        print()
        if not matched:
            return 1
        root = find_python_tests_root(matched[0][1])
        results = [
            compare_pair(
                left_path,
                right_path,
                root,
                left_arch,
                args.full,
                args.csv,
                right_arch,
                sides,
                extras.get(key),
            )
            for key, left_path, right_path in matched
        ]
        return 0 if all(results) else 1

    if args.functional and args.perf:
        root = find_python_tests_root(args.functional)
        print(f"Resolving sweeps for CHIP_ARCH={args.arch}")
        return (
            0
            if compare_pair(
                args.functional, args.perf, root, args.arch, args.full, args.csv
            )
            else 1
        )

    directory = args.dir.resolve()
    matched, tests_only, perfs_only = discover_pairs(directory)

    print(f"Sweeping {directory} for CHIP_ARCH={args.arch}")
    print(
        f"Matched {len(matched)} test_/perf_ pair(s): "
        f"{', '.join(k for k, _, _ in matched) or '-'}"
    )
    _print_names("test_* without a perf_* counterpart", tests_only)
    _print_names("perf_* without a test_* counterpart", perfs_only)
    print()

    if not matched:
        return 1

    root = find_python_tests_root(matched[0][1])
    results = []
    for _key, functional, perf in matched:
        results.append(
            compare_pair(functional, perf, root, args.arch, args.full, args.csv)
        )
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
