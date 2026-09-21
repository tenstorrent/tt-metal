# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare the @parametrize sweeps of two test modules.

The default mode maps a functional ``test_*.py`` module to its ``perf_*.py``
counterpart and, axis by axis, reports which parameters are identical and which
differ. For a differing axis it prints both value lists. When one sweep is a
subset of the other it labels that relationship; otherwise it reports a true
mismatch. Measurement control axes (``iterations``, ``loop_factor``,
``run_types``, ``is_perf``) are reported but excluded from coverage comparisons,
on whichever side declares them, because they are measurement knobs and not
coverage dimensions. An axis whose values cannot be read is reported as
unreadable rather than compared.

``--cross-arch LEFT RIGHT`` compares perf modules across architectures instead:
``perf_matmul.py`` under CHIP_ARCH=LEFT against ``quasar/perf_matmul_quasar.py``
under CHIP_ARCH=RIGHT. Function names are paired after stripping ``test_perf_`` /
``perf_`` prefixes and a trailing ``_quasar``.

This is an axis-level comparison: it compares the union of values observed on
each axis, not the dependency or combination structure between axes.

Many sweeps bundle a whole configuration into a single axis, so an axis value can
be a seven element tuple of enums, dataclasses and dimensions. Those axes are also
reported one parameter at a time: each value is flattened into its components
(dataclasses into their fields), named after the component rather than the axis,
so two modules that bundle the same configuration under different axis names still
line up. Only the axes that decompose are repeated that way. ``--csv DIR`` writes
the whole parameter/value table, split and plain axes alike, per function pair.

By default it sweeps a folder (its own folder, i.e. ``python_tests``): it
collects every ``test_*.py`` and ``perf_*.py`` module, pairs them by the part of
the name after the prefix (so ``test_matmul.py`` pairs with ``perf_matmul.py``),
and runs the comparison over each matched pair. Use ``--dir`` to sweep another
folder that contains matched files, or pass two explicit paths for one pair.
Quasar pairs live in the ``quasar`` subfolder, which is a separate sweep.

This is a standalone diagnostic script, not a pytest test: it introspects the
``parametrize`` mark left on each function by the custom ``@parametrize``
decorator and does not invoke pytest. Importing test modules can execute their
normal module and ``conftest`` imports, so run it in the usual LLK environment.
It is not named ``test_*.py`` and pytest does not collect it.

The target architecture comes from ``CHIP_ARCH`` (``export CHIP_ARCH=quasar``) and
``--arch`` overrides it; the resolved value is printed with the sweep header.
``--cross-arch`` sets CHIP_ARCH independently on each side.

Usage (run from the python_tests folder):
    python compare_test_and_perf.py                         # sweep this folder
    python compare_test_and_perf.py --full                  # no value-list truncation
    python compare_test_and_perf.py --dir quasar            # sweep the Quasar pairs
    python compare_test_and_perf.py --arch blackhole        # WH/BH folder, Blackhole skips
    python compare_test_and_perf.py --cross-arch blackhole quasar
    python compare_test_and_perf.py --csv reports/          # export parameter tables
    python compare_test_and_perf.py <left.py> <right.py>    # single explicit pair
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
_ARCH_TAGS = {"wormhole": "WH", "blackhole": "BH", "quasar": "QSR"}


@dataclass(frozen=True)
class Side:
    """One side of a comparison (functional vs perf, or one arch vs another)."""

    label: str
    tag: str


FUNCTIONAL_SIDE = Side("functional", "T")
PERF_SIDE = Side("perf", "P")

# `pytest.param(...)` returns a ParameterSet, which is itself a 3-tuple of
# (values, marks, id) and so is indistinguishable from a 3-axis parameter row.
# The type comes from the public factory rather than from `_pytest.mark`.
_PARAMETER_SET = type(pytest.param(None))


def side_for_arch(arch: str) -> Side:
    return Side(arch, _ARCH_TAGS.get(arch, arch[:2].upper()))


def modules_dir_for_arch(arch: str) -> Path:
    """WH/BH perf and functional tests live at the python_tests root; Quasar in quasar/."""
    return _HERE / "quasar" if arch == "quasar" else _HERE


# --------------------------------------------------------------------------- #
# Import bootstrap: make `helpers`, `quasar`, ... importable without pytest.   #
# --------------------------------------------------------------------------- #
def find_python_tests_root(sample: Path) -> Path:
    for parent in [sample.resolve(), *sample.resolve().parents]:
        if (parent / "helpers").is_dir() and (parent / "pytest.ini").exists():
            return parent
    raise RuntimeError(
        f"Could not find python_tests root (helpers/ + pytest.ini) above {sample}"
    )


def set_chip_architecture(arch: str) -> None:
    """Pin CHIP_ARCH for the next import, including modules already in sys.modules."""
    os.environ["CHIP_ARCH"] = arch
    chip_architecture = sys.modules.get("helpers.chip_architecture")
    if chip_architecture is not None:
        chip_architecture._cached_chip_architecture = None
        chip_architecture.get_chip_architecture()
    test_config = sys.modules.get("helpers.test_config")
    if test_config is not None and hasattr(test_config.TestConfig, "setup_arch"):
        try:
            test_config.TestConfig.setup_arch()
        except Exception:
            pass


def import_test_module(path: Path, root: Path, arch: str) -> ModuleType:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Replicate conftest's LLK_HOME default for modules that do not import it.
    os.environ.setdefault("LLK_HOME", str(root.parent.parent))
    # Importing a test module runs get_chip_architecture() at module load. Without
    # CHIP_ARCH it probes for a physical device (or simulator context) and fails on a
    # plain host. The explicit CLI selection must override the caller's environment
    # and any architecture cached from a previous import (cross-arch mode).
    set_chip_architecture(arch)
    dotted = ".".join(path.resolve().relative_to(root).with_suffix("").parts)
    return importlib.import_module(dotted)


# --------------------------------------------------------------------------- #
# Extract parametrize axes/values from a module.                              #
# --------------------------------------------------------------------------- #
def canon(value: Any) -> str:
    """Return a readable key that preserves every dataclass configuration field."""
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
    """name -> list of parametrize Mark objects, for funcs *defined* in this module."""
    out: dict[str, list] = {}
    for name, obj in vars(module).items():
        if not callable(obj) or getattr(obj, "__module__", None) != module.__name__:
            continue  # skips imported symbols like the `run_matmul` alias
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
    """Normalize the parametrize argnames, which may be a string or a sequence."""
    if isinstance(argnames, str):
        argnames = argnames.split(",")
    return [str(name).strip() for name in argnames]


@dataclass
class Sweep:
    """One test function's resolved parametrize sweep."""

    axes: dict[str, list[str]]  # axis -> ordered unique canonical values
    rows: list[dict[str, str]]  # one canonicalized row per parameter combination
    raw: dict[str, list[Any]]  # axis -> the same values before canonicalization


def axis_value_sets(pmarks: list) -> Sweep:
    """Collect the axes, the rows, and the raw values of one parametrize sweep."""
    axis_names: list[str] = []
    rows: list[tuple] | None = None
    for mark in pmarks:
        names = axis_names_of(mark.args[0])
        mark_rows = [as_tuple(v, len(names)) for v in mark.args[1]]
        if rows is None:
            axis_names, rows = names, mark_rows
        else:  # stacked @parametrize marks -> cartesian product
            axis_names += names
            rows = [a + b for a in rows for b in mark_rows]
    rows = rows or []

    per_axis: dict[str, list[str]] = {n: [] for n in axis_names}
    raw_per_axis: dict[str, list[Any]] = {n: [] for n in axis_names}
    seen: dict[str, set] = {n: set() for n in axis_names}
    canonical_rows = []
    for row in rows:
        canonical_row = {}
        for name, val in zip(axis_names, row):
            key = canon(val)
            canonical_row[name] = key
            if key not in seen[name]:
                seen[name].add(key)
                per_axis[name].append(key)
                raw_per_axis[name].append(val)
        canonical_rows.append(canonical_row)
    return Sweep(axes=per_axis, rows=canonical_rows, raw=raw_per_axis)


def projected_variant_count(
    rows: list[dict[str, str]], ignored_axes: frozenset[str]
) -> int:
    """Count unique parameter rows after removing ignored measurement axes."""
    return len(
        {
            tuple(
                (name, value) for name, value in row.items() if name not in ignored_axes
            )
            for row in rows
        }
    )


# --------------------------------------------------------------------------- #
# Flatten composite axes into individual parameters.                          #
# --------------------------------------------------------------------------- #
_SCALAR_TYPES = (str, bytes, int, float, bool, type(None))


def flatten_param(value: Any, label: str) -> list[tuple[str, Any]]:
    """Split one axis value into (parameter name, leaf value) pairs.

    LLK sweeps often bundle a whole configuration into one axis, so
    ``mathop_formats_dest_acc_sync_implied_math_input_dims`` carries a seven
    element tuple. A dataclass expands into its fields and a tuple holding
    non-scalars expands per position, where the component type supplies the name:
    two modules may order the components differently, so a positional name would
    pair unrelated parameters. A list of plain scalars stays whole, because
    ``[32, 32]`` is one dimension pair rather than two integers.
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
    """Flatten one axis value, numbering parameters that share a name."""
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
    """The parameters one sweep is built from, after flattening its axes."""

    values: dict[str, list[str]]  # parameter -> ordered unique canonical values
    composite_axes: set[str]  # axes that decomposed into several parameters
    from_composite: set[str]  # parameters that came out of such an axis


def parameter_values(sweep: Sweep, ignored_axes: frozenset[str]) -> Parameters:
    """Flatten every axis of a sweep into per-parameter value lists.

    An axis that holds one plain value per row contributes a single parameter
    under its own name; only the axes that really decomposed are worth reporting
    again, so those are tracked separately.
    """
    per_param: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    composite_axes: set[str] = set()
    from_composite: set[str] = set()
    for axis, values in sweep.raw.items():
        if axis in ignored_axes:
            continue
        axis_params: set[str] = set()
        for value in values:
            leaves = flatten_axis_value(value, axis)
            if [name for name, _ in leaves] != [axis]:
                composite_axes.add(axis)
            for name, leaf in leaves:
                axis_params.add(name)
                key = canon(leaf)
                if key not in seen.setdefault(name, set()):
                    seen[name].add(key)
                    per_param.setdefault(name, []).append(key)
        if axis in composite_axes:
            from_composite |= axis_params
    return Parameters(per_param, composite_axes, from_composite)


# --------------------------------------------------------------------------- #
# Pairing + comparison.                                                       #
# --------------------------------------------------------------------------- #
def normalize(name: str) -> str:
    normalized = name
    for pre in ("test_perf_", "perf_test_", "test_", "perf_"):
        if normalized.startswith(pre):
            normalized = normalized[len(pre) :]
            break
    for suffix in ("_quasar", "_perf", "_test"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


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


# Measurement controls are shown but excluded from coverage verdicts. Functional
# modules declare some of them too (Quasar tests pin `run_types` and `loop_factor`),
# so an axis is excluded wherever it appears, not only when it is perf-only.
MEASUREMENT_AXES = frozenset({"iterations", "loop_factor", "run_types", "is_perf"})


def fmt_values(values: list[str], full: bool) -> str:
    if full or len(values) <= _VALUE_PREVIEW_LIMIT:
        return ", ".join(values) if values else "-"
    return (
        ", ".join(values[:_VALUE_PREVIEW_LIMIT])
        + f", ... (+{len(values) - _VALUE_PREVIEW_LIMIT} more)"
    )


def _label_width(left: Side, right: Side) -> int:
    return max(len(left.label), len(right.label))


def fmt_side(side: Side, left: Side, right: Side) -> str:
    return f"{side.label:<{_label_width(left, right)}}"


def axis_value_relation(left_values: list[str], right_values: list[str]) -> str:
    """Classify how the two value sets relate (right subset of left, or reverse)."""
    if not left_values or not right_values:
        # A declared axis with no values means the rows could not be read, so the
        # two sides must not be reported as identical (or as a deliberate subset).
        return "unreadable"
    set_left, set_right = set(left_values), set(right_values)
    if set_left == set_right:
        return "identical"
    if set_right <= set_left:
        return "right_subset"
    if set_left <= set_right:
        return "left_subset"
    return "different"


def verdict(
    name: str,
    left_values: list[str],
    right_values: list[str],
    in_left: bool,
    in_right: bool,
    kind: str,
    left: Side,
    right: Side,
) -> tuple[str, str]:
    """Classify one axis or parameter and render its headline."""
    if not in_left:
        return "diff", f"[{right.tag}] {name}: {right.label.upper()}-ONLY {kind}"
    if not in_right:
        return "diff", f"[{left.tag}] {name}: {left.label.upper()}-ONLY {kind}"
    relation = axis_value_relation(left_values, right_values)
    if relation == "identical":
        return "same", f"[=] {name}: identical ({len(left_values)} value(s))"
    if relation == "unreadable":
        return (
            "unreadable",
            f"[!] {name}: UNREADABLE - no values parsed "
            f"({left.label}={len(left_values)}, {right.label}={len(right_values)})",
        )
    if relation == "right_subset":
        return (
            "diff",
            f"[~] {name}: {right.label} subset of {left.label} "
            f"({len(right_values)}/{len(left_values)} value(s))",
        )
    if relation == "left_subset":
        return (
            "diff",
            f"[~] {name}: {left.label} subset of {right.label} "
            f"({len(left_values)}/{len(right_values)} value(s))",
        )
    return "diff", f"[x] {name}: DIFFERENT"


def compare(
    left_axes: dict,
    right_axes: dict,
    ignored_axes: frozenset[str],
    composite_axes: set[str],
    full: bool,
    left: Side,
    right: Side,
) -> None:
    """Report identical vs differing axes; for differing ones print both sweeps."""
    same, diff, ignored, unreadable = [], [], [], []
    buckets = {"same": same, "diff": diff, "unreadable": unreadable}
    left_hdr, right_hdr = fmt_side(left, left, right), fmt_side(right, left, right)
    for axis in dict.fromkeys([*left_axes, *right_axes]):
        in_left, in_right = axis in left_axes, axis in right_axes
        left_vals, right_vals = left_axes.get(axis, []), right_axes.get(axis, [])

        if axis in ignored_axes:
            ignored.append(axis)
            print(f"  [i] {axis}: ignored measurement axis")
            if in_left:
                print(f"        {left_hdr} : {fmt_values(left_vals, full)}")
            print(f"        {right_hdr} : {fmt_values(right_vals, full)}")
            continue

        bucket, headline = verdict(
            axis, left_vals, right_vals, in_left, in_right, "axis", left, right
        )
        buckets[bucket].append(axis)
        print(f"  {headline}")
        if bucket == "same":
            if full:
                print(f"        values : {fmt_values(left_vals, full)}")
        elif axis in composite_axes and not full:
            # Whole configuration tuples are unreadable; the parameter view below
            # carries the same information one parameter at a time.
            print("        values : split per parameter below (--full for tuples)")
        else:
            print(f"        {left_hdr} : {fmt_values(left_vals, full)}")
            print(f"        {right_hdr} : {fmt_values(right_vals, full)}")
    print(f"\n  Summary: {len(same)} identical axis/axes, {len(diff)} differing.")
    if same:
        print(f"    identical : {', '.join(same)}")
    if diff:
        print(f"    differing : {', '.join(diff)}")
    if ignored:
        print(f"    ignored   : {', '.join(ignored)} (measurement controls)")
    if unreadable:
        print(f"    UNREADABLE: {', '.join(unreadable)} (verdict withheld)")


def compare_parameters(
    left_params: Parameters,
    right_params: Parameters,
    full: bool,
    left: Side,
    right: Side,
) -> None:
    """Report the parameters that composite axes were built from, one at a time.

    Parameters carry the name of the component they came from rather than the name
    of the axis that bundled them, so two modules that pack the same configuration
    into differently named tuples still line up parameter by parameter. Axes that
    did not decompose are left out, since the axis view already reported them.
    """
    left_values, right_values = left_params.values, right_params.values
    split = left_params.from_composite | right_params.from_composite
    names = [n for n in dict.fromkeys([*left_values, *right_values]) if n in split]
    same, diff, unreadable = [], [], []
    buckets = {"same": same, "diff": diff, "unreadable": unreadable}
    left_hdr, right_hdr = fmt_side(left, left, right), fmt_side(right, left, right)
    print(f"\n  Composite axes split into {len(names)} parameter(s):")
    for name in names:
        in_left, in_right = name in left_values, name in right_values
        lv, rv = left_values.get(name, []), right_values.get(name, [])
        bucket, headline = verdict(
            name, lv, rv, in_left, in_right, "parameter", left, right
        )
        buckets[bucket].append(name)
        print(f"    {headline}")
        if bucket != "same" or full:
            print(f"          {left_hdr} : {fmt_values(lv, full)}")
            print(f"          {right_hdr} : {fmt_values(rv, full)}")
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
    left_params: dict[str, list[str]],
    right_params: dict[str, list[str]],
    left: Side,
    right: Side,
) -> None:
    """Write the full, untruncated parameter/value table for one function pair."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", left.label, right.label])
        for name in dict.fromkeys([*left_params, *right_params]):
            lv, rv = left_params.get(name, []), right_params.get(name, [])
            for value in dict.fromkeys([*lv, *rv]):
                writer.writerow([name, value, int(value in lv), int(value in rv)])


def perf_stem_key(stem: str) -> str:
    """``perf_matmul`` and ``perf_matmul_quasar`` share the key ``matmul``."""
    key = stem[len("perf_") :] if stem.startswith("perf_") else stem
    if key.endswith("_quasar"):
        key = key[: -len("_quasar")]
    return key


def discover_pairs(
    directory: Path,
) -> tuple[list[tuple[str, Path, Path]], list[Path], list[Path]]:
    """Pair test_*.py and perf_*.py by the name after the prefix.

    Returns (matched, tests_without_perf, perfs_without_test) where `matched`
    is a list of (common_name, functional_path, perf_path).
    """
    tests: dict[str, Path] = {}
    perfs: dict[str, Path] = {}
    for path in sorted(directory.glob("*.py")):
        if path.resolve() == _SELF:
            continue  # never pair this script with itself
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


def discover_cross_arch_perf_pairs(
    left_dir: Path, right_dir: Path
) -> tuple[list[tuple[str, Path, Path]], list[Path], list[Path]]:
    """Pair perf_*.py files across two folders by the op name after perf_/_quasar."""
    left: dict[str, Path] = {}
    right: dict[str, Path] = {}
    for path in sorted(left_dir.glob("perf_*.py")):
        if path.resolve() == _SELF:
            continue
        left[perf_stem_key(path.stem)] = path
    for path in sorted(right_dir.glob("perf_*.py")):
        if path.resolve() == _SELF:
            continue
        right[perf_stem_key(path.stem)] = path

    matched = [(key, left[key], right[key]) for key in left if key in right]
    matched.sort(key=lambda item: item[0])
    left_only = [left[key] for key in sorted(left) if key not in right]
    right_only = [right[key] for key in sorted(right) if key not in left]
    return matched, left_only, right_only


def compare_pair(
    left_path: Path,
    right_path: Path,
    root: Path,
    left_arch: str,
    right_arch: str,
    full: bool,
    csv_dir: Path | None = None,
    left: Side = FUNCTIONAL_SIDE,
    right: Side = PERF_SIDE,
) -> bool:
    """Import a module pair and print their axis comparison.

    ``left_arch`` / ``right_arch`` are the CHIP_ARCH values used for each import.
    Cross-arch mode passes different values so Blackhole and Quasar parametrize
    marks resolve independently.

    Returns True if at least one function pair was compared, False otherwise.
    """
    print("#" * _BANNER_WIDTH)
    print(f"# {left_path.name}  vs  {right_path.name}")
    print("#" * _BANNER_WIDTH)
    try:
        left_mod = import_test_module(left_path, root, left_arch)
        right_mod = import_test_module(right_path, root, right_arch)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # keep sweeping even if a module aborts on import
        # BaseException (not just Exception) so a module that calls sys.exit /
        # pytest.exit at import time is skipped instead of killing the whole sweep.
        print(f"  ! skipped: failed to import ({type(exc).__name__}: {exc})\n")
        return False

    left_funcs = parametrized_functions(left_mod)
    right_funcs = parametrized_functions(right_mod)
    if not left_funcs:
        print(f"  ! no parametrized functions found in {left_path.name}\n")
        return False
    if not right_funcs:
        print(f"  ! no parametrized functions found in {right_path.name}\n")
        return False

    left_hdr, right_hdr = fmt_side(left, left, right), fmt_side(right, left, right)
    compared = False
    for lname, rname, match_method in pair_functions(left_funcs, right_funcs):
        print("=" * _BANNER_WIDTH)
        print(f"{left_hdr}: {left_mod.__name__}.{lname or '<none>'}")
        print(f"{right_hdr}: {right_mod.__name__}.{rname or '<none>'}")
        print("=" * _BANNER_WIDTH)
        if lname is None or rname is None:
            print("  (unmatched - no counterpart found)\n")
            continue
        if match_method == "position":
            print("  ! paired by position because normalized function names differ")
        try:
            left_sweep = axis_value_sets(left_funcs[lname])
            right_sweep = axis_value_sets(right_funcs[rname])
        except Exception as exc:  # one odd sweep must not abort the whole run
            print(
                "  ! skipped: cannot read parametrize marks "
                f"({type(exc).__name__}: {exc})\n"
            )
            continue
        if not left_sweep.rows or not right_sweep.rows:
            print(
                "  ! empty sweep: "
                f"{left.label}={len(left_sweep.rows)} row(s), "
                f"{right.label}={len(right_sweep.rows)} row(s)\n"
            )
            continue
        ignored_axes = frozenset(
            axis
            for axis in MEASUREMENT_AXES
            if axis in right_sweep.axes or axis in left_sweep.axes
        )
        left_n = projected_variant_count(left_sweep.rows, ignored_axes)
        right_n = projected_variant_count(right_sweep.rows, ignored_axes)
        print(f"  variants: {left.label}={left_n}, {right.label}={right_n}\n")

        left_params = parameter_values(left_sweep, ignored_axes)
        right_params = parameter_values(right_sweep, ignored_axes)
        composite_axes = left_params.composite_axes | right_params.composite_axes
        compare(
            left_sweep.axes,
            right_sweep.axes,
            ignored_axes,
            composite_axes,
            full,
            left,
            right,
        )
        if composite_axes:
            compare_parameters(left_params, right_params, full, left, right)
        if csv_dir is not None:
            target = csv_dir / f"{left_path.stem}.{lname}.csv"
            write_parameter_csv(
                target, left_params.values, right_params.values, left, right
            )
            print(f"\n  parameter table written to {target}")
        print()
        compared = True
    return compared


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "left",
        type=Path,
        nargs="?",
        help="left module (functional in test-vs-perf; left-arch perf in --cross-arch)",
    )
    ap.add_argument(
        "right",
        type=Path,
        nargs="?",
        help="right module (perf in test-vs-perf; right-arch perf in --cross-arch)",
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
    # An exported CHIP_ARCH is the usual way to select the target, so it wins over
    # the built-in default and only an explicit --arch overrides it.
    env_arch = os.environ.get("CHIP_ARCH", "").lower()
    ap.add_argument(
        "--arch",
        choices=_ARCH_CHOICES,
        default=env_arch if env_arch in _ARCH_CHOICES else "wormhole",
        help="CHIP_ARCH used to resolve the sweeps (default: $CHIP_ARCH or wormhole)",
    )
    ap.add_argument(
        "--cross-arch",
        nargs=2,
        metavar=("LEFT_ARCH", "RIGHT_ARCH"),
        help=(
            "compare perf_*.py of LEFT_ARCH against perf_*_quasar.py of RIGHT_ARCH "
            "(typically: --cross-arch blackhole quasar)"
        ),
    )
    args = ap.parse_args()

    if bool(args.left) ^ bool(args.right):
        ap.error(
            "provide both left and right modules for single-pair mode, or neither to sweep"
        )

    if args.cross_arch:
        left_arch, right_arch = (a.lower() for a in args.cross_arch)
        for arch in (left_arch, right_arch):
            if arch not in _ARCH_CHOICES:
                ap.error(
                    f"invalid --cross-arch architecture {arch!r}; "
                    f"choose from {', '.join(_ARCH_CHOICES)}"
                )
        left_side, right_side = side_for_arch(left_arch), side_for_arch(right_arch)
        if args.left and args.right:
            root = find_python_tests_root(args.left)
            print(
                f"Resolving sweeps for {left_arch} vs {right_arch}: "
                f"{args.left.name} / {args.right.name}"
            )
            return (
                0
                if compare_pair(
                    args.left,
                    args.right,
                    root,
                    left_arch,
                    right_arch,
                    args.full,
                    args.csv,
                    left_side,
                    right_side,
                )
                else 1
            )

        left_dir, right_dir = modules_dir_for_arch(left_arch), modules_dir_for_arch(
            right_arch
        )
        matched, left_only, right_only = discover_cross_arch_perf_pairs(
            left_dir, right_dir
        )
        print(
            f"Sweeping perf modules: {left_arch} ({left_dir}) vs "
            f"{right_arch} ({right_dir})"
        )
        print(
            f"Matched {len(matched)} perf pair(s): "
            f"{', '.join(k for k, _, _ in matched) or '-'}"
        )
        if left_only:
            print(
                f"{left_arch} perf without a {right_arch} counterpart: "
                f"{', '.join(p.name for p in left_only)}"
            )
        if right_only:
            print(
                f"{right_arch} perf without a {left_arch} counterpart: "
                f"{', '.join(p.name for p in right_only)}"
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
                right_arch,
                args.full,
                args.csv,
                left_side,
                right_side,
            )
            for _key, left_path, right_path in matched
        ]
        return 0 if all(results) else 1

    # Single-pair mode: explicit functional + perf paths.
    if args.left and args.right:
        root = find_python_tests_root(args.left)
        print(f"Resolving sweeps for CHIP_ARCH={args.arch}")
        return (
            0
            if compare_pair(
                args.left,
                args.right,
                root,
                args.arch,
                args.arch,
                args.full,
                args.csv,
            )
            else 1
        )

    # Sweep mode: pair every test_*/perf_* in the folder by common name.
    directory = args.dir.resolve()
    matched, tests_only, perfs_only = discover_pairs(directory)

    print(f"Sweeping {directory} for CHIP_ARCH={args.arch}")
    print(
        f"Matched {len(matched)} test_/perf_ pair(s): "
        f"{', '.join(k for k, _, _ in matched) or '-'}"
    )
    if tests_only:
        print(
            f"test_* without a perf_* counterpart: {', '.join(p.name for p in tests_only)}"
        )
    if perfs_only:
        print(
            f"perf_* without a test_* counterpart: {', '.join(p.name for p in perfs_only)}"
        )
    print()

    if not matched:
        return 1

    root = find_python_tests_root(matched[0][1])
    results = [
        compare_pair(functional, perf, root, args.arch, args.arch, args.full, args.csv)
        for _key, functional, perf in matched
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
