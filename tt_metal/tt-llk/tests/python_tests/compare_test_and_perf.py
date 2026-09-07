# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare the @parametrize sweeps of a functional test module vs a perf module.

For each matched pair it maps parametrized functions in the functional module to
counterparts in the perf module and, axis by axis, reports which parameters are
identical and which differ. For a differing axis it prints both value lists.
When one sweep is a subset of the other it labels that relationship (perf subset
of functional, or the reverse); otherwise it reports a true mismatch. Measurement
control axes (``iterations``, ``loop_factor``, ``run_types``, ``is_perf``) are
reported but excluded from coverage comparisons, on whichever side declares them,
because they are measurement knobs and not coverage dimensions. An axis whose
values cannot be read is reported as unreadable rather than compared.

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
and runs the comparison over each matched pair. Functions that do not pair by
name still have their sweeps printed (functional-only or perf-only), and files
with no counterpart get the same one-sided dump in a trailing section, so the
report is not only the functions that happened to share a name. Use ``--dir``
to sweep another folder that contains matched files, or pass two explicit paths
for one pair. Quasar pairs live in the ``quasar`` subfolder, which is a
separate sweep. ``--perf-dirs LEFT RIGHT`` compares ``perf_*.py`` modules in
two folders instead of functional vs perf: ``perf_matmul.py`` pairs with
``perf_matmul_quasar.py``. Each side is imported with its own ``CHIP_ARCH``
(``--arch`` / ``--left-arch`` for the left folder, ``--right-arch`` or the
right folder's name for the right). Value lists are truncated unless
``--full`` is passed. ``--matches-only`` reports the paired modules and skips
the unpaired dumps.

This is a standalone diagnostic script, not a pytest test: it introspects the
``parametrize`` mark left on each function by the custom ``@parametrize``
decorator and does not invoke pytest. Importing test modules can execute their
normal module and ``conftest`` imports, so run it in the usual LLK environment.
It is not named ``test_*.py`` and pytest does not collect it.

The target architecture comes from ``CHIP_ARCH`` (``export CHIP_ARCH=quasar``) and
``--arch`` overrides it; the resolved value is printed with the sweep header.

Usage (run from the python_tests folder):
    python compare_test_and_perf.py                         # sweep this folder
    python compare_test_and_perf.py --full                  # no value-list truncation
    python compare_test_and_perf.py --dir quasar            # sweep the Quasar pairs
    python compare_test_and_perf.py --csv reports/          # export parameter tables
    python compare_test_and_perf.py <functional.py> <perf.py>   # single explicit pair
    python compare_test_and_perf.py --perf-dirs . quasar --arch wormhole
    python compare_test_and_perf.py --perf-dirs . quasar --arch blackhole
    python compare_test_and_perf.py --perf-dirs . quasar --arch blackhole --full --matches-only
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
_ARCH_FILE_SUFFIXES = ("_quasar", "_wormhole", "_blackhole")
_ARCH_TAGS = {"wormhole": "[W]", "blackhole": "[B]", "quasar": "[Q]"}

# `pytest.param(...)` returns a ParameterSet, which is itself a 3-tuple of
# (values, marks, id) and so is indistinguishable from a 3-axis parameter row.
# The type comes from the public factory rather than from `_pytest.mark`.
_PARAMETER_SET = type(pytest.param(None))


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


def import_test_module(path: Path, root: Path, arch: str) -> ModuleType:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Replicate conftest's LLK_HOME default for modules that do not import it.
    os.environ.setdefault("LLK_HOME", str(root.parent.parent))
    # Importing a test module runs get_chip_architecture() at module load. Without
    # CHIP_ARCH it probes for a physical device (or simulator context) and fails on a
    # plain host. The explicit CLI selection must override the caller's environment
    # and any architecture cached by an earlier import in this process.
    os.environ["CHIP_ARCH"] = arch
    try:
        from helpers.chip_architecture import clear_chip_architecture_cache

        clear_chip_architecture_cache()
    except ImportError:
        pass
    try:
        from helpers.test_config import TestConfig

        if "CHIP_ARCH" in vars(TestConfig):
            TestConfig.setup_arch()
    except ImportError:
        pass
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
    for suffix in (*_ARCH_FILE_SUFFIXES, "_perf", "_test"):
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


@dataclass(frozen=True)
class SideLabels:
    """Display names and tags for the two sides of a comparison."""

    left: str
    right: str
    left_tag: str
    right_tag: str

    @property
    def width(self) -> int:
        return max(len(self.left), len(self.right))

    @property
    def left_padded(self) -> str:
        return self.left.ljust(self.width)

    @property
    def right_padded(self) -> str:
        return self.right.ljust(self.width)

    @classmethod
    def for_arches(cls, left_arch: str, right_arch: str) -> "SideLabels":
        return cls(
            left_arch,
            right_arch,
            _ARCH_TAGS.get(left_arch, "[L]"),
            _ARCH_TAGS.get(right_arch, "[R]"),
        )


FUNCTIONAL_PERF = SideLabels("functional", "perf", "[T]", "[P]")


def fmt_values(values: list[str], full: bool) -> str:
    if full or len(values) <= _VALUE_PREVIEW_LIMIT:
        return ", ".join(values) if values else "-"
    return (
        ", ".join(values[:_VALUE_PREVIEW_LIMIT])
        + f", ... (+{len(values) - _VALUE_PREVIEW_LIMIT} more)"
    )


def axis_value_relation(test_values: list[str], perf_values: list[str]) -> str:
    """Classify how functional and perf value sets relate."""
    if not test_values or not perf_values:
        # A declared axis with no values means the rows could not be read, so the
        # two sides must not be reported as identical (or as a deliberate subset).
        return "unreadable"
    set_t, set_p = set(test_values), set(perf_values)
    if set_t == set_p:
        return "identical"
    if set_p <= set_t:
        return "perf_subset"
    if set_t <= set_p:
        return "functional_subset"
    return "different"


def verdict(
    name: str,
    t: list[str],
    p: list[str],
    in_t: bool,
    in_p: bool,
    kind: str,
    labels: SideLabels,
) -> tuple[str, str]:
    """Classify one axis or parameter and render its headline."""
    if not in_t:
        return "diff", f"{labels.right_tag} {name}: {labels.right.upper()}-ONLY {kind}"
    if not in_p:
        return "diff", f"{labels.left_tag} {name}: {labels.left.upper()}-ONLY {kind}"
    relation = axis_value_relation(t, p)
    if relation == "identical":
        return "same", f"[=] {name}: identical ({len(t)} value(s))"
    if relation == "unreadable":
        return (
            "unreadable",
            f"[!] {name}: UNREADABLE - no values parsed "
            f"({labels.left}={len(t)}, {labels.right}={len(p)})",
        )
    if relation == "perf_subset":
        return (
            "diff",
            f"[~] {name}: {labels.right} subset of {labels.left} "
            f"({len(p)}/{len(t)} value(s))",
        )
    if relation == "functional_subset":
        return (
            "diff",
            f"[~] {name}: {labels.left} subset of {labels.right} "
            f"({len(t)}/{len(p)} value(s))",
        )
    return "diff", f"[x] {name}: DIFFERENT"


def compare(
    test_axes: dict,
    perf_axes: dict,
    ignored_axes: frozenset[str],
    composite_axes: set[str],
    full: bool,
    labels: SideLabels = FUNCTIONAL_PERF,
) -> None:
    """Report identical vs differing axes; for differing ones print both sweeps."""
    same, diff, ignored, unreadable = [], [], [], []
    buckets = {"same": same, "diff": diff, "unreadable": unreadable}
    left, right = labels.left_padded, labels.right_padded
    for axis in dict.fromkeys([*test_axes, *perf_axes]):
        in_t, in_p = axis in test_axes, axis in perf_axes
        t, p = test_axes.get(axis, []), perf_axes.get(axis, [])

        if axis in ignored_axes:
            ignored.append(axis)
            print(f"  [i] {axis}: ignored measurement axis")
            if in_t:
                print(f"        {left} : {fmt_values(t, full)}")
            print(f"        {right} : {fmt_values(p, full)}")
            continue

        bucket, headline = verdict(axis, t, p, in_t, in_p, "axis", labels)
        buckets[bucket].append(axis)
        print(f"  {headline}")
        if bucket == "same":
            if full:
                print(f"        values : {fmt_values(t, full)}")
        elif axis in composite_axes and not full:
            # Whole configuration tuples are unreadable; the parameter view below
            # carries the same information one parameter at a time.
            print("        values : split per parameter below (--full for tuples)")
        else:
            print(f"        {left} : {fmt_values(t, full)}")
            print(f"        {right} : {fmt_values(p, full)}")
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
    functional: Parameters,
    perf: Parameters,
    full: bool,
    labels: SideLabels = FUNCTIONAL_PERF,
) -> None:
    """Report the parameters that composite axes were built from, one at a time.

    Parameters carry the name of the component they came from rather than the name
    of the axis that bundled them, so two modules that pack the same configuration
    into differently named tuples still line up parameter by parameter. Axes that
    did not decompose are left out, since the axis view already reported them.
    """
    test_params, perf_params = functional.values, perf.values
    split = functional.from_composite | perf.from_composite
    names = [n for n in dict.fromkeys([*test_params, *perf_params]) if n in split]
    same, diff, unreadable = [], [], []
    buckets = {"same": same, "diff": diff, "unreadable": unreadable}
    left, right = labels.left_padded, labels.right_padded
    print(f"\n  Composite axes split into {len(names)} parameter(s):")
    for name in names:
        in_t, in_p = name in test_params, name in perf_params
        t, p = test_params.get(name, []), perf_params.get(name, [])
        bucket, headline = verdict(name, t, p, in_t, in_p, "parameter", labels)
        buckets[bucket].append(name)
        print(f"    {headline}")
        if bucket != "same" or full:
            print(f"          {left} : {fmt_values(t, full)}")
            print(f"          {right} : {fmt_values(p, full)}")
    print(
        f"\n  Parameter summary: {len(same)} identical parameter(s), "
        f"{len(diff)} differing."
    )
    if diff:
        print(f"    differing : {', '.join(diff)}")
    if unreadable:
        print(f"    UNREADABLE: {', '.join(unreadable)} (verdict withheld)")


def print_onesided_sweep(
    sweep: Sweep, side: str, tag: str, padded: str, full: bool
) -> None:
    """Print every axis of a sweep that has no counterpart on the other side."""
    ignored_axes = frozenset(axis for axis in MEASUREMENT_AXES if axis in sweep.axes)
    n = projected_variant_count(sweep.rows, ignored_axes)
    print(f"  variants: {side}={n}\n")
    for axis, values in sweep.axes.items():
        if axis in ignored_axes:
            print(f"  [i] {axis}: ignored measurement axis")
            print(f"        {padded} : {fmt_values(values, full)}")
            continue
        print(f"  {tag} {axis}: {side.upper()}-ONLY axis")
        print(f"        {padded} : {fmt_values(values, full)}")
    params = parameter_values(sweep, ignored_axes)
    names = [n for n in params.values if n in params.from_composite]
    if not names:
        return
    print(f"\n  Composite axes split into {len(names)} parameter(s):")
    for name in names:
        print(f"    {tag} {name}: {side.upper()}-ONLY parameter")
        print(f"          {padded} : {fmt_values(params.values[name], full)}")


def emit_onesided_function(
    pmarks: list, side: str, tag: str, padded: str, full: bool
) -> None:
    """Read and print one unmatched function's sweep, or why it could not be read."""
    try:
        sweep = axis_value_sets(pmarks)
    except Exception as exc:  # one odd sweep must not abort the whole run
        print(
            "  ! skipped: cannot read parametrize marks "
            f"({type(exc).__name__}: {exc})\n"
        )
        return
    if not sweep.rows:
        print(f"  ! empty sweep: {side}=0 row(s)\n")
        return
    print_onesided_sweep(sweep, side, tag, padded, full)
    print()


def write_parameter_csv(
    path: Path,
    test_params: dict[str, list[str]],
    perf_params: dict[str, list[str]],
    labels: SideLabels = FUNCTIONAL_PERF,
) -> None:
    """Write the full, untruncated parameter/value table for one function pair."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", labels.left, labels.right])
        for name in dict.fromkeys([*test_params, *perf_params]):
            t, p = test_params.get(name, []), perf_params.get(name, [])
            for value in dict.fromkeys([*t, *p]):
                writer.writerow([name, value, int(value in t), int(value in p)])


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


def perf_stem_key(stem: str) -> str:
    """Map ``perf_matmul_quasar`` / ``perf_matmul`` to the shared key ``matmul``."""
    name = stem[len("perf_") :] if stem.startswith("perf_") else stem
    for suffix in _ARCH_FILE_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def discover_perf_modules(directory: Path) -> dict[str, Path]:
    """Return ``perf_*.py`` in ``directory``, keyed by :func:`perf_stem_key`."""
    found: dict[str, Path] = {}
    for path in sorted(directory.glob("*.py")):
        if path.resolve() == _SELF or not path.stem.startswith("perf_"):
            continue
        found[perf_stem_key(path.stem)] = path
    return found


def infer_arch_from_dir(directory: Path) -> str | None:
    name = directory.resolve().name.lower()
    return name if name in _ARCH_CHOICES else None


def discover_perf_dir_pairs(
    left_dir: Path, right_dir: Path
) -> tuple[list[tuple[str, Path, Path]], list[Path], list[Path]]:
    """Pair perf_*.py modules in two folders by the name after ``perf_`` / arch suffix."""
    left = discover_perf_modules(left_dir)
    right = discover_perf_modules(right_dir)
    matched = [(key, left[key], right[key]) for key in left if key in right]
    matched.sort(key=lambda item: item[0])
    left_only = [left[key] for key in sorted(left) if key not in right]
    right_only = [right[key] for key in sorted(right) if key not in left]
    return matched, left_only, right_only


def report_unpaired_module(
    path: Path,
    root: Path,
    arch: str,
    full: bool,
    side: str,
    tag: str,
    padded: str,
    reason: str,
) -> bool:
    """Import one unpaired module and print each parametrized function's sweep."""
    print("#" * _BANNER_WIDTH)
    print(f"# {path.name}  ({reason})")
    print("#" * _BANNER_WIDTH)
    try:
        mod = import_test_module(path, root, arch)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        print(f"  ! skipped: failed to import ({type(exc).__name__}: {exc})\n")
        return False
    funcs = parametrized_functions(mod)
    if not funcs:
        print(f"  ! no parametrized functions found in {path.name}\n")
        return False
    for name, pmarks in funcs.items():
        print("=" * _BANNER_WIDTH)
        print(f"{side}: {mod.__name__}.{name}")
        print("=" * _BANNER_WIDTH)
        emit_onesided_function(pmarks, side, tag, padded, full)
    sys.stdout.flush()
    return True


def compare_pair(
    left: Path,
    right: Path,
    root: Path,
    left_arch: str,
    right_arch: str,
    full: bool,
    csv_dir: Path | None = None,
    labels: SideLabels = FUNCTIONAL_PERF,
) -> bool:
    """Import a module pair and print their axis comparison.

    Returns True if at least one function was reported (matched or one-sided).
    """
    print("#" * _BANNER_WIDTH)
    print(f"# {left.name}  vs  {right.name}")
    print("#" * _BANNER_WIDTH)
    try:
        left_mod = import_test_module(left, root, left_arch)
        right_mod = import_test_module(right, root, right_arch)
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
        print(f"  ! no parametrized functions found in {left.name}\n")
        return False
    if not right_funcs:
        print(f"  ! no parametrized functions found in {right.name}\n")
        return False

    reported = False
    left_l, right_l = labels.left_padded, labels.right_padded
    for tname, pname, match_method in pair_functions(left_funcs, right_funcs):
        print("=" * _BANNER_WIDTH)
        print(f"{left_l}: {left_mod.__name__}.{tname or '<none>'}")
        print(f"{right_l}: {right_mod.__name__}.{pname or '<none>'}")
        print("=" * _BANNER_WIDTH)
        if tname is None or pname is None:
            print("  (unmatched - no counterpart found)")
            if tname is not None:
                emit_onesided_function(
                    left_funcs[tname],
                    labels.left,
                    labels.left_tag,
                    labels.left_padded,
                    full,
                )
            else:
                emit_onesided_function(
                    right_funcs[pname],
                    labels.right,
                    labels.right_tag,
                    labels.right_padded,
                    full,
                )
            reported = True
            continue
        if match_method == "position":
            print("  ! paired by position because normalized function names differ")
        try:
            t_sweep = axis_value_sets(left_funcs[tname])
            p_sweep = axis_value_sets(right_funcs[pname])
        except Exception as exc:  # one odd sweep must not abort the whole run
            print(
                "  ! skipped: cannot read parametrize marks "
                f"({type(exc).__name__}: {exc})\n"
            )
            continue
        if not t_sweep.rows or not p_sweep.rows:
            print(
                "  ! empty sweep: "
                f"{labels.left}={len(t_sweep.rows)} row(s), "
                f"{labels.right}={len(p_sweep.rows)} row(s)\n"
            )
            continue
        ignored_axes = frozenset(
            axis
            for axis in MEASUREMENT_AXES
            if axis in p_sweep.axes or axis in t_sweep.axes
        )
        t_n = projected_variant_count(t_sweep.rows, ignored_axes)
        p_n = projected_variant_count(p_sweep.rows, ignored_axes)
        print(f"  variants: {labels.left}={t_n}, {labels.right}={p_n}\n")

        t_params = parameter_values(t_sweep, ignored_axes)
        p_params = parameter_values(p_sweep, ignored_axes)
        composite_axes = t_params.composite_axes | p_params.composite_axes
        compare(t_sweep.axes, p_sweep.axes, ignored_axes, composite_axes, full, labels)
        if composite_axes:
            compare_parameters(t_params, p_params, full, labels)
        if csv_dir is not None:
            target = csv_dir / f"{left.stem}.{tname}.csv"
            write_parameter_csv(target, t_params.values, p_params.values, labels)
            print(f"\n  parameter table written to {target}")
        print()
        reported = True
    sys.stdout.flush()
    return reported


def _truncation_note(full: bool) -> None:
    if not full:
        print(
            f"Value lists truncated to {_VALUE_PREVIEW_LIMIT} entries; "
            "pass --full to print every value."
        )


def dump_unpaired_modules(
    paths: list[Path],
    root: Path,
    arch: str,
    full: bool,
    labels: SideLabels,
    which: str,
    reason: str,
) -> int:
    if which == "left":
        side, tag, padded = labels.left, labels.left_tag, labels.left_padded
    else:
        side, tag, padded = labels.right, labels.right_tag, labels.right_padded
    dumped = 0
    for path in paths:
        dumped += int(
            report_unpaired_module(path, root, arch, full, side, tag, padded, reason)
        )
    return dumped


def sweep_and_report(
    matched: list[tuple[str, Path, Path]],
    left_only: list[Path],
    right_only: list[Path],
    root: Path,
    left_arch: str,
    right_arch: str,
    full: bool,
    csv_dir: Path | None,
    labels: SideLabels,
    left_only_reason: str,
    right_only_reason: str,
    matches_only: bool = False,
) -> int:
    results = [
        compare_pair(left, right, root, left_arch, right_arch, full, csv_dir, labels)
        for _key, left, right in matched
    ]
    n_pairs = len(results)
    n_pair_ok = sum(results)
    if matches_only:
        print("#" * _BANNER_WIDTH)
        print(f"Sweep complete: {n_pair_ok}/{n_pairs} file pair(s) produced a report.")
    else:
        unpaired_ok = dump_unpaired_modules(
            left_only, root, left_arch, full, labels, "left", left_only_reason
        )
        unpaired_ok += dump_unpaired_modules(
            right_only, root, right_arch, full, labels, "right", right_only_reason
        )
        n_unpaired = len(left_only) + len(right_only)
        print("#" * _BANNER_WIDTH)
        print(
            f"Sweep complete: {n_pair_ok}/{n_pairs} file pair(s) produced a report, "
            f"{unpaired_ok}/{n_unpaired} unpaired module(s) dumped."
        )
    _truncation_note(full)
    return 0 if (not matched or all(results)) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "functional",
        type=Path,
        nargs="?",
        help="functional test module (single-pair mode; requires `perf` too)",
    )
    ap.add_argument(
        "perf",
        type=Path,
        nargs="?",
        help="perf test module (single-pair mode)",
    )
    ap.add_argument(
        "--dir",
        type=Path,
        default=_HERE,
        help="folder to sweep for test_*/perf_* pairs (default: this script's folder)",
    )
    ap.add_argument(
        "--perf-dirs",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        type=Path,
        help="compare perf_*.py in two folders (e.g. . quasar for WH/BH vs Quasar)",
    )
    ap.add_argument(
        "--full", action="store_true", help="print full value lists (no truncation)"
    )
    ap.add_argument(
        "--matches-only",
        action="store_true",
        help="report matched pairs only; skip unpaired module dumps",
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
        help="CHIP_ARCH for functional/perf sweeps, and the left --perf-dirs folder",
    )
    ap.add_argument(
        "--left-arch",
        choices=_ARCH_CHOICES,
        default=None,
        help="CHIP_ARCH for the left --perf-dirs folder (default: --arch)",
    )
    ap.add_argument(
        "--right-arch",
        choices=_ARCH_CHOICES,
        default=None,
        help="CHIP_ARCH for the right --perf-dirs folder "
        "(default: the folder name if it is wormhole/blackhole/quasar)",
    )
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if args.perf_dirs and (args.functional or args.perf):
        ap.error("--perf-dirs cannot be combined with explicit functional/perf paths")

    # Perf-vs-perf: two folders of perf_*.py, each with its own architecture.
    if args.perf_dirs:
        left_dir, right_dir = (p.resolve() for p in args.perf_dirs)
        left_arch = args.left_arch or args.arch
        right_arch = args.right_arch or infer_arch_from_dir(right_dir)
        if right_arch is None:
            ap.error(
                "pass --right-arch, or name the right folder wormhole/blackhole/quasar"
            )
        matched, left_only, right_only = discover_perf_dir_pairs(left_dir, right_dir)
        labels = SideLabels.for_arches(left_arch, right_arch)
        print(
            f"Comparing perf_*.py in {left_dir} ({left_arch}) vs "
            f"{right_dir} ({right_arch})"
        )
        print(
            f"Matched {len(matched)} perf pair(s): "
            f"{', '.join(k for k, _, _ in matched) or '-'}"
        )
        if not args.matches_only:
            if left_only:
                print(
                    f"{left_arch} without a {right_arch} counterpart: "
                    f"{', '.join(p.name for p in left_only)}"
                )
            if right_only:
                print(
                    f"{right_arch} without a {left_arch} counterpart: "
                    f"{', '.join(p.name for p in right_only)}"
                )
        _truncation_note(args.full)
        print()
        if matched:
            sample = matched[0][1]
        elif left_only:
            sample = left_only[0]
        elif right_only:
            sample = right_only[0]
        else:
            return 1
        root = find_python_tests_root(sample)
        return sweep_and_report(
            matched,
            left_only,
            right_only,
            root,
            left_arch,
            right_arch,
            args.full,
            args.csv,
            labels,
            f"no {right_arch} counterpart",
            f"no {left_arch} counterpart",
            args.matches_only,
        )

    if bool(args.functional) ^ bool(args.perf):
        ap.error(
            "provide both `functional` and `perf` for single-pair mode, or neither to sweep"
        )

    # Single-pair mode: explicit functional + perf paths.
    if args.functional and args.perf:
        root = find_python_tests_root(args.functional)
        print(f"Resolving sweeps for CHIP_ARCH={args.arch}")
        return (
            0
            if compare_pair(
                args.functional,
                args.perf,
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
    if not args.matches_only:
        if tests_only:
            print(
                f"test_* without a perf_* counterpart: {', '.join(p.name for p in tests_only)}"
            )
        if perfs_only:
            print(
                f"perf_* without a test_* counterpart: {', '.join(p.name for p in perfs_only)}"
            )
    _truncation_note(args.full)
    print()

    if matched:
        sample = matched[0][1]
    elif tests_only:
        sample = tests_only[0]
    elif perfs_only:
        sample = perfs_only[0]
    else:
        return 1
    root = find_python_tests_root(sample)
    return sweep_and_report(
        matched,
        tests_only,
        perfs_only,
        root,
        args.arch,
        args.arch,
        args.full,
        args.csv,
        FUNCTIONAL_PERF,
        "no perf counterpart",
        "no test counterpart",
        args.matches_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
