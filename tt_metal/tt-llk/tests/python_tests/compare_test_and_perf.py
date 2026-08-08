# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare the @parametrize sweeps of a functional test module vs a perf module.

For each matched pair it maps parametrized functions in the functional module to
counterparts in the perf module and, axis by axis, reports which parameters are
identical and which differ. For a differing axis it prints both value lists.
When one sweep is a subset of the other it labels that relationship (perf subset
of functional, or the reverse); otherwise it reports a true mismatch. Perf-only
``iterations`` and ``loop_factor`` axes are reported but excluded from coverage
comparisons because they are measurement controls, not coverage dimensions.

This is an axis-level comparison: it compares the union of values observed on
each axis, not the dependency or combination structure between axes.

By default it sweeps a folder (its own folder, i.e. ``python_tests``): it
collects every ``test_*.py`` and ``perf_*.py`` module, pairs them by the part of
the name after the prefix (so ``test_matmul.py`` pairs with ``perf_matmul.py``),
and runs the comparison over each matched pair. Use ``--dir`` to sweep another
folder that contains matched files, or pass two explicit paths for one pair.

This is a standalone diagnostic script, not a pytest test: it introspects the
``parametrize`` mark left on each function by the custom ``@parametrize``
decorator and does not invoke pytest. Importing test modules can execute their
normal module and ``conftest`` imports, so run it in the usual LLK environment.
It is not named ``test_*.py`` and pytest does not collect it.

Usage (run from the python_tests folder):
    python compare_test_and_perf.py                         # sweep this folder
    python compare_test_and_perf.py --full                  # no value-list truncation
    python compare_test_and_perf.py --arch quasar            # resolve Quasar sweeps
    python compare_test_and_perf.py <functional.py> <perf.py>   # single explicit pair
"""
from __future__ import annotations

import argparse
import enum
import importlib
import os
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

_HERE = Path(__file__).resolve().parent
_SELF = Path(__file__).resolve()
_BANNER_WIDTH = 88
_VALUE_PREVIEW_LIMIT = 8
_ARCH_CHOICES = ("wormhole", "blackhole", "quasar")


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
    # plain host. The explicit CLI selection must override the caller's environment.
    os.environ["CHIP_ARCH"] = arch
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
    if isinstance(v, tuple) and len(v) == n:
        return v
    return (v,) if n == 1 else tuple(v)


def axis_value_sets(
    pmarks: list,
) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    """Return per-axis values and canonicalized parameter rows."""
    axis_names: list[str] = []
    rows: list[tuple] | None = None
    for mark in pmarks:
        names = [n.strip() for n in mark.args[0].split(",")]
        mark_rows = [as_tuple(v, len(names)) for v in mark.args[1]]
        if rows is None:
            axis_names, rows = names, mark_rows
        else:  # stacked @parametrize marks -> cartesian product
            axis_names += names
            rows = [a + b for a in rows for b in mark_rows]
    rows = rows or []

    per_axis: dict[str, list[str]] = {n: [] for n in axis_names}
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
        canonical_rows.append(canonical_row)
    return per_axis, canonical_rows


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
# Pairing + comparison.                                                       #
# --------------------------------------------------------------------------- #
def normalize(name: str) -> str:
    normalized = name
    for pre in ("test_perf_", "perf_test_", "test_", "perf_"):
        if normalized.startswith(pre):
            normalized = normalized[len(pre) :]
            break
    for suffix in ("_perf", "_test"):
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


# Perf-only measurement controls are shown but excluded from coverage verdicts.
PERF_MEASUREMENT_AXES = frozenset({"iterations", "loop_factor"})


def fmt_values(values: list[str], full: bool) -> str:
    if full or len(values) <= _VALUE_PREVIEW_LIMIT:
        return ", ".join(values) if values else "-"
    return (
        ", ".join(values[:_VALUE_PREVIEW_LIMIT])
        + f", ... (+{len(values) - _VALUE_PREVIEW_LIMIT} more)"
    )


def axis_value_relation(test_values: list[str], perf_values: list[str]) -> str:
    """Classify how functional and perf value sets relate."""
    set_t, set_p = set(test_values), set(perf_values)
    if set_t == set_p:
        return "identical"
    if set_p <= set_t:
        return "perf_subset"
    if set_t <= set_p:
        return "functional_subset"
    return "different"


def compare(
    test_axes: dict,
    perf_axes: dict,
    ignored_perf_axes: frozenset[str],
    full: bool,
) -> None:
    """Report identical vs differing axes; for differing ones print both sweeps."""
    all_axes = list(dict.fromkeys([*test_axes, *perf_axes]))
    same, diff, ignored = [], [], []
    for axis in all_axes:
        in_t, in_p = axis in test_axes, axis in perf_axes
        t, p = test_axes.get(axis, []), perf_axes.get(axis, [])

        if axis in ignored_perf_axes:
            ignored.append(axis)
            print(f"  [i] {axis}: ignored perf measurement axis")
            print(f"        perf       : {fmt_values(p, full)}")
        elif not in_t:
            diff.append(axis)
            print(f"  [P] {axis}: PERF-ONLY axis")
            print(f"        functional : {fmt_values(t, full)}")
            print(f"        perf       : {fmt_values(p, full)}")
        elif not in_p:
            diff.append(axis)
            print(f"  [T] {axis}: FUNCTIONAL-ONLY axis")
            print(f"        functional : {fmt_values(t, full)}")
            print(f"        perf       : {fmt_values(p, full)}")
        else:
            relation = axis_value_relation(t, p)
            if relation == "identical":
                same.append(axis)
                print(f"  [=] {axis}: identical ({len(t)} value(s))")
                if full:
                    print(f"        values : {fmt_values(t, full)}")
            else:
                diff.append(axis)
                if relation == "perf_subset":
                    print(
                        f"  [~] {axis}: perf subset of functional "
                        f"({len(p)}/{len(t)} value(s))"
                    )
                elif relation == "functional_subset":
                    print(
                        f"  [~] {axis}: functional subset of perf "
                        f"({len(t)}/{len(p)} value(s))"
                    )
                else:
                    print(f"  [x] {axis}: DIFFERENT")
                print(f"        functional : {fmt_values(t, full)}")
                print(f"        perf       : {fmt_values(p, full)}")
    print(f"\n  Summary: {len(same)} identical axis/axes, {len(diff)} differing.")
    if same:
        print(f"    identical : {', '.join(same)}")
    if diff:
        print(f"    differing : {', '.join(diff)}")
    if ignored:
        print(f"    ignored   : {', '.join(ignored)} (perf measurement controls)")


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


def compare_pair(
    functional: Path, perf: Path, root: Path, arch: str, full: bool
) -> bool:
    """Import a functional/perf module pair and print their axis comparison.

    Returns True if at least one function pair was compared, False otherwise.
    """
    print("#" * _BANNER_WIDTH)
    print(f"# {functional.name}  vs  {perf.name}")
    print("#" * _BANNER_WIDTH)
    try:
        test_mod = import_test_module(functional, root, arch)
        perf_mod = import_test_module(perf, root, arch)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # keep sweeping even if a module aborts on import
        # BaseException (not just Exception) so a module that calls sys.exit /
        # pytest.exit at import time is skipped instead of killing the whole sweep.
        print(f"  ! skipped: failed to import ({type(exc).__name__}: {exc})\n")
        return False

    test_funcs = parametrized_functions(test_mod)
    perf_funcs = parametrized_functions(perf_mod)
    if not test_funcs:
        print(f"  ! no parametrized functions found in {functional.name}\n")
        return False
    if not perf_funcs:
        print(f"  ! no parametrized functions found in {perf.name}\n")
        return False

    compared = False
    for tname, pname, match_method in pair_functions(test_funcs, perf_funcs):
        print("=" * _BANNER_WIDTH)
        print(f"functional: {test_mod.__name__}.{tname or '<none>'}")
        print(f"perf      : {perf_mod.__name__}.{pname or '<none>'}")
        print("=" * _BANNER_WIDTH)
        if tname is None or pname is None:
            print("  (unmatched - no counterpart found)\n")
            continue
        if match_method == "position":
            print("  ! paired by position because normalized function names differ")
        t_axes, t_rows = axis_value_sets(test_funcs[tname])
        p_axes, p_rows = axis_value_sets(perf_funcs[pname])
        if not t_rows or not p_rows:
            print(
                "  ! empty sweep: "
                f"functional={len(t_rows)} row(s), perf={len(p_rows)} row(s)\n"
            )
            continue
        ignored_perf_axes = frozenset(
            axis
            for axis in PERF_MEASUREMENT_AXES
            if axis in p_axes and axis not in t_axes
        )
        t_n = projected_variant_count(t_rows, ignored_perf_axes)
        p_n = projected_variant_count(p_rows, ignored_perf_axes)
        print(f"  variants: functional={t_n}, perf={p_n}\n")
        compare(t_axes, p_axes, ignored_perf_axes, full)
        print()
        compared = True
    return compared


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
        "--full", action="store_true", help="print full value lists (no truncation)"
    )
    ap.add_argument(
        "--arch",
        choices=_ARCH_CHOICES,
        default="wormhole",
        help="CHIP_ARCH used to resolve the sweeps (default: wormhole)",
    )
    args = ap.parse_args()

    if bool(args.functional) ^ bool(args.perf):
        ap.error(
            "provide both `functional` and `perf` for single-pair mode, or neither to sweep"
        )

    # Single-pair mode: explicit functional + perf paths.
    if args.functional and args.perf:
        root = find_python_tests_root(args.functional)
        return (
            0
            if compare_pair(args.functional, args.perf, root, args.arch, args.full)
            else 1
        )

    # Sweep mode: pair every test_*/perf_* in the folder by common name.
    directory = args.dir.resolve()
    matched, tests_only, perfs_only = discover_pairs(directory)

    print(f"Sweeping {directory}")
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
    results = []
    for _key, functional, perf in matched:
        results.append(compare_pair(functional, perf, root, args.arch, args.full))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
