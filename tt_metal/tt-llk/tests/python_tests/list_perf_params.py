# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""List the @parametrize sweeps of every perf test module.

For each ``perf_*.py`` module it prints every parametrized function, the axes
that function sweeps, and the values on each axis. Composite axes (a whole
configuration packed into one tuple or dataclass) are also expanded one
parameter at a time, using the same flattening as ``compare_test_and_perf.py``.
Measurement control axes (``iterations``, ``loop_factor``, ``run_types``,
``is_perf``) are labelled as such and excluded from the variant count, because
they are measurement knobs and not coverage dimensions.

Unlike ``compare_test_and_perf.py`` this script does not look at functional
``test_*.py`` modules at all: every perf file is listed, including those with
no functional counterpart.

This is an axis-level listing: it reports the union of values observed on each
axis, not the dependency or combination structure between axes.

By default it sweeps a folder (its own folder, i.e. ``python_tests``). Use
``--dir`` to sweep another folder (for example ``quasar``), or pass one or more
explicit ``perf_*.py`` paths. ``--csv DIR`` writes the whole parameter/value
table, split and plain axes alike, per function.

This is a standalone diagnostic script, not a pytest test: it introspects the
``parametrize`` mark left on each function by the custom ``@parametrize``
decorator and does not invoke pytest. Importing test modules can execute their
normal module and ``conftest`` imports, so run it in the usual LLK environment.
It is not named ``test_*.py`` and pytest does not collect it.

The target architecture comes from ``CHIP_ARCH`` (``export CHIP_ARCH=quasar``)
and ``--arch`` overrides it; the resolved value is printed with the sweep
header.

Usage (run from the python_tests folder):
    python list_perf_params.py                         # sweep this folder
    python list_perf_params.py --full                  # no value-list truncation
    python list_perf_params.py --dir quasar            # sweep the Quasar perf modules
    python list_perf_params.py --csv reports/          # export parameter tables
    python list_perf_params.py perf_matmul.py          # one or more explicit modules
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from compare_test_and_perf import (
    MEASUREMENT_AXES,
    Parameters,
    axis_value_sets,
    find_python_tests_root,
    fmt_values,
    import_test_module,
    parameter_values,
    parametrized_functions,
    projected_variant_count,
)

_SELF = Path(__file__).resolve()
_HERE = _SELF.parent
_BANNER_WIDTH = 88
_ARCH_CHOICES = ("wormhole", "blackhole", "quasar")


def discover_perf_modules(directory: Path) -> list[Path]:
    """Return every ``perf_*.py`` in ``directory``, excluding this script."""
    modules = []
    for path in sorted(directory.glob("*.py")):
        if path.resolve() == _SELF:
            continue
        if path.stem.startswith("perf_"):
            modules.append(path)
    return modules


def write_parameter_csv(path: Path, params: dict[str, list[str]]) -> None:
    """Write the full, untruncated parameter/value table for one function."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        for name, values in params.items():
            for value in values:
                writer.writerow([name, value])


def list_axes(
    sweep_axes: dict,
    ignored_axes: frozenset[str],
    composite_axes: set[str],
    full: bool,
) -> None:
    """Print each axis and its value list."""
    ignored = []
    empty = []
    for axis, values in sweep_axes.items():
        if axis in ignored_axes:
            ignored.append(axis)
            print(f"  [i] {axis}: measurement axis ({len(values)} value(s))")
            print(f"        {fmt_values(values, full)}")
            continue
        if not values:
            empty.append(axis)
            print(f"  [!] {axis}: UNREADABLE - no values parsed")
            continue
        print(f"  {axis}: {len(values)} value(s)")
        if axis in composite_axes and not full:
            print("        values : split per parameter below (--full for tuples)")
        else:
            print(f"        {fmt_values(values, full)}")
    if ignored:
        print(f"\n  ignored measurement axes: {', '.join(ignored)}")
    if empty:
        print(f"  UNREADABLE: {', '.join(empty)}")


def list_parameters(params: Parameters, full: bool) -> None:
    """Print the parameters that composite axes were built from."""
    names = [n for n in params.values if n in params.from_composite]
    print(f"\n  Composite axes split into {len(names)} parameter(s):")
    for name in names:
        values = params.values[name]
        print(f"    {name}: {len(values)} value(s)")
        print(f"          {fmt_values(values, full)}")


def list_module(
    path: Path,
    root: Path,
    arch: str,
    full: bool,
    csv_dir: Path | None = None,
) -> bool:
    """Import one perf module and print its parametrize sweeps.

    Returns True if at least one function sweep was listed, False otherwise.
    """
    print("#" * _BANNER_WIDTH)
    print(f"# {path.name}")
    print("#" * _BANNER_WIDTH)
    try:
        module = import_test_module(path, root, arch)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # keep sweeping even if a module aborts on import
        # BaseException (not just Exception) so a module that calls sys.exit /
        # pytest.exit at import time is skipped instead of killing the whole sweep.
        print(f"  ! skipped: failed to import ({type(exc).__name__}: {exc})\n")
        return False

    funcs = parametrized_functions(module)
    if not funcs:
        print(f"  ! no parametrized functions found in {path.name}\n")
        return True

    listed = False
    for name, pmarks in funcs.items():
        print("=" * _BANNER_WIDTH)
        print(f"perf: {module.__name__}.{name}")
        print("=" * _BANNER_WIDTH)
        try:
            sweep = axis_value_sets(pmarks)
        except Exception as exc:  # one odd sweep must not abort the whole run
            print(
                "  ! skipped: cannot read parametrize marks "
                f"({type(exc).__name__}: {exc})\n"
            )
            continue
        if not sweep.rows:
            print(f"  ! empty sweep: {len(sweep.rows)} row(s)\n")
            continue
        ignored_axes = frozenset(
            axis for axis in MEASUREMENT_AXES if axis in sweep.axes
        )
        n_variants = projected_variant_count(sweep.rows, ignored_axes)
        print(f"  variants: {n_variants}\n")

        params = parameter_values(sweep, ignored_axes)
        list_axes(sweep.axes, ignored_axes, params.composite_axes, full)
        if params.composite_axes:
            list_parameters(params, full)
        if csv_dir is not None:
            target = csv_dir / f"{path.stem}.{name}.csv"
            write_parameter_csv(target, params.values)
            print(f"\n  parameter table written to {target}")
        print()
        listed = True
    return listed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "modules",
        type=Path,
        nargs="*",
        help="explicit perf module path(s); omit to sweep --dir",
    )
    ap.add_argument(
        "--dir",
        type=Path,
        default=_HERE,
        help="folder to sweep for perf_*.py modules (default: this script's folder)",
    )
    ap.add_argument(
        "--full", action="store_true", help="print full value lists (no truncation)"
    )
    ap.add_argument(
        "--csv",
        type=Path,
        metavar="DIR",
        help="also write the untruncated parameter/value table of each function there",
    )
    env_arch = os.environ.get("CHIP_ARCH", "").lower()
    ap.add_argument(
        "--arch",
        choices=_ARCH_CHOICES,
        default=env_arch if env_arch in _ARCH_CHOICES else "wormhole",
        help="CHIP_ARCH used to resolve the sweeps (default: $CHIP_ARCH or wormhole)",
    )
    args = ap.parse_args()

    if args.modules:
        modules = [path.resolve() for path in args.modules]
        root = find_python_tests_root(modules[0])
        print(f"Listing {len(modules)} module(s) for CHIP_ARCH={args.arch}\n")
    else:
        directory = args.dir.resolve()
        modules = discover_perf_modules(directory)
        print(f"Sweeping {directory} for CHIP_ARCH={args.arch}")
        print(
            f"Found {len(modules)} perf module(s): "
            f"{', '.join(p.stem[len('perf_'):] for p in modules) or '-'}"
        )
        print()
        if not modules:
            return 1
        root = find_python_tests_root(modules[0])

    results = [
        list_module(path, root, args.arch, args.full, args.csv) for path in modules
    ]
    listed = sum(1 for ok in results if ok)
    skipped = len(results) - listed
    print(
        f"Listed {listed} module(s)" + (f", skipped {skipped}" if skipped else "") + "."
    )
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
