# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare the @parametrize coverage of the same perf modules on two git commits.

For each requested ``perf_*.py`` it extracts that commit's ``python_tests`` tree,
imports the module the same way ``compare_test_and_perf.py`` does, and reads the
``parametrize`` mark left by ``@parametrize``. Measurement axes (``iterations``,
``loop_factor``, ``run_types``, ``is_perf``) are dropped, composite axes are
flattened into named parameters, and each remaining parameter combination is one
coverage variant.

The report is combination-level, not just the union of values on each axis:

* **kept**   — variants present on both commits (coverage that was there and remains)
* **lost**   — variants on commit A that are gone on commit B
* **gained** — variants on commit B that were not on commit A (new coverage)

When a later commit adds a parametrize axis that used to be an implicit kernel
default (``DestSync.Half``, ``num_blocks=1``), that default is filled on the
older side so Half/1-block rows can match instead of looking like a total
rewrite. Pass ``--no-defaults`` to disable that. A parameter-value summary is
printed as well, so a dest-sync or format that disappeared is visible even
when the variant list is long.

Anonymous ``[M, K]`` / ``[K, N]`` dimension lists flatten as ``matrix_a`` /
``matrix_b``.

This is a standalone diagnostic script, not a pytest test. Importing test
modules can execute their normal ``helpers`` / ``conftest`` imports, so run it
in the usual LLK environment. It does not check out the branch; each commit is
read with ``git archive`` into a temp directory.

Default modules are ``perf_matmul.py`` and ``perf_math_matmul.py``.

Usage (run from the python_tests folder):
    python compare_perf_commits.py <commit_a> <commit_b>
    python compare_perf_commits.py origin/main HEAD --full
    python compare_perf_commits.py abc123 def456 --csv reports/
    python compare_perf_commits.py A B --tests perf_matmul.py
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tarfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

from compare_test_and_perf import (
    _ARCH_CHOICES,
    _BANNER_WIDTH,
    _VALUE_PREVIEW_LIMIT,
    MEASUREMENT_AXES,
    as_tuple,
    axis_names_of,
    canon,
    find_python_tests_root,
    flatten_axis_value,
    fmt_values,
    import_test_module,
    parametrized_functions,
)

_SELF = Path(__file__).resolve()
_HERE = _SELF.parent
_DEFAULT_TESTS = ("perf_matmul.py", "perf_math_matmul.py")
_TESTS_ARCHIVE_PATH = "tt_metal/tt-llk/tests"
_KEEP_MODULES = frozenset(
    {
        "compare_test_and_perf",
        "compare_perf_commits",
        "list_perf_params",
    }
)
# Axes that used to be implicit kernel defaults. Filled on the commit that
# does not declare them so variant identity survives a parametrize expansion.
_IMPLICIT_DEFAULTS = {
    "DestSync": "DestSync.Half",
    "num_blocks": "1",
}


# --------------------------------------------------------------------------- #
# Git: resolve commits and extract python_tests without checking out.          #
# --------------------------------------------------------------------------- #
def repo_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=_HERE, text=True
        ).strip()
    )


def resolve_commit(repo: Path, rev: str) -> tuple[str, str]:
    """Return (full sha, short description) for a revision."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{rev}^{{commit}}"],
        cwd=repo,
        text=True,
    ).strip()
    subject = subprocess.check_output(
        ["git", "log", "-1", "--format=%h %s", sha],
        cwd=repo,
        text=True,
    ).strip()
    return sha, subject


@contextmanager
def extracted_python_tests(repo: Path, commit: str) -> Iterator[Path]:
    """Unpack ``python_tests`` from ``commit`` into a temp dir and yield its root."""
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            commit,
            _TESTS_ARCHIVE_PATH,
        ],
        capture_output=True,
    )
    if archive.returncode != 0:
        err = (archive.stderr or archive.stdout).decode(errors="replace").strip()
        raise RuntimeError(f"git archive {commit} {_TESTS_ARCHIVE_PATH}: {err}")
    with TemporaryDirectory(prefix=f"llk-perf-{commit[:8]}-") as tmp:
        dest = Path(tmp)
        with tarfile.open(fileobj=BytesIO(archive.stdout), mode="r:") as tar:
            tar.extractall(dest)
        root = dest / _TESTS_ARCHIVE_PATH / "python_tests"
        if not root.is_dir():
            raise RuntimeError(
                f"{_TESTS_ARCHIVE_PATH}/python_tests missing at {commit}"
            )
        yield root


# --------------------------------------------------------------------------- #
# Import isolation: each commit has its own helpers.                           #
# --------------------------------------------------------------------------- #
def drop_llk_test_modules() -> None:
    """Forget imported test/helpers modules so the next commit is loaded cleanly."""
    for name in list(sys.modules):
        if name in _KEEP_MODULES or name.startswith("compare_"):
            continue
        if (
            name == "helpers"
            or name.startswith("helpers.")
            or name.startswith("perf_")
            or name.startswith("test_")
            or name.startswith("quasar")
            or name.startswith("fuser")
        ):
            del sys.modules[name]


def flatten_row(
    axis_names: list[str], row: tuple, ignored_axes: frozenset[str]
) -> dict[str, str]:
    """Flatten one parametrize combination into named canonical parameters."""
    leaves: list[tuple[str, Any]] = []
    for name, val in zip(axis_names, row):
        if name in ignored_axes:
            continue
        leaves.extend(flatten_axis_value(val, name))
    repeated = {
        name for name, count in Counter(n for n, _ in leaves).items() if count > 1
    }
    ordinal: Counter = Counter()
    flattened: dict[str, str] = {}
    for name, leaf in leaves:
        if name in repeated:
            key = f"{name}#{ordinal[name]}"
            ordinal[name] += 1
        else:
            key = name
        flattened[key] = canon(leaf)
    # Two scalar lists from a matmul (matrix_a, matrix_b) tuple land as list#0/1.
    if "list#0" in flattened and "list#1" in flattened:
        flattened["matrix_a"] = flattened.pop("list#0")
        flattened["matrix_b"] = flattened.pop("list#1")
    return flattened


@dataclass(frozen=True)
class Coverage:
    """One function's coverage at one commit."""

    rows: dict[tuple[tuple[str, str], ...], dict[str, str]]
    parameters: dict[str, list[str]]  # parameter -> ordered unique values


def empty_coverage() -> Coverage:
    return Coverage(rows={}, parameters={})


def _apply_fill(coverage: Coverage, fill: dict[str, str]) -> Coverage:
    if not fill:
        return coverage
    parameters = dict(coverage.parameters)
    for name, value in fill.items():
        parameters.setdefault(name, [value])
    rows: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for row in coverage.rows.values():
        filled = {**row, **fill}
        rows[tuple(sorted(filled.items()))] = filled
    return Coverage(rows=rows, parameters=parameters)


def align_coverages(
    before: Coverage, after: Coverage, use_defaults: bool
) -> tuple[Coverage, Coverage, list[str]]:
    """Give both sides the same parameter names so added axes can still match."""
    names = list(dict.fromkeys([*before.parameters, *after.parameters]))
    fill_a: dict[str, str] = {}
    fill_b: dict[str, str] = {}
    notes: list[str] = []
    for name in names:
        in_a = name in before.parameters
        in_b = name in after.parameters
        if in_a == in_b:
            continue
        default = _IMPLICIT_DEFAULTS.get(name) if use_defaults else None
        if default is None:
            absent = "<absent>"
            if in_a:
                fill_b[name] = absent
                notes.append(f"{name}: after-only (no implicit default)")
            else:
                fill_a[name] = absent
                notes.append(f"{name}: before-only (no implicit default)")
            continue
        if in_a:
            fill_b[name] = default
            notes.append(f"{name}: filled on after with {default}")
        else:
            fill_a[name] = default
            notes.append(
                f"{name}: filled on before with {default} "
                "(implicit default before this axis was parametrized)"
            )
    return _apply_fill(before, fill_a), _apply_fill(after, fill_b), notes


def collect_coverage(pmarks: list) -> Coverage:
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
    ignored = frozenset(axis for axis in MEASUREMENT_AXES if axis in axis_names)

    coverage_rows: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    per_param: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for row in rows:
        flat = flatten_row(axis_names, row, ignored)
        coverage_rows[tuple(sorted(flat.items()))] = flat
        for name, value in flat.items():
            if value not in seen.setdefault(name, set()):
                seen[name].add(value)
                per_param.setdefault(name, []).append(value)
    return Coverage(rows=coverage_rows, parameters=per_param)


def load_module_coverage(
    module_path: Path, root: Path, arch: str
) -> dict[str, Coverage]:
    drop_llk_test_modules()
    inserted = str(root) not in sys.path
    if inserted:
        sys.path.insert(0, str(root))
    try:
        module = import_test_module(module_path, root, arch)
        funcs = parametrized_functions(module)
        return {name: collect_coverage(pmarks) for name, pmarks in funcs.items()}
    finally:
        if inserted:
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass
        drop_llk_test_modules()


def load_commit_coverage(
    repo: Path, commit: str, modules: list[str], arch: str
) -> dict[str, dict[str, Coverage]]:
    """module stem -> function name -> Coverage."""
    out: dict[str, dict[str, Coverage]] = {}
    with extracted_python_tests(repo, commit) as root:
        find_python_tests_root(root)  # confirms helpers/ + pytest.ini
        for name in modules:
            path = root / name
            if not path.is_file():
                out[name] = {}
                continue
            try:
                out[name] = load_module_coverage(path, root, arch)
            except KeyboardInterrupt:
                raise
            except BaseException as exc:
                print(
                    f"  ! {name} at {commit[:12]}: failed to import "
                    f"({type(exc).__name__}: {exc})"
                )
                out[name] = {}
    return out


# --------------------------------------------------------------------------- #
# Report.                                                                     #
# --------------------------------------------------------------------------- #
def fmt_row(row: dict[str, str]) -> str:
    return ", ".join(f"{name}={value}" for name, value in row.items())


def print_row_block(title: str, rows: list[dict[str, str]], limit: int) -> None:
    print(f"\n  {title}: {len(rows)}")
    if not rows:
        return
    shown = rows if limit <= 0 else rows[:limit]
    for row in shown:
        print(f"    - {fmt_row(row)}")
    hidden = len(rows) - len(shown)
    if hidden:
        print(f"    ... (+{hidden} more; pass --full to print all)")


def compare_parameters(
    before: dict[str, list[str]],
    after: dict[str, list[str]],
    full: bool,
) -> None:
    names = list(dict.fromkeys([*before, *after]))
    print(f"\n  Parameter values ({len(names)} parameter(s)):")
    for name in names:
        a_vals, b_vals = before.get(name, []), after.get(name, [])
        set_a, set_b = set(a_vals), set(b_vals)
        kept = [
            v for v in dict.fromkeys([*a_vals, *b_vals]) if v in set_a and v in set_b
        ]
        lost = [v for v in a_vals if v not in set_b]
        gained = [v for v in b_vals if v not in set_a]
        if not a_vals:
            print(f"    [+] {name}: AFTER-ONLY ({len(gained)} value(s))")
            print(f"          gained : {fmt_values(gained, full)}")
        elif not b_vals:
            print(f"    [-] {name}: BEFORE-ONLY ({len(lost)} value(s))")
            print(f"          lost   : {fmt_values(lost, full)}")
        elif not lost and not gained:
            print(f"    [=] {name}: identical ({len(kept)} value(s))")
            if full:
                print(f"          values : {fmt_values(kept, full)}")
        else:
            print(
                f"    [~] {name}: kept={len(kept)} lost={len(lost)} gained={len(gained)}"
            )
            if lost:
                print(f"          lost   : {fmt_values(lost, full)}")
            if gained:
                print(f"          gained : {fmt_values(gained, full)}")


def write_variant_csv(
    path: Path,
    kept: list[dict[str, str]],
    lost: list[dict[str, str]],
    gained: list[dict[str, str]],
) -> None:
    columns = list(
        dict.fromkeys(name for row in (*kept, *lost, *gained) for name in row)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["status", *columns])
        for status, rows in (
            ("kept", kept),
            ("lost", lost),
            ("gained", gained),
        ):
            for row in rows:
                writer.writerow([status, *[row.get(col, "") for col in columns]])


def write_parameter_csv(
    path: Path,
    before: dict[str, list[str]],
    after: dict[str, list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", "before", "after"])
        for name in dict.fromkeys([*before, *after]):
            a_vals, b_vals = set(before.get(name, [])), set(after.get(name, []))
            for value in dict.fromkeys([*before.get(name, []), *after.get(name, [])]):
                writer.writerow(
                    [name, value, int(value in a_vals), int(value in b_vals)]
                )


def compare_function(
    module: str,
    func: str,
    before: Coverage,
    after: Coverage,
    full: bool,
    row_limit: int,
    csv_dir: Path | None,
    use_defaults: bool,
) -> tuple[int, int, int]:
    print("=" * _BANNER_WIDTH)
    print(f"{module} :: {func}")
    print("=" * _BANNER_WIDTH)

    before, after, notes = align_coverages(before, after, use_defaults)
    if notes:
        print("  alignment:")
        for note in notes:
            print(f"    - {note}")

    keys_a, keys_b = set(before.rows), set(after.rows)
    kept_keys = sorted(keys_a & keys_b)
    lost_keys = sorted(keys_a - keys_b)
    gained_keys = sorted(keys_b - keys_a)
    kept = [before.rows[k] for k in kept_keys]
    lost = [before.rows[k] for k in lost_keys]
    gained = [after.rows[k] for k in gained_keys]

    print(
        f"  coverage: before={len(before.rows)}  after={len(after.rows)}  "
        f"kept={len(kept)}  lost={len(lost)}  gained={len(gained)}"
    )
    compare_parameters(before.parameters, after.parameters, full)
    limit = 0 if full else row_limit
    print_row_block("Lost variants (before only)", lost, limit)
    print_row_block("Gained variants (after only)", gained, limit)
    if csv_dir is not None:
        stem = f"{Path(module).stem}.{func}"
        variants = csv_dir / f"{stem}.variants.csv"
        params = csv_dir / f"{stem}.parameters.csv"
        write_variant_csv(variants, kept, lost, gained)
        write_parameter_csv(params, before.parameters, after.parameters)
        print(f"\n  wrote {variants}")
        print(f"  wrote {params}")
    print()
    return len(kept), len(lost), len(gained)


def compare_module(
    name: str,
    before: dict[str, Coverage],
    after: dict[str, Coverage],
    full: bool,
    row_limit: int,
    csv_dir: Path | None,
    use_defaults: bool,
) -> tuple[int, int, int]:
    print("#" * _BANNER_WIDTH)
    print(f"# {name}")
    print("#" * _BANNER_WIDTH)
    if not before and not after:
        print("  ! module missing or failed to import on both commits\n")
        return 0, 0, 0
    if not before:
        print("  ! module missing on commit A; all after coverage is gained\n")
    if not after:
        print("  ! module missing on commit B; all before coverage is lost\n")

    kept = lost = gained = 0
    for func in dict.fromkeys([*before, *after]):
        k, l, g = compare_function(
            name,
            func,
            before.get(func, empty_coverage()),
            after.get(func, empty_coverage()),
            full,
            row_limit,
            csv_dir,
            use_defaults,
        )
        kept += k
        lost += l
        gained += g
    return kept, lost, gained


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("commit_a", help="before commit (coverage baseline)")
    ap.add_argument("commit_b", help="after commit (new coverage)")
    ap.add_argument(
        "--tests",
        nargs="+",
        default=list(_DEFAULT_TESTS),
        metavar="PERF.py",
        help="perf modules to compare (default: perf_matmul.py perf_math_matmul.py)",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="print every lost/gained variant and full parameter value lists",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=_VALUE_PREVIEW_LIMIT * 4,
        metavar="N",
        help="lost/gained variants to print per function without --full (default: 32)",
    )
    ap.add_argument(
        "--no-defaults",
        action="store_true",
        help="do not fill implicit DestSync.Half / num_blocks=1 on the side that lacks them",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        metavar="DIR",
        help="write per-function variant and parameter CSVs there",
    )
    env_arch = os.environ.get("CHIP_ARCH", "").lower()
    ap.add_argument(
        "--arch",
        choices=_ARCH_CHOICES,
        default=env_arch if env_arch in _ARCH_CHOICES else "wormhole",
        help="CHIP_ARCH used to resolve the sweeps (default: $CHIP_ARCH or wormhole)",
    )
    args = ap.parse_args()

    repo = repo_root()
    sha_a, desc_a = resolve_commit(repo, args.commit_a)
    sha_b, desc_b = resolve_commit(repo, args.commit_b)
    modules = [Path(name).name for name in args.tests]

    print(f"Comparing perf coverage for CHIP_ARCH={args.arch}")
    print(f"  before: {desc_a}")
    print(f"  after : {desc_b}")
    print(f"  tests : {', '.join(modules)}")
    print()

    print(f"Loading commit A ({sha_a[:12]})...")
    coverage_a = load_commit_coverage(repo, sha_a, modules, args.arch)
    print(f"Loading commit B ({sha_b[:12]})...")
    coverage_b = load_commit_coverage(repo, sha_b, modules, args.arch)
    print()

    totals = [0, 0, 0]
    for name in modules:
        kept, lost, gained = compare_module(
            name,
            coverage_a.get(name, {}),
            coverage_b.get(name, {}),
            args.full,
            args.max_rows,
            args.csv,
            use_defaults=not args.no_defaults,
        )
        totals[0] += kept
        totals[1] += lost
        totals[2] += gained

    print("#" * _BANNER_WIDTH)
    print(
        f"Total: kept={totals[0]}  lost={totals[1]}  gained={totals[2]}  "
        f"(before={totals[0] + totals[1]}, after={totals[0] + totals[2]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
