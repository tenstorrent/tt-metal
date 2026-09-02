# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


class PerfSchemaError(AssertionError):
    """Raised when a perf report's columns aren't a valid, unique schema: either a
    CSV that mixes more than one column schema, or two columns sharing a header
    name (which pandas would mangle into a ``<name>.1`` phantom). Fail loud so the
    bad CSV never ships."""


MARKER = "marker"

FORMAT_HEADERS = (
    "formats.input_A",
    "formats.input_B",
    "formats.register_A",
    "formats.register_B",
    "formats.output",
    "formats.sfpu_src",
    "formats.sfpu_dst",
)
FLAG_HEADERS = ("unpack_to_dest", "dest_acc")

# All fixed (non-parameter) sweep columns the pipeline always emits.
FIXED_SWEEP_HEADERS = FORMAT_HEADERS + FLAG_HEADERS

LOOP_FACTOR_COLUMN = "loop_factor"
TILE_CNT_COLUMN = "tile_cnt"
TEST_NAME_COLUMN = "test_name"

MEAN = "mean"
STD = "std"
TEXT_SIZE_PREFIX = "TEXT_SIZE("


def stat_prefix(kind: str) -> str:
    """Prefix of a stat column, e.g. stat_prefix("mean") -> "mean(" (for startswith)."""
    return f"{kind}("


def stat_column(base: str, kind: str) -> str:
    """Stat column, e.g. stat_column("L1_TO_L1", "mean") -> "mean(L1_TO_L1)"."""
    return f"{stat_prefix(kind)}{base})"


def metric_column(run_type_name: str, base: str) -> str:
    """Run-type-prefixed column, e.g. metric_column("L1_TO_L1", "x") -> "L1_TO_L1_x"."""
    return f"{run_type_name}_{base}"


def text_size_column(run_type_name: str) -> str:
    """Kernel code-size column, e.g. text_size_column("L1_TO_L1") -> "TEXT_SIZE(L1_TO_L1)"."""
    return f"{TEXT_SIZE_PREFIX}{run_type_name})"


def counter_base(bank: str, counter: str) -> str:
    """Raw HW-counter base name, e.g. counter_base("FPU", "FPU_COUNTER") -> "FPU.FPU_COUNTER"."""
    return f"{bank}.{counter}"


def cycles_of(base: str) -> str:
    """Cycles variant of a counter base, e.g. cycles_of("FPU.FPU_COUNTER") -> "FPU.FPU_COUNTER.cycles"."""
    return f"{base}.cycles"


def find_duplicate_columns(columns) -> list:
    seen, dupes = set(), []
    for c in columns:
        if c in seen and c not in dupes:
            dupes.append(c)
        seen.add(c)
    return dupes


def assert_unique_columns(columns, context: str = "") -> None:
    dupes = find_duplicate_columns(columns)
    if dupes:
        raise PerfSchemaError(
            f"Perf report has duplicate column header(s) {dupes} in "
            f"{context or 'report'}. Two parameters resolve to the same column "
            f"name (they share a dataclass field name), or the same parameter "
            f"was passed twice. Rename one field so every header is unique."
        )


# Golden CSV-header catalog. Hand-maintained: a header changes only when someone
# edits a set below, so a rename shows up as a reviewed diff. When a test adds or
# renames a header, update the matching set in the same PR.

# Run-type names — mirror helpers/llk_params.py::PerfRunType.
RUN_TYPE_NAMES = frozenset(
    {
        "L1_TO_L1",
        "UNPACK_ISOLATE",
        "MATH_ISOLATE",
        "PACK_ISOLATE",
        "L1_CONGESTION",
        "SFPU_ISOLATE",
    }
)

# Stat kinds a timing/counter column may carry.
STAT_KINDS = (MEAN, STD)

# Non-sweep key columns present in a report.
KEY_COLUMNS = (MARKER, TEST_NAME_COLUMN, LOOP_FACTOR_COLUMN, TILE_CNT_COLUMN)


def _load_perf_metrics_common():
    """Import the shared metric module; fall back to the file in the surrounding tt-metal checkout
    when the tracy package is not installed (the LLK CI container)."""
    try:
        from tracy import perf_metrics_common

        return perf_metrics_common
    except ImportError:
        pass
    import importlib.util
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tools" / "tracy" / "perf_metrics_common.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "perf_metrics_common", candidate
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError(
        "Could not locate tools/tracy/perf_metrics_common.py above this file"
    )


PERF_METRICS_COMMON = _load_perf_metrics_common()

# Derived metric base names: exactly the keys compute_metrics emits (and _exportable() keeps),
# taken from the shared module so this schema cannot drift from the formulas.
METRIC_BASES = frozenset(PERF_METRICS_COMMON.METRIC_LABELS)
