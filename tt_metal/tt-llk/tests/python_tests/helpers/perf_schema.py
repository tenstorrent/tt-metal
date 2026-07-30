# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


class PerfSchemaError(AssertionError):
    """
    Raised when a perf report's columns are not a valid, unique schema.

    Two failure modes share this error: a single CSV that accumulates more than
    one column schema (ragged, NaN-filled rows), and a report with two columns
    that carry the same header name (a duplicate that would be silently mangled
    into a ``<name>.1`` phantom). Both are fail-loud so the bad CSV never ships.
    """


MARKER = "marker"

FORMAT_HEADERS = (
    "formats.input_A",
    "formats.input_B",
    "formats.register_A",
    "formats.register_B",
    "formats.output",
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


# Golden CSV-header catalog
#
# This catalog is HAND-MAINTAINED on purpose: a header changes ONLY when someone
# edits a set below, so a rename becomes a reviewed diff instead of a silent
# drift. When a test legitimately adds or renames a header, update the matching
# set below in the SAME pull request.

# Run-type names — mirror helpers/llk_params.py::PerfRunType.
RUN_TYPE_NAMES = frozenset(
    {"L1_TO_L1", "UNPACK_ISOLATE", "MATH_ISOLATE", "PACK_ISOLATE", "L1_CONGESTION"}
)

# Stat kinds a timing/counter column may carry.
STAT_KINDS = (MEAN, STD)

# Non-sweep key columns present in a report.
KEY_COLUMNS = (MARKER, TEST_NAME_COLUMN, LOOP_FACTOR_COLUMN, TILE_CNT_COLUMN)

# Derived efficiency metric base names — mirror the ``*_pct`` keys that
# helpers/metrics.py::compute_metrics exports (the only keys _exportable() keeps).
METRIC_BASES = frozenset(
    {
        "fpu_utilization_pct",
        "compute_utilization_pct",
        "unpack_thread_stall_pct",
        "math_thread_stall_pct",
        "pack_thread_stall_pct",
        "math_sem_wait_pct",
        "pack_sem_wait_pct",
        "unpack0_write_eff_pct",
        "unpack1_write_eff_pct",
        "unpack_write_eff_pct",
        "unpack_to_math_flow0_pct",
        "unpack_to_math_flow1_pct",
        "unpack_to_math_flow_pct",
        "pack_utilization_pct",
        "pack_dest_eff_pct",
        "fidelity_stall_pct",
        "math_src_stall_pct",
    }
)
