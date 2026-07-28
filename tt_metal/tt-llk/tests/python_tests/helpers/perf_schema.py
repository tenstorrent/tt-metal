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

FORMAT_HEADERS = ("formats.input_A", "formats.input_B", "formats.output")
FLAG_HEADERS = ("unpack_to_dest", "dest_acc")

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
    """Timing stat column, e.g. stat_column("L1_TO_L1", "mean") -> "mean(L1_TO_L1)"."""
    return f"{stat_prefix(kind)}{base})"


def metric_column(run_type_name: str, base: str) -> str:
    """Run-type-prefixed column, e.g. metric_column("L1_TO_L1", "x") -> "L1_TO_L1_x"."""
    return f"{run_type_name}_{base}"


def text_size_column(run_type_name: str) -> str:
    """Kernel code-size column, e.g. -> "TEXT_SIZE(L1_TO_L1)"."""
    return f"{TEXT_SIZE_PREFIX}{run_type_name})"


def counter_base(bank: str, counter: str) -> str:
    """Raw HW-counter base name, e.g. counter_base("FPU", "FPU_COUNTER") -> "FPU.FPU_COUNTER"."""
    return f"{bank}.{counter}"


def cycles_of(base: str) -> str:
    """Cycles variant of a counter base name, e.g. -> "FPU.FPU_COUNTER.cycles"."""
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
