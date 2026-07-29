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


# Golden CSV-header catalog
#
# This catalog is HAND-MAINTAINED on purpose: a header changes ONLY when someone edits a set
# below, so a rename becomes a reviewed diff instead of a silent
# drift.When a test legitimately adds or renames a header, update the matching set below in the
# SAME pull request and, for a rename whose old name is fully retired, add a
# ``HEADER_ALIASES`` entry so old baselines still compare.

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

# Sweep-parameter field names — one per TemplateParameter/RuntimeParameter field
# that becomes a CSV header. Every name must be unique across all parameter
# classes (a duplicate would collide two columns; see assert_unique_columns).
GOLDEN_SWEEP_PARAMS = frozenset(
    {
        "acc_to_dest",
        "add_top_row",
        "approx_mode",
        "bcast_dim",
        "binop_mathop",
        "block_ct_dim",
        "block_rt_dim",
        "broadcast_type",
        "c_dimm",
        "clamp_negative",
        "configure_test_run_idx",
        "count",
        "ct_dim",
        "data_copy_type",
        "data_format",
        "dest_sync",
        "disable_src_zero_flag",
        "do_restore",
        "dst_index",
        "dst_tile_idx",
        "enable_2x_format",
        "enable_direct_indexing",
        "face_c_dim",
        "face_r_dim",
        "fast_mode",
        "full_ct_dim",
        "full_rt_dim",
        "host_is_stream_consumer",
        "host_is_stream_producer",
        "implied_math_format",
        "in0_c_dim",
        "in0_face_c_dim",
        "in0_face_r_dim",
        "in0_r_dim",
        "in1_c_dim",
        "in1_face_c_dim",
        "in1_face_r_dim",
        "in1_r_dim",
        "input_format",
        "input_num_blocks",
        "input_num_tiles_in_block",
        "input_tile_cnt",
        "int_op",
        "is_max_op",
        "is_reduce_to_one",
        "iterations",
        "k_dimm",
        "l1_acc",
        "loop_factor",
        "math_fidelity",
        "math_transpose_faces",
        "mathop",
        "narrow_tile",
        "num_blocks",
        "num_faces",
        "num_faces_A",
        "num_faces_B",
        "num_faces_c_dim_A",
        "num_faces_c_dim_B",
        "num_faces_r_dim_A",
        "num_faces_r_dim_B",
        "num_rows_to_pack",
        "num_tiles_in_block",
        "offset",
        "op",
        "output_format",
        "output_num_blocks",
        "output_num_tiles_in_block",
        "output_tile_cnt",
        "partial_a",
        "partial_b",
        "partial_face_math",
        "partial_face_pack",
        "perf_run_type",
        "pool_type",
        "r_dimm",
        "reduce_pool_type",
        "relu_config",
        "reuse_dest_type",
        "src0_tile_idx",
        "src1_tile_idx",
        "srca_reuse_count",
        "stable_sort",
        "stochastic_rounding",
        "ternary_mathop",
        "ternary_scalar_bits",
        "throttle_level",
        "tile_cnt",
        "tile_index",
        "tilize",
        "to_from_int8",
        "topk_k",
        "topk_matrix_width",
        "topk_sort_direction",
        "topk_stable_sort",
        "unary_extra",
        "unpack_transpose_faces",
        "unpack_transpose_within_face",
        "unpacker_engine_sel",
        "value_bits",
        "vector_mode",
        "via_reconfig",
        "victim_face_r_dim",
        "victim_num_faces",
    }
)


# Old->new header aliases
#
# Bridge a header rename so the compare feature can still line up a baseline CSV
# written by OLDER code against a report written by CURRENT code: on read, a
# stale ``old`` column is renamed to its current ``new`` name before the join.
#
#
# NOTE — the 2026 disambiguation renames (op→int_op, mathop→ternary_mathop /
# binop_mathop, value_bits→ternary_scalar_bits, tile_cnt→input_tile_cnt /
# output_tile_cnt) are deliberately NOT listed here: each old name is STILL a
# live header owned by another class (SFPU_BINARY_OP.op, MATH_OP.mathop,
# SFPU_UNARY_SCALAR.value_bits, TILE_COUNT.tile_cnt). They are not globally
# aliasable; bridging them needs per-test scoping (a separate follow-up).
HEADER_ALIASES: dict = {}


def apply_header_aliases(columns) -> list:
    """Rename any stale header to its current name (identity for unknown names)."""
    return [HEADER_ALIASES.get(c, c) for c in columns]
