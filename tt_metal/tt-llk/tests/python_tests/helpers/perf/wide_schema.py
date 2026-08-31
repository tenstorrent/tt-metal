# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared wide nullable schema for LLK performance reports (v1).

DB_SCHEMA is the one published table: every column, and the exact schema the
Parquet is written with and handed to the data team. Each column declares its
``origin`` — ``"test"`` (a perf test emits it) or ``"ci"`` (the publish layer
stamps it). Two views derive from that single source of truth:

  OUTPUT_SCHEMA  the origin=="test" columns — what a report is validated against.
  PROVENANCE     the origin=="ci" columns — run context (commit, arch, run id, ...).

One row per test config per run. Every column except ``marker`` and the mandatory
provenance keys is nullable: a test fills the columns it uses, the rest stay NULL.
New columns are added later as nullable, so v1 need not be complete.

Imports no device libraries, so it loads without hardware.
"""

from dataclasses import dataclass

from .schema import (
    MEAN,
    METRIC_BASES,
    RUN_TYPE_NAMES,
    STD,
    metric_column,
    stat_column,
)


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # int64 | float64 | bool | string
    nullable: bool
    category: str
    origin: str = "test"  # who fills the column: "test" | "ci"


# Timing columns are formula-driven: mean(<base>) for every run type, std(<base>)
# once a run type runs >=2 iterations. Enumerate the full {mean, std} x base grid
# so the schema is a superset of any run config, not just what one nightly sampled.
_TIMING_BASES = (
    "L1_TO_L1",
    "UNPACK_ISOLATE",
    "MATH_ISOLATE",
    "PACK_ISOLATE",
    "L1_CONGESTION[UNPACK]",
    "L1_CONGESTION[PACK]",
)
_TIMING_COLUMNS = [
    Column(stat_column(base, kind), "float64", True, "timing")
    for base in _TIMING_BASES
    for kind in (MEAN, STD)
]


# ── The one published table: every column, headers + provenance ─────────────
# This IS the table handed to the data team; the Parquet is written with it. Each
# column's `origin` says who fills it — "test" (default) or "ci".
# Counter metrics are deliberately not here, see DROPPED_COLUMNS below. Quasar has
# its own published table in wide_schema_quasar.py — do not mix Quasar columns in.
DB_SCHEMA = [
    Column("marker", "string", False, "identity"),
    # formats
    Column("formats.input_A", "string", True, "formats"),
    Column("formats.input_B", "string", True, "formats"),
    Column("formats.output", "string", True, "formats"),
    Column("formats.register_A", "string", True, "formats"),
    Column("formats.register_B", "string", True, "formats"),
    Column("formats.sfpu_src", "string", True, "formats"),
    Column("formats.sfpu_dst", "string", True, "formats"),
    # flags
    Column("dest_acc", "string", True, "flags"),
    Column("speed_of_light", "bool", True, "flags"),
    Column("unpack_to_dest", "string", True, "flags"),
    # key
    Column("loop_factor", "int64", True, "key"),
    Column("tile_cnt", "int64", True, "key"),
    # configuration
    Column("alpha_bits", "int64", True, "configuration"),
    Column("approx_mode", "string", True, "configuration"),
    Column("beta_bits", "int64", True, "configuration"),
    Column("binop_mathop", "string", True, "configuration"),
    Column("block_ct_dim", "int64", True, "configuration"),
    Column("block_rt_dim", "int64", True, "configuration"),
    Column("broadcast_type", "string", True, "configuration"),
    Column("c_dimm", "int64", True, "configuration"),
    Column("clamp_negative", "bool", True, "configuration"),
    Column("ct_dim", "int64", True, "configuration"),
    Column("dest_sync", "string", True, "configuration"),
    Column("dst_index", "int64", True, "configuration"),
    Column("fast_mode", "string", True, "configuration"),
    Column("full_ct_dim", "int64", True, "configuration"),
    Column("full_rt_dim", "int64", True, "configuration"),
    Column("fused_sort", "string", True, "configuration"),
    Column("in0_c_dim", "int64", True, "configuration"),
    Column("in0_r_dim", "int64", True, "configuration"),
    Column("in1_c_dim", "int64", True, "configuration"),
    Column("in1_r_dim", "int64", True, "configuration"),
    Column("input_format", "string", True, "configuration"),
    Column("input_num_blocks", "int64", True, "configuration"),
    Column("input_num_tiles_in_block", "int64", True, "configuration"),
    Column("iterations", "int64", True, "configuration"),
    Column("k_dimm", "int64", True, "configuration"),
    Column("l1_acc", "string", True, "configuration"),
    Column("math_fidelity", "string", True, "configuration"),
    Column("math_transpose_faces", "string", True, "configuration"),
    Column("mathop", "string", True, "configuration"),
    Column("num_blocks", "int64", True, "configuration"),
    Column("num_faces", "int64", True, "configuration"),
    Column("num_faces_A", "int64", True, "configuration"),
    Column("num_faces_B", "int64", True, "configuration"),
    Column("num_tiles_in_block", "int64", True, "configuration"),
    Column("output_format", "string", True, "configuration"),
    Column("output_num_blocks", "int64", True, "configuration"),
    Column("output_num_tiles_in_block", "int64", True, "configuration"),
    Column("partial_a", "bool", True, "configuration"),
    Column("partial_b", "bool", True, "configuration"),
    Column("partial_face_math", "bool", True, "configuration"),
    Column("partial_face_pack", "bool", True, "configuration"),
    Column("pool_type", "string", True, "configuration"),
    Column("r_dimm", "int64", True, "configuration"),
    Column("reduce_pool_type", "string", True, "configuration"),
    Column("relu_config", "int64", True, "configuration"),
    Column("srca_reuse_count", "int64", True, "configuration"),
    Column("stable_sort", "string", True, "configuration"),
    Column("ternary_mathop", "string", True, "configuration"),
    Column("ternary_scalar_bits", "int64", True, "configuration"),
    Column("throttle_level", "int64", True, "configuration"),
    Column("tilize", "string", True, "configuration"),
    Column("unpack_transpose_faces", "string", True, "configuration"),
    Column("unpack_transpose_within_face", "string", True, "configuration"),
    Column("value_bits", "int64", True, "configuration"),
    # timing (complete {mean, std} x base grid — see _TIMING_COLUMNS above)
    *_TIMING_COLUMNS,
    # ── provenance: stamped by the publish layer, never emitted by a test ──
    Column("test_name", "string", False, "identity", origin="ci"),
    Column("commit_sha", "string", False, "provenance", origin="ci"),
    Column("arch", "string", False, "provenance", origin="ci"),
    Column("run_id", "string", False, "provenance", origin="ci"),
    Column("timestamp", "string", False, "provenance", origin="ci"),
    Column("pipeline", "string", False, "provenance", origin="ci"),  # PR | nightly
    Column("pr_number", "string", True, "provenance", origin="ci"),  # NULL for nightly
]

# Views onto the one schema, by who fills each column. The converter validates a
# test's report against OUTPUT_SCHEMA; the publish layer stamps PROVENANCE. Both
# derive from DB_SCHEMA, so there is a single source of truth.
OUTPUT_SCHEMA = [c for c in DB_SCHEMA if c.origin == "test"]
PROVENANCE = [c for c in DB_SCHEMA if c.origin == "ci"]

MANDATORY = [c.name for c in DB_SCHEMA if not c.nullable]

# Columns a test emits but the published table intentionally drops. The converter
# removes them instead of failing on an unknown column.
#   TEXT_SIZE(...)  — per-stage ELF code size; not used by the gate
#   <RUN_TYPE>_<metric>  — counter-derived metrics, only produced under
#   --enable-perf-counters, which no pipeline passes
DROPPED_COLUMNS = {
    "TEXT_SIZE(L1_TO_L1)",
    "TEXT_SIZE(MATH_ISOLATE)",
    "TEXT_SIZE(PACK_ISOLATE)",
    "TEXT_SIZE(UNPACK_ISOLATE)",
} | {
    metric_column(run_type, base)
    for run_type in RUN_TYPE_NAMES
    for metric in METRIC_BASES
    for base in (metric, stat_column(metric, MEAN), stat_column(metric, STD))
}

# Row identity: one test config in one run. The sweep-parameter columns (which
# vary per test) complete the key on top of these fixed columns.
ROW_KEY = ["test_name", "commit_sha", "arch", "run_id"]

# TODO(canonicalization, deferred — see #51245): some columns name the same thing
# differently across tests and are not yet unified, e.g.
#   - c_dimm / k_dimm / r_dimm  vs  in0_c_dim / ct_dim
#   - formats.input_A / formats.output  vs  input_format / output_format
# Picking one canonical name per concept needs sign-off, so don't merge them here.
