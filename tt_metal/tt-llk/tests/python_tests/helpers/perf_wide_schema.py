# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared wide nullable schema for LLK performance reports (v1).

Two layers make up the published table:

  OUTPUT_SCHEMA  columns a perf test produces (config, timings, counters, code
                 size, marker). Reports are validated against this.
  PROVENANCE     run context (commit, arch, run id, ...) added by CI, not the test.
  DB_SCHEMA      OUTPUT_SCHEMA + PROVENANCE, one row per test config per run.

Every OUTPUT column is nullable: a test fills what it uses, the rest stay NULL.
New columns are added later as nullable, so v1 need not be complete.

Imports no device libraries, so it loads without hardware.
"""

from dataclasses import dataclass

from .perf_schema import MEAN, STD, stat_column


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # int64 | float64 | bool | string
    nullable: bool
    category: str


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


# ── Output schema: columns a perf test produces (validate reports against this) ──
# TODO: counters/metrics (counter-enabled runs) and Quasar columns join here as
# nullable once captured.
OUTPUT_SCHEMA = [
    Column("marker", "string", False, "identity"),
    # formats
    Column("formats.input_A", "string", True, "formats"),
    Column("formats.input_B", "string", True, "formats"),
    Column("formats.output", "string", True, "formats"),
    Column("formats.register_A", "string", True, "formats"),
    Column("formats.register_B", "string", True, "formats"),
    # flags
    Column("dest_acc", "string", True, "flags"),
    Column("unpack_to_dest", "string", True, "flags"),
    # key
    Column("loop_factor", "int64", True, "key"),
    Column("tile_cnt", "int64", True, "key"),
    # configuration
    Column("approx_mode", "string", True, "configuration"),
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
    Column("srca_reuse_count", "int64", True, "configuration"),
    Column("stable_sort", "string", True, "configuration"),
    Column("throttle_level", "int64", True, "configuration"),
    Column("tilize", "string", True, "configuration"),
    Column("unpack_transpose_faces", "string", True, "configuration"),
    Column("unpack_transpose_within_face", "string", True, "configuration"),
    Column("value_bits", "int64", True, "configuration"),
    # timing (complete {mean, std} x base grid — see _TIMING_COLUMNS above)
    *_TIMING_COLUMNS,
    # code_size
    Column("TEXT_SIZE(L1_TO_L1)", "int64", True, "code_size"),
    Column("TEXT_SIZE(MATH_ISOLATE)", "int64", True, "code_size"),
    Column("TEXT_SIZE(PACK_ISOLATE)", "int64", True, "code_size"),
    Column("TEXT_SIZE(UNPACK_ISOLATE)", "int64", True, "code_size"),
]

# ── Provenance: run context added by CI/publish layer (NOT produced by tests) ──
PROVENANCE = [
    Column("test_name", "string", False, "identity"),
    Column("commit_sha", "string", False, "provenance"),
    Column("arch", "string", False, "provenance"),
    Column("run_id", "string", False, "provenance"),
    Column("timestamp", "string", False, "provenance"),
    Column("pipeline", "string", False, "provenance"),  # PR | nightly
    Column("pr_number", "string", True, "provenance"),  # NULL for nightly
]

# ── Published table = test output + provenance ──
DB_SCHEMA = OUTPUT_SCHEMA + PROVENANCE

MANDATORY = [c.name for c in DB_SCHEMA if not c.nullable]

# Row identity: one test config in one run. The sweep-parameter columns (which
# vary per test) complete the key on top of these fixed columns.
ROW_KEY = ["test_name", "commit_sha", "arch", "run_id"]

# TODO(canonicalization, deferred — see #51245): some columns name the same thing
# differently across tests and are not yet unified, e.g.
#   - c_dimm / k_dimm / r_dimm  vs  in0_c_dim / ct_dim
#   - formats.input_A / formats.output  vs  input_format / output_format
#   - num_blocks  vs  input_num_blocks / output_num_blocks
# Picking one canonical name per concept needs sign-off, so don't merge them here.
