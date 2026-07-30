# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared wide nullable schema for LLK performance reports (v1).

Two layers compose the published table:

  OUTPUT_SCHEMA  what a perf test PRODUCES (config + timings + counters +
                 code size + marker). This is the "expected output" contract a
                 test report is validated against. Emitted by the test.
  PROVENANCE     run context (commit, arch, run id, ...) added by the CI/publish
                 layer, NOT by the test. Broadcast onto every output row.
  DB_SCHEMA      = OUTPUT_SCHEMA + PROVENANCE — one published row per test
                 configuration per execution context.

Every OUTPUT column is nullable: a test fills the columns it uses and leaves the
rest NULL. New columns (Quasar, counters/metrics, new params) are added later as
nullable — a non-breaking change — so this v1 need not be complete.

Import-free on purpose (no device libraries) so it loads/validates without hardware.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # int64 | float64 | bool | string
    nullable: bool
    category: str


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
    # timing
    Column("mean(L1_CONGESTION[PACK])", "float64", True, "timing"),
    Column("mean(L1_CONGESTION[UNPACK])", "float64", True, "timing"),
    Column("mean(L1_TO_L1)", "float64", True, "timing"),
    Column("mean(MATH_ISOLATE)", "float64", True, "timing"),
    Column("mean(PACK_ISOLATE)", "float64", True, "timing"),
    Column("mean(UNPACK_ISOLATE)", "float64", True, "timing"),
    Column("std(L1_TO_L1)", "float64", True, "timing"),
    Column("std(PACK_ISOLATE)", "float64", True, "timing"),
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

# ── Row identity ──
# A published row = one test configuration in one execution context, uniquely
# identified by: test + its full sweep configuration + commit + arch + run.
# The fixed part of the key (the configuration columns that complete it vary
# per test and are the sweep-parameter values that row carries):
ROW_KEY = ["test_name", "commit_sha", "arch", "run_id"]

# TODO(canonicalization — deferred, see #51245): same-purpose columns still use
# divergent names across tests and are NOT yet unified here, e.g.
#   - c_dimm / k_dimm / r_dimm  vs  in0_c_dim / ct_dim   (`_dimm` vs `_dim`)
#   - formats.input_A / formats.output  vs  input_format / output_format
#   - num_blocks  vs  input_num_blocks / output_num_blocks
# Aligning to one canonical name per concept needs sign-off; ambiguous fields
# must not be merged automatically.
