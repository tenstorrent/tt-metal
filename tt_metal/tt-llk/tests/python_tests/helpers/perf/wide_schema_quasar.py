# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared wide nullable schema for Quasar performance reports (v1).

Separate published table from the WH/BH schema in ``wide_schema.py``. A Quasar
run is aligned to this schema; a WH/BH run is aligned to that one. The two must
not be mixed: Quasar-only columns (face dims, unpacker engine, implied math
format, ...) stay out of the WH/BH table, and WH/BH-only columns stay out of
this one.

Same conventions as ``wide_schema``: one row per test config per run; every
column except ``marker`` and the mandatory provenance keys is nullable; Column
dtype/nullability/origin are the contract the Parquet writer enforces.

Imports no device libraries, so it loads without hardware.
"""

from .schema import MEAN, STD, stat_column
from .wide_schema import Column

# Same PerfRunType timing grid as WH/BH, plus the 4-TRISC parallel FPU/SFPU
# bases (#53072): L1_TO_L1[FPU], L1_TO_L1[SFPU], SFPU_ISOLATE.
_TIMING_BASES = (
    "L1_TO_L1",
    "L1_TO_L1[FPU]",
    "L1_TO_L1[SFPU]",
    "UNPACK_ISOLATE",
    "MATH_ISOLATE",
    "PACK_ISOLATE",
    "L1_CONGESTION[UNPACK]",
    "L1_CONGESTION[PACK]",
    "SFPU_ISOLATE",
)
_TIMING_COLUMNS = [
    Column(stat_column(base, kind), "float64", True, "timing")
    for base in _TIMING_BASES
    for kind in (MEAN, STD)
]


# ── The published Quasar table: headers + provenance ─────────────────────────
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
    # configuration — Quasar test parameters + pipeline-injected dims
    Column("acc_to_dest", "bool", True, "configuration"),
    Column("approx_mode", "string", True, "configuration"),
    Column("block_ct_dim", "int64", True, "configuration"),
    Column("block_rt_dim", "int64", True, "configuration"),
    Column("broadcast_type", "string", True, "configuration"),
    Column("c_dimm", "int64", True, "configuration"),
    Column("data_copy_type", "string", True, "configuration"),
    Column("dest_sync", "string", True, "configuration"),
    Column("dst_index", "int64", True, "configuration"),
    Column("dst_rounding", "string", True, "configuration"),
    Column("dst_tile_idx", "int64", True, "configuration"),
    Column("enable_2x_format", "bool", True, "configuration"),
    Column("enable_direct_indexing", "bool", True, "configuration"),
    Column("face_c_dim", "int64", True, "configuration"),
    Column("face_r_dim", "int64", True, "configuration"),
    Column("full_ct_dim", "int64", True, "configuration"),
    Column("full_rt_dim", "int64", True, "configuration"),
    Column("implied_math_format", "string", True, "configuration"),
    Column("input_format", "string", True, "configuration"),
    Column("input_num_blocks", "int64", True, "configuration"),
    Column("input_num_tiles_in_block", "int64", True, "configuration"),
    Column("input_tile_cnt", "int64", True, "configuration"),
    Column("k_dimm", "int64", True, "configuration"),
    Column("math_fidelity", "string", True, "configuration"),
    Column("math_transpose_faces", "string", True, "configuration"),
    Column("mathop", "string", True, "configuration"),
    Column("num_blocks", "int64", True, "configuration"),
    Column("num_faces", "int64", True, "configuration"),
    Column("num_faces_A", "int64", True, "configuration"),
    Column("num_faces_B", "int64", True, "configuration"),
    Column("num_faces_c_dim_A", "int64", True, "configuration"),
    Column("num_faces_c_dim_B", "int64", True, "configuration"),
    Column("num_faces_r_dim_A", "int64", True, "configuration"),
    Column("num_faces_r_dim_B", "int64", True, "configuration"),
    Column("num_tiles_in_block", "int64", True, "configuration"),
    Column("op", "string", True, "configuration"),
    Column("output_format", "string", True, "configuration"),
    Column("output_num_blocks", "int64", True, "configuration"),
    Column("output_num_tiles_in_block", "int64", True, "configuration"),
    Column("output_tile_cnt", "int64", True, "configuration"),
    Column("pool_type", "string", True, "configuration"),
    Column("r_dimm", "int64", True, "configuration"),
    Column("relu_config", "int64", True, "configuration"),
    Column("reuse_dest_type", "string", True, "configuration"),
    Column("sign_magnitude", "bool", True, "configuration"),
    Column("src0_tile_idx", "int64", True, "configuration"),
    Column("src1_tile_idx", "int64", True, "configuration"),
    Column("unpack_transpose_faces", "string", True, "configuration"),
    Column("unpack_transpose_within_face", "string", True, "configuration"),
    Column("unpacker_engine_sel", "string", True, "configuration"),
    Column("zero_point_bits", "int64", True, "configuration"),
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

OUTPUT_SCHEMA = [c for c in DB_SCHEMA if c.origin == "test"]
PROVENANCE = [c for c in DB_SCHEMA if c.origin == "ci"]
MANDATORY = [c.name for c in DB_SCHEMA if not c.nullable]

# Columns a Quasar test emits but the published table intentionally drops.
# TEXT_SIZE(...) is per-stage ELF code size; not used by the gate. SFPU_ISOLATE
# is 4-TRISC-only (#53072) and must not live on the WH/BH dropped set.
DROPPED_COLUMNS = {
    "TEXT_SIZE(L1_TO_L1)",
    "TEXT_SIZE(MATH_ISOLATE)",
    "TEXT_SIZE(PACK_ISOLATE)",
    "TEXT_SIZE(UNPACK_ISOLATE)",
    "TEXT_SIZE(SFPU_ISOLATE)",
}
