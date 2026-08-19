# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free Parquet tests for the Quasar published schema.

Mirrors test_perf_parquet.py against wide_schema_quasar.DB_SCHEMA. Needs pyarrow,
no chip.
"""

import pandas as pd
import pyarrow.parquet as pq
import pytest
from helpers.chip_architecture import ChipArchitecture
from helpers.perf.parquet import (
    arrow_schema,
    convert_csvs_to_parquet,
    schema_for_arch,
)
from helpers.perf.schema import MEAN, stat_column
from helpers.perf.test_schemas import PERF_TEST_SCHEMAS_QSR
from helpers.perf.wide_schema import DB_SCHEMA as WH_BH_SCHEMA
from helpers.perf.wide_schema_quasar import DB_SCHEMA as QSR_SCHEMA
from helpers.perf.wide_schema_quasar import DROPPED_COLUMNS as QSR_DROPPED_COLUMNS

_QSR_PROV = dict(
    commit_sha="abc123",
    arch="quasar",
    run_id="42",
    timestamp="2026-01-01T00:00:00",
    pipeline="PR",
    pr_number="7",
)

_UNPACK_TILIZE_QSR_COLS = (
    "data_copy_type",
    "face_c_dim",
    "face_r_dim",
    "implied_math_format",
    "num_faces_c_dim_A",
    "num_faces_c_dim_B",
    "num_faces_r_dim_A",
    "num_faces_r_dim_B",
    "unpacker_engine_sel",
)


def _write_csv(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_schema_for_arch_selects_quasar_table():
    assert schema_for_arch(ChipArchitecture.QUASAR) is QSR_SCHEMA
    assert schema_for_arch(ChipArchitecture.WORMHOLE) is WH_BH_SCHEMA
    assert schema_for_arch(ChipArchitecture.BLACKHOLE) is WH_BH_SCHEMA


def test_schema_for_arch_rejects_unsupported_architecture():
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="Unsupported architecture"
    ):
        schema_for_arch("grayskull")


def test_quasar_schema_is_not_wh_bh_schema():
    # The split: Quasar-only columns must not land in the WH/BH published table.
    qsr_only = {c.name for c in QSR_SCHEMA} - {c.name for c in WH_BH_SCHEMA}
    assert "face_c_dim" in qsr_only
    assert "unpacker_engine_sel" in qsr_only
    assert "implied_math_format" in qsr_only
    assert "enable_2x_format" in qsr_only
    assert "enable_direct_indexing" in qsr_only
    assert stat_column("L1_TO_L1[FPU]", MEAN) in qsr_only
    assert stat_column("L1_TO_L1[SFPU]", MEAN) in qsr_only
    assert stat_column("SFPU_ISOLATE", MEAN) in qsr_only
    # WH/BH-only columns must not land in the Quasar published table.
    wh_only = {c.name for c in WH_BH_SCHEMA} - {c.name for c in QSR_SCHEMA}
    assert "clamp_negative" in wh_only
    assert "ternary_mathop" in wh_only


def test_arrow_schema_matches_quasar_db_schema():
    schema = arrow_schema(QSR_SCHEMA)
    assert schema.names == [c.name for c in QSR_SCHEMA]
    for field, col in zip(schema, QSR_SCHEMA):
        assert field.nullable == col.nullable


def test_catalog_columns_are_in_quasar_db_schema():
    schema_names = {c.name for c in QSR_SCHEMA} | QSR_DROPPED_COLUMNS
    missing = {}
    for test, entry in PERF_TEST_SCHEMAS_QSR.items():
        unknown = sorted(set(entry["columns"]) - schema_names)
        if unknown:
            missing[test] = unknown
    assert not missing, (
        "Quasar perf-test catalog column(s) are not in "
        "helpers.perf.wide_schema_quasar.DB_SCHEMA and would be dropped from "
        f"Parquet: {missing}. Add them as nullable columns."
    )


def test_convert_keeps_quasar_unpack_tilize_columns(tmp_path):
    # The 9 unpack-tilize columns that used to trip the live-run drop warning
    # against the WH/BH schema: they must survive CSV -> Parquet on a Quasar run.
    df = pd.DataFrame(
        {
            "marker": ["INIT"],
            "data_copy_type": ["DataCopyType.A2D"],
            "face_c_dim": [16],
            "face_r_dim": [16],
            "implied_math_format": ["ImpliedMathFormat.Yes"],
            "num_faces_c_dim_A": [2],
            "num_faces_c_dim_B": [2],
            "num_faces_r_dim_A": [2],
            "num_faces_r_dim_B": [2],
            "unpacker_engine_sel": ["UnpackerEngine.UnpA"],
            "tile_cnt": [8],
        }
    )
    p = _write_csv(tmp_path, "perf_unpack_tilize_quasar.csv", df)

    diag = convert_csvs_to_parquet([p], tmp_path / "out.parquet", **_QSR_PROV)

    assert diag["unknown_columns"] == {}
    names = pq.read_table(tmp_path / "out.parquet").schema.names
    assert names == [c.name for c in QSR_SCHEMA]
    for col in _UNPACK_TILIZE_QSR_COLS:
        assert col in names


def test_convert_drops_quasar_columns_on_wh_bh_schema(tmp_path):
    # Same CSV against a wormhole run: Quasar-only columns are unknown and dropped.
    df = pd.DataFrame(
        {
            "marker": ["INIT"],
            "face_c_dim": [16],
            "unpacker_engine_sel": ["UnpackerEngine.UnpA"],
            "tile_cnt": [8],
        }
    )
    p = _write_csv(tmp_path, "perf_unpack_tilize_quasar.csv", df)

    diag = convert_csvs_to_parquet(
        [p],
        tmp_path / "out.parquet",
        strict=False,
        commit_sha="abc123",
        arch="wormhole",
        run_id="42",
        timestamp="2026-01-01T00:00:00",
        pipeline="PR",
        pr_number="7",
    )

    assert diag["unknown_columns"]["perf_unpack_tilize_quasar"] == [
        "face_c_dim",
        "unpacker_engine_sel",
    ]
    names = pq.read_table(tmp_path / "out.parquet").schema.names
    assert "face_c_dim" not in names
    assert names == [c.name for c in WH_BH_SCHEMA]


def test_convert_drops_sfpu_isolate_text_size_on_quasar(tmp_path):
    # TEXT_SIZE(SFPU_ISOLATE) is Quasar-dropped (#53072), not a published column.
    df = pd.DataFrame(
        {
            "marker": ["INIT"],
            "enable_2x_format": [True],
            "implied_math_format": ["ImpliedMathFormat.Yes"],
            "TEXT_SIZE(SFPU_ISOLATE)": [4096],
            "tile_cnt": [8],
        }
    )
    p = _write_csv(tmp_path, "perf_sfpu_exp_parallel_matmul_quasar.csv", df)

    diag = convert_csvs_to_parquet([p], tmp_path / "out.parquet", **_QSR_PROV)

    assert diag["unknown_columns"] == {}
    names = pq.read_table(tmp_path / "out.parquet").schema.names
    assert "TEXT_SIZE(SFPU_ISOLATE)" not in names
    assert "enable_2x_format" in names
    assert "implied_math_format" in names
    assert names == [c.name for c in QSR_SCHEMA]
