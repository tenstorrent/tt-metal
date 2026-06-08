# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Off-device unit tests for the pure parts of the accuracy harness."""

import numpy as np
import pandas as pd
from accuracy.accuracy_harness import (
    CSV_COLUMNS,
    build_sweep_spec,
    merge_shards,
    rows_dataframe,
    variant_name,
    write_shard,
)
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
)
from helpers.stimuli_generator import DistributionKind


def test_variant_name_encodes_op_formats_and_config():
    name = variant_name(
        MathOperation.Exp,
        DataFormat.Float16_b,
        DataFormat.Float16_b,
        ApproximationMode.No,
        FastMode.No,
        DestAccumulation.Yes,
    )
    assert name == "exp__Float16_b_Float16_b__approx0_fast0_dest1"


def test_build_sweep_spec_is_ramp_within_defined_domain():
    # Log is undefined for x <= 0; the sweep spec must be a RAMP whose
    # intervals stay strictly positive after exclude_undefined.
    spec = build_sweep_spec(MathOperation.Log, DataFormat.Float16_b)
    assert spec.distribution == DistributionKind.RAMP
    assert spec.intervals is not None
    for lo, hi in spec.intervals:
        assert lo > 0 and hi > 0


def test_build_sweep_spec_reciprocal_excludes_zero():
    # Reciprocal's registry domain is two bands; undefined hole around 0.
    spec = build_sweep_spec(MathOperation.Reciprocal, DataFormat.Float16_b)
    assert spec.distribution == DistributionKind.RAMP
    for lo, hi in spec.intervals:
        assert not (lo <= 0.0 <= hi)


def test_csv_columns_have_expected_schema_order():
    assert CSV_COLUMNS == [
        "op",
        "input_format",
        "output_format",
        "chip_arch",
        "distribution",
        "intervals",
        "variant_name",
        "seed",
        "sample_index",
        "x",
        "golden",
        "hw",
        "approx_mode",
        "fast_mode",
        "dest_acc",
        "abs_error",
        "signed_error",
        "rel_error",
        "signed_ulp_error",
        "abs_ulp_error",
        "is_finite_hw",
        "is_finite_golden",
    ]


def _toy_df(op="exp", in_fmt="Float16_b", x=(0.0, 1.0)):
    return rows_dataframe(
        op_name=op,
        in_fmt=in_fmt,
        out_fmt="Float16_b",
        chip_arch="wormhole",
        distribution="ramp",
        intervals="[(0.0, 1.0)]",
        variant=f"{op}__toy",
        seed="",
        approx="0",
        fast="0",
        dest="1",
        x=np.array(x, dtype=np.float64),
        golden=np.array([1.0, 2.0], dtype=np.float64),
        hw=np.array([1.0, 2.5], dtype=np.float64),
        out_fmt_enum_name="Float16_b",
    )


def test_rows_dataframe_has_all_columns_and_sample_index(tmp_path):
    df = _toy_df()
    assert list(df.columns) == CSV_COLUMNS
    assert df["sample_index"].tolist() == [0, 1]


def test_write_and_merge_overwrites_from_current_run_only(tmp_path, monkeypatch):
    # Redirect output dirs into tmp_path so the test never touches the repo.
    monkeypatch.setattr("accuracy.accuracy_harness.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(
        "accuracy.accuracy_harness.SHARD_DIR", tmp_path / "out" / "_shards"
    )
    (tmp_path / "out" / "_shards").mkdir(parents=True)

    # A stale final CSV from a "previous run" must be overwritten, not appended.
    (tmp_path / "out").mkdir(exist_ok=True)
    pd.DataFrame({"op": ["exp"], "x": [999.0]}).to_csv(
        tmp_path / "out" / "exp.csv", index=False
    )

    write_shard(_toy_df(op="exp"))
    write_shard(_toy_df(op="log"))
    merge_shards()

    exp = pd.read_csv(tmp_path / "out" / "exp.csv")
    assert 999.0 not in exp["x"].values  # stale data gone
    assert sorted(exp["x"].tolist()) == [0.0, 1.0]
    assert (tmp_path / "out" / "log.csv").exists()


def test_merge_sorts_deterministically(tmp_path, monkeypatch):
    monkeypatch.setattr("accuracy.accuracy_harness.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(
        "accuracy.accuracy_harness.SHARD_DIR", tmp_path / "out" / "_shards"
    )
    (tmp_path / "out" / "_shards").mkdir(parents=True)

    write_shard(_toy_df(op="exp", x=(1.0, 0.0)))  # unsorted x on purpose
    merge_shards()
    exp = pd.read_csv(tmp_path / "out" / "exp.csv")
    assert exp["x"].tolist() == [0.0, 1.0]  # sorted ascending by x
