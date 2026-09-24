# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the warehouse baseline reader. No Snowflake, no hardware."""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "perf"),
)
import baseline_from_warehouse as bw

PARAMS = json.dumps(
    {
        "c_dimm": 3,
        "dest_sync": "DestSync.Full",
        "dst_index": 0,
        "formats.sfpu_src": "Float16",
        "in0_r_dim": 32,
        "partial_a": False,
        "throttle_level": 5,
        "unpack_transpose_faces": "Transpose.No",
    }
)
DENSE_VALUES = (
    "Float16", "Float16", "Float32", "Float16", "Float16",
    "DestAccumulation.Yes", "MathFidelity.HiFi3", 6, 1, 1024, True, False,
)
METRICS = ("L1_TO_L1", "UNPACK_ISOLATE", "MATH_ISOLATE", "PACK_ISOLATE",
           "L1_CONGESTION[UNPACK]", "L1_CONGESTION[PACK]")


def _rows(marker="INIT", base=183.0):
    return [DENSE_VALUES + (PARAMS, marker, m, base + i)
            for i, m in enumerate(METRICS)]


class FakeCursor:
    def __init__(self, one=None, many=None):
        self._one, self._many = one, many or []
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many

    def fetchmany(self, n):
        batch, self._many = self._many[:n], self._many[n:]
        return batch


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_six_metric_rows_collapse_into_one_csv_row():
    records, columns, _ = bw.pivot(_rows())
    assert len(records) == 1
    for m in METRICS:
        assert f"mean({m})" in columns
        assert f"mean({m})" in records[0]


def test_markers_stay_separate_points():
    records, _, _ = bw.pivot(_rows("INIT") + _rows("KERNEL", 9000.0))
    assert {r["marker"] for r in records} == {"INIT", "KERNEL"}
    assert len(records) == 2


def test_dense_columns_take_their_csv_names():
    record = bw.pivot(_rows())[0][0]
    assert record["formats.input_A"] == "Float16"
    assert record["formats.output"] == "Float32"
    assert record["dest_acc"] == "DestAccumulation.Yes"
    assert record["math_fidelity"] == "MathFidelity.HiFi3"
    assert record["tile_cnt"] == 6
    assert record["speed_of_light"] is True


def test_params_are_expanded_under_their_own_keys():
    record = bw.pivot(_rows())[0][0]
    assert record["dest_sync"] == "DestSync.Full"
    assert record["throttle_level"] == 5
    assert record["in0_r_dim"] == 32
    assert record["unpack_transpose_faces"] == "Transpose.No"


def test_no_provenance_column_reaches_the_csv():
    """A leaked commit_sha makes every point key differ and the gate go green."""
    _, columns, _ = bw.pivot(_rows())
    for banned in bw.PROVENANCE:
        assert banned not in columns
        assert banned.lower() not in columns


def test_csv_round_trips_with_the_dtypes_a_perf_csv_gives():
    records, columns, _ = bw.pivot(_rows())
    bw.write_csv(records, columns, "baseline_perf/b.csv")
    df = pd.read_csv("baseline_perf/b.csv")
    assert df.speed_of_light.dtype == bool
    assert df.tile_cnt.dtype.kind == "i"
    assert df.loc[0, "dest_acc"] == "DestAccumulation.Yes"


def test_pick_run_filters_arch_pipeline_and_mode():
    cur = FakeCursor(one=("nightly-20260922-x-blackhole", "abc", "2026-09-22"))
    run = bw.pick_run(cur, bw.VIEW, "blackhole", "baseline", False)
    assert run["run_id"] == "nightly-20260922-x-blackhole"
    sql, params = cur.executed[0]
    assert params == ("blackhole", "baseline", False)
    assert "MAX(RUN_TS) DESC" in sql and "TIMESTAMP" not in sql


def test_pick_run_returns_none_when_no_such_run():
    assert bw.pick_run(FakeCursor(one=None), bw.VIEW, "wormhole",
                       "baseline", True) is None


def test_fetch_pins_one_run_and_one_mode():
    cur = FakeCursor(many=[])
    list(bw.fetch(cur, bw.VIEW, "r1", True))
    sql, params = cur.executed[0]
    assert params == ("r1", True)
    assert "WHERE RUN_ID = %s AND SPEED_OF_LIGHT = %s" in sql


def test_missing_baseline_is_recorded_as_absent(tmp_path):
    bw._write_meta(str(tmp_path / "m.json"), {}, have=False)
    assert (tmp_path / "have_baseline.txt").read_text() == "false"
    assert json.load(open(tmp_path / "m.json"))["source"] == bw.VIEW


def test_connect_refuses_an_empty_key(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_PRIVATE_KEY", "")
    with pytest.raises(SystemExit):
        bw.connect()


def test_repeated_rows_are_medianed_not_overwritten():
    rows = [DENSE_VALUES + (PARAMS, "INIT", "L1_TO_L1", v)
            for v in (100.0, 900.0, 300.0)]
    records, _, seen = bw.pivot(rows)
    assert seen == 3 and len(records) == 1
    assert records[0]["mean(L1_TO_L1)"] == 300.0


def test_the_baseline_does_not_depend_on_row_order():
    rows = [DENSE_VALUES + (PARAMS, "INIT", "L1_TO_L1", v)
            for v in (100.0, 900.0, 300.0, 700.0)] + _rows("KERNEL")
    first, _, _ = bw.pivot(rows)
    second, _, _ = bw.pivot(list(reversed(rows)))
    key = lambda rs: sorted(repr(sorted(r.items())) for r in rs)
    assert key(first) == key(second)
