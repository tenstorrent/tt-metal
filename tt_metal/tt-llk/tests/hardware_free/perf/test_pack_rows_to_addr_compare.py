# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "perf"))
from compare_pack_rows_to_addr import (  # noqa: E402
    CONFIG,
    FORMATS,
    MEAN,
    ROWS,
    STD,
    compare,
    load_report,
)


def raw_report(cycles):
    return pd.DataFrame(
        [
            {
                "formats.output": fmt,
                "num_rows_to_pack": rows,
                "marker": "TILE_LOOP",
                "dest_acc": "Yes",
                "dest_sync": "Half",
                "speed_of_light": False,
                "loop_factor": 64,
                "tile_cnt": 16,
                "formats.input_A": "Float16_b",
                "formats.register_A": "Float16_b",
                "formats.sfpu_src": "Float16_b",
                "unpack_to_dest": False,
                MEAN: cycles * 1024,
                STD: 256.0,
            }
            for fmt in FORMATS
            for rows in ROWS
        ]
    )


def write_report(tmp_path, name, frame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_normalizes_raw_timings_and_computes_speedup(tmp_path):
    quasar = load_report(write_report(tmp_path, "quasar.csv", raw_report(10)))
    blackhole = load_report(write_report(tmp_path, "blackhole.csv", raw_report(20)))
    result = compare(quasar, blackhole)
    assert len(result) == 27
    assert (result.quasar_cycles_per_call == 10).all()
    assert (result.blackhole_cycles_per_call == 20).all()
    assert (result.quasar_std_cycles_per_call == 0.25).all()
    assert (result.quasar_speedup_vs_blackhole == 2).all()


def test_rejects_incomplete_sweep(tmp_path):
    path = write_report(tmp_path, "raw.csv", raw_report(10).iloc[:-1])
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="complete"
    ):
        load_report(path)


def test_rejects_duplicate_measurements(tmp_path):
    frame = raw_report(10)
    path = write_report(tmp_path, "raw.csv", pd.concat([frame, frame.iloc[:1]]))
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="duplicate"
    ):
        load_report(path)


@pytest.mark.parametrize("cycles", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_timings(tmp_path, cycles):
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError
    ):
        load_report(write_report(tmp_path, "raw.csv", raw_report(cycles)))


def test_rejects_already_normalized_input(tmp_path):
    path = write_report(tmp_path, "report.post.csv", raw_report(10))
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="already normalized"
    ):
        load_report(path)


@pytest.mark.parametrize("column", CONFIG)
def test_rejects_mismatched_configuration(column):
    quasar, blackhole = raw_report(10), raw_report(20)
    blackhole[column] = "different"
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match=column
    ):
        compare(quasar, blackhole)
