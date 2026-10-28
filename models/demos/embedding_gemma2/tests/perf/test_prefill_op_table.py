# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks of the grid the op table's roofline uses (prefill_op_table.grid_cores_of / --grid-cores).

    pytest models/demos/embedding_gemma2/tests/perf/test_prefill_op_table.py
"""

import csv
import json
import sys

import pytest

from models.demos.embedding_gemma2.tests.perf import prefill_op_table as pot

FIELDS = [
    "OP CODE",
    "OP TYPE",
    "DEVICE KERNEL DURATION [ns]",
    "OP TO OP LATENCY [ns]",
    "CORE COUNT",
    "AVAILABLE WORKER CORE COUNT",
    "ATTRIBUTES",
]


def _op(avail, code="RMSNorm", dur=1000):
    return {
        "OP CODE": code,
        "OP TYPE": "tt_dnn_device",
        "DEVICE KERNEL DURATION [ns]": str(dur),
        "OP TO OP LATENCY [ns]": "500",
        "CORE COUNT": "8",
        "AVAILABLE WORKER CORE COUNT": avail,
        "ATTRIBUTES": "",
    }


def _signpost(name):
    return {k: "" for k in FIELDS} | {"OP CODE": name, "OP TYPE": "signpost"}


def _write_csv(path, avail_values):
    rows = [_signpost("PERF_TEXT")] + [_op(a) for a in avail_values] + [_signpost("PERF_TEXT_END")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_grid_from_csv_column(tmp_path):
    p = _write_csv(tmp_path / "ops.csv", ["120", "120", "120"])
    assert pot.grid_cores_of(p, pot.load_forward(p)) == 120


def test_grid_from_device_info_when_column_empty(tmp_path):
    p = _write_csv(tmp_path / "ops.csv", ["", ""])
    (tmp_path / "device_info.json").write_text(json.dumps({"compute_grid": [12, 10]}))
    assert pot.grid_cores_of(p, pot.load_forward(p)) == 120


def test_column_wins_over_device_info(tmp_path):
    p = _write_csv(tmp_path / "ops.csv", ["130", "130"])
    (tmp_path / "device_info.json").write_text(json.dumps({"compute_grid": [12, 10]}))
    assert pot.grid_cores_of(p, pot.load_forward(p)) == 130


def test_no_grid_record_is_an_error(tmp_path, expect_error):
    p = _write_csv(tmp_path / "ops.csv", ["", ""])
    with expect_error(ValueError, "no grid record"):
        pot.grid_cores_of(p, pot.load_forward(p))


def test_mixed_grid_values_are_an_error(tmp_path, expect_error):
    p = _write_csv(tmp_path / "ops.csv", ["120", "130", ""])
    with expect_error(ValueError, "different grids"):
        pot.grid_cores_of(p, pot.load_forward(p))


def test_explicit_grid_overrides_csv(tmp_path):
    p = _write_csv(tmp_path / "ops.csv", ["130", "130"])
    assert pot.table(p, 128, grid_cores=120)["grid_cores"] == 120
    assert pot.table(p, 128)["grid_cores"] == 130


@pytest.mark.parametrize("bad", [0, -120])
def test_non_positive_grid_cores_is_an_error(tmp_path, bad, expect_error):
    p = _write_csv(tmp_path / "ops.csv", ["120", "120"])
    with expect_error(ValueError, "positive core count"):
        pot.table(p, 128, grid_cores=bad)


def test_cli_grid_cores_zero_is_an_error(tmp_path, monkeypatch, capsys, expect_error):
    p = _write_csv(tmp_path / "ops.csv", ["120", "120"])
    monkeypatch.setattr(sys, "argv", ["prefill_op_table", str(p), "--bucket", "128", "--grid-cores", "0"])
    with expect_error(SystemExit, "2") as e:  # argparse error: exit code 2
        pot.main()
    assert e.value.code == 2
    assert "--grid-cores must be a positive core count" in capsys.readouterr().err
