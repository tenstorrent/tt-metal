# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of QWEN38_GR_GATE_ROWTRIM: default on, 2 iterations only for rows <= 4, kernels keep 8 otherwise."""

from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_read
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_read import gate3

KERNELS = Path(gr_read.__file__).parent / "kernels"
DEFINE = "QWEN38_GR_ROWTRIM_ITERS"


def test_default_on_and_off_switch(monkeypatch):
    monkeypatch.delenv(gate3.ROWTRIM_ENV, raising=False)
    assert gate3.rowtrim_defines(1) == [(DEFINE, "2")]
    assert gate3.rowtrim_defines(8) == [(DEFINE, "4")]
    assert gate3.rowtrim_defines(16) == []
    monkeypatch.setenv(gate3.ROWTRIM_ENV, "0")
    for rows in (1, 2, 4, 8, 16):
        assert gate3.rowtrim_defines(rows) == []


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_on_small_rows_runs_two_iterations(monkeypatch, rows):
    monkeypatch.setenv(gate3.ROWTRIM_ENV, "1")
    assert gate3.rowtrim_defines(rows) == [(DEFINE, "2")]


@pytest.mark.parametrize("rows", [5, 6, 7, 8])
def test_on_rows_five_to_eight_runs_four_iterations(monkeypatch, rows):
    monkeypatch.setenv(gate3.ROWTRIM_ENV, "1")
    assert gate3.rowtrim_defines(rows) == [(DEFINE, "4")]


@pytest.mark.parametrize("rows", [16, 32])
def test_on_large_rows_keeps_eight(monkeypatch, rows):
    monkeypatch.setenv(gate3.ROWTRIM_ENV, "1")
    assert gate3.rowtrim_defines(rows) == []


def test_kernels_default_to_eight_and_every_sfpu_op_uses_the_define():
    for name in ("gate_compute3.cpp", "lowrank_compute3.cpp"):
        text = (KERNELS / name).read_text()
        assert f"#ifndef {DEFINE}\n#define {DEFINE} 8\n#endif" in text
        assert "8 /* ITERATIONS */" not in text
        assert "sigmoid_tile<" not in text  # the stock wrapper hard-codes 8
        assert text.count(f"{DEFINE} /* ITERATIONS */") >= 2
    assert (KERNELS / "gate_compute3.cpp").read_text().count(
        f"{DEFINE} /* ITERATIONS */"
    ) == 2  # sigmoid, gate multiply
    assert (KERNELS / "lowrank_compute3.cpp").read_text().count(f"{DEFINE} /* ITERATIONS */") == 3  # silu, sigmoid, x2
