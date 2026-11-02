# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of QWEN38_FRONT_ROWTRIM: default on, a 2-iteration rsqrt only for rows <= 4, rsqrt_tile untouched
when the define is absent."""

from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_recip_last as grl
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_recip_last import front3

KERNEL = Path(grl.__file__).parent / "kernels" / "stats_u_recip_compute.cpp"
DEFINE = "QWEN38_FRONT_ROWTRIM_ITERS"


def test_default_on_and_off_switch(monkeypatch):
    monkeypatch.delenv(front3.ROWTRIM_ENV, raising=False)
    assert front3.ROWTRIM_ENV == "QWEN38_FRONT_ROWTRIM"
    assert front3.rowtrim_defines(1) == [(DEFINE, "2")]
    assert front3.rowtrim_defines(8) == [(DEFINE, "4")]
    assert front3.rowtrim_defines(16) == []
    monkeypatch.setenv(front3.ROWTRIM_ENV, "0")
    for rows in (1, 2, 4, 8, 16):
        assert front3.rowtrim_defines(rows) == []


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_on_small_rows(monkeypatch, rows):
    monkeypatch.setenv(front3.ROWTRIM_ENV, "1")
    assert front3.rowtrim_defines(rows) == [(DEFINE, "2")]


@pytest.mark.parametrize("rows", [5, 6, 7, 8])
def test_on_rows_five_to_eight_runs_four_iterations(monkeypatch, rows):
    monkeypatch.setenv(front3.ROWTRIM_ENV, "1")
    assert front3.rowtrim_defines(rows) == [(DEFINE, "4")]


@pytest.mark.parametrize("rows", [0, 16])
def test_on_large_rows_unchanged(monkeypatch, rows):
    monkeypatch.setenv(front3.ROWTRIM_ENV, "1")
    assert front3.rowtrim_defines(rows) == []


def test_kernel_has_rsqrt_as_the_only_sfpu_op_and_keeps_the_wrapper_when_off():
    text = KERNEL.read_text()
    assert f"#ifdef {DEFINE}\n            rsqrt_tile_rows(0);\n#else\n            rsqrt_tile(0);\n#endif" in text
    helper = text[text.index(f"#ifdef {DEFINE}\n// rsqrt_tile") : text.index("ALWI void ACQ()")]
    assert "calculate_rsqrt" in helper and "VectorMode::R)" in helper and "VectorMode::RC" not in helper
    assert helper.count(f"{DEFINE} /* ITERATIONS */") == 1
    assert "copy_tile" not in text and "typecast_tile" not in text
    for op in ("exp_tile", "recip_tile", "sigmoid_tile", "sqrt_tile", "silu_tile"):
        assert op not in text.replace("rsqrt_tile", "")
