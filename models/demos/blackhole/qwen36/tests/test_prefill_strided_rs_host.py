# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) of the QWEN36_PREFILL_STRIDED_RS gates (tp_common.prefill_strided_rs_ok and its M-block
choice): flag off never takes the strided op, the row floors and the 1x8-ring check hold, and only whole M blocks per
matmul grid row are taken.

Run (worktree first on PYTHONPATH, ttnn from the built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_prefill_strided_rs_host.py -q
"""
import types

import pytest

from models.demos.blackhole.qwen36.tt import tp_common as tpc

KNOBS = (
    "QWEN36_PREFILL_STRIDED_RS",
    "QWEN36_PREFILL_STRIDED_RS_DOWN_MIN",
    "QWEN36_PREFILL_STRIDED_RS_WO_MIN",
    "QWEN36_PREFILL_STRIDED_RS_WIN",
    "QWEN36_PREFILL_STRIDED_RS_SMALL_M",
)
RING = types.SimpleNamespace(shape=(1, 8))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(tpc, "_STRIDED_RS", None)
    yield
    tpc._STRIDED_RS = None


def test_flag_off_never_strided():
    for rows in (2048, 8192, 16384):
        assert not tpc.prefill_strided_rs_ok("down", rows, RING)
        assert not tpc.prefill_strided_rs_ok("wo", rows, RING)
    assert tpc.strided_rs_counters(None, object()) is None  # no device allocation when off


def test_default_floors(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_STRIDED_RS", "1")
    assert tpc.prefill_strided_rs_ok("down", 2048, RING)
    assert tpc.prefill_strided_rs_ok("down", 8192, RING)
    assert not tpc.prefill_strided_rs_ok("down", 1024, RING)
    assert not tpc.prefill_strided_rs_ok("wo", 2048, RING)  # wo is off by default (no gain in the model's chunk)
    assert not tpc.prefill_strided_rs_ok("wo", 8192, RING)
    assert not tpc.prefill_strided_rs_ok("down", 32, RING)  # decode widths


def test_only_the_1x8_ring(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_STRIDED_RS", "1")
    for shape in ((1, 4), (8, 1), (4, 8), (1, 1)):
        assert not tpc.prefill_strided_rs_ok("down", 8192, types.SimpleNamespace(shape=shape))


def test_floor_knobs(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_STRIDED_RS", "1")
    monkeypatch.setenv("QWEN36_PREFILL_STRIDED_RS_DOWN_MIN", "1000000")
    monkeypatch.setenv("QWEN36_PREFILL_STRIDED_RS_WO_MIN", "256")
    assert not tpc.prefill_strided_rs_ok("down", 8192, RING)
    assert tpc.prefill_strided_rs_ok("wo", 2048, RING)
    assert tpc.prefill_strided_rs_ok("wo", 1024, RING)


@pytest.mark.parametrize(
    "rows,mb",
    [(8192, 8), (16384, 8), (4096, 8), (2048, 4), (1024, 4), (3072, 4), (6144, 8), (256, None), (512, None),
     (2080, None), (5120, None), (100, None)],
)
def test_m_block(rows, mb):
    # 8 matmul grid rows: whole M blocks of 8 tiles (>= 16 tiles per grid row) or 4 tiles, else not taken
    assert tpc._strided_rs_m_block(rows) == mb


def test_small_m_stress_knob(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_STRIDED_RS", "1")
    monkeypatch.setenv("QWEN36_PREFILL_STRIDED_RS_DOWN_MIN", "256")
    assert not tpc.prefill_strided_rs_ok("down", 256, RING)  # off by default
    tpc._STRIDED_RS = None
    monkeypatch.setenv("QWEN36_PREFILL_STRIDED_RS_SMALL_M", "1")
    assert tpc.prefill_strided_rs_ok("down", 256, RING) and tpc.prefill_strided_rs_ok("down", 512, RING)
    assert not tpc.prefill_strided_rs_ok("down", 128, RING)  # 4 tiles cannot split over 8 grid rows
    assert tpc._strided_rs_m_block(256, True) == 1 and tpc._strided_rs_m_block(512, True) == 2
    assert tpc._strided_rs_m_block(8192, True) == 8
