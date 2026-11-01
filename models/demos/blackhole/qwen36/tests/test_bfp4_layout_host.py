# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host tests (ttnn importable, no device) for QWEN36_BFP4_LAYOUT: flag parsing, the BFP4-only gate, overrides."""
import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv(tpc.BFP4_LAYOUT_ENV, raising=False)
    monkeypatch.delenv(tpc.BFP4_LAYOUT_LOFI_ENV, raising=False)
    for t in tpc.BFP4_LAYOUT_TARGETS:
        monkeypatch.delenv(f"{tpc.BFP4_LAYOUT_ENV}_{t.upper()}", raising=False)
    return monkeypatch


def test_off_by_default(clean):
    for t in tpc.BFP4_LAYOUT_TARGETS:
        assert tpc.bfp4_layout(t, ttnn.bfloat4_b) is None


def test_bfp8_weight_never_retuned(clean):
    clean.setenv(tpc.BFP4_LAYOUT_ENV, "1")
    clean.setenv(f"{tpc.BFP4_LAYOUT_ENV}_DOWN", "1d:84:17")
    assert tpc.bfp4_layout("down", ttnn.bfloat8_b) is None
    assert tpc.bfp4_layout("down", ttnn.bfloat4_b) == ("1d", 84, 17)


def test_tuned_table_and_overrides(clean):
    clean.setenv(tpc.BFP4_LAYOUT_ENV, "1")
    for t in tpc.BFP4_LAYOUT_TARGETS:
        assert tpc.bfp4_layout(t, ttnn.bfloat4_b) == tpc.BFP4_LAYOUT_TUNED[t]
    clean.setenv(f"{tpc.BFP4_LAYOUT_ENV}_GATEUP", "dsmc:8:16:8")
    clean.setenv(f"{tpc.BFP4_LAYOUT_ENV}_GDN_OUT", "off")
    assert tpc.bfp4_layout("gateup", ttnn.bfloat4_b) == ("dsmc", 8, 16, 8)
    assert tpc.bfp4_layout("gdn_out", ttnn.bfloat4_b) is None


@pytest.mark.parametrize("bad", ["1d:84", "dsmc:2:8", "1d:0:8", "foo:1:2", "dsmc:a:b:c"])
def test_bad_override(clean, bad):
    clean.setenv(tpc.BFP4_LAYOUT_ENV, "1")
    clean.setenv(f"{tpc.BFP4_LAYOUT_ENV}_DOWN", bad)
    with pytest.raises(ValueError):
        tpc.bfp4_layout("down", ttnn.bfloat4_b)


def test_bad_flag_and_target(clean):
    clean.setenv(tpc.BFP4_LAYOUT_ENV, "2")
    with pytest.raises(ValueError):
        tpc.bfp4_layout("down", ttnn.bfloat4_b)
    with pytest.raises(ValueError):
        tpc.bfp4_layout("qkv", ttnn.bfloat4_b)


def test_lofi_gate(clean):
    assert not tpc.bfp4_layout_lofi(ttnn.bfloat4_b)  # flag off
    clean.setenv(tpc.BFP4_LAYOUT_ENV, "1")
    assert tpc.bfp4_layout_lofi(ttnn.bfloat4_b)
    assert not tpc.bfp4_layout_lofi(ttnn.bfloat8_b)
    clean.setenv(tpc.BFP4_LAYOUT_LOFI_ENV, "0")
    assert not tpc.bfp4_layout_lofi(ttnn.bfloat4_b)
