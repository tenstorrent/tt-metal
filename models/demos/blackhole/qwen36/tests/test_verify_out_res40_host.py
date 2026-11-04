# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of QWEN36_VERIFY_OUT_RES40 (no device): the flag-off path is untouched, and the RES40-out progcfg
keeps the production K loop (same in0_block_w as create_matmul_1d_decode_progcfg) while splitting N over the 10x4
RES40 grid. Device byte identity of the matmul itself: ttsim res40_bench (B == A bytewise for down / gdn_out / wo)."""
import types

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc


class _T:
    def __init__(self, shape, sharded=False):
        self.shape = shape
        self._sharded = sharded

    def is_sharded(self):
        return self._sharded


def _dar():
    return types.SimpleNamespace(dim=5120, res_mem="RES40")


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    tpc._RES40_PC_CACHE.clear()
    yield
    tpc._RES40_PC_CACHE.clear()


def test_flag_off_returns_none(monkeypatch):
    monkeypatch.delenv(tpc.VERIFY_OUT_RES40_ENV, raising=False)
    called = []
    monkeypatch.setattr(ttnn, "linear", lambda *a, **k: called.append(1))
    assert tpc.res40_out_matmul(_T([1, 1, 32, 2176]), _T([2176, 5120]), None, _dar()) is None
    assert not called


def test_no_decode_ar_or_bad_shapes(monkeypatch):
    monkeypatch.setenv(tpc.VERIFY_OUT_RES40_ENV, "1")
    monkeypatch.setattr(ttnn, "linear", lambda *a, **k: pytest.fail("must not run"))
    assert tpc.res40_out_matmul(_T([1, 1, 32, 768]), _T([768, 5120]), None, None) is None
    assert tpc.res40_out_matmul(_T([1, 1, 64, 768]), _T([768, 5120]), None, _dar()) is None  # > 1 tile of rows
    assert tpc.res40_out_matmul(_T([1, 1, 32, 768], sharded=True), _T([768, 5120]), None, _dar()) is None
    assert tpc.res40_out_matmul(_T([1, 1, 32, 768]), _T([768, 4096]), None, _dar()) is None  # n != dim


@pytest.mark.parametrize("k", [2176, 768])
def test_progcfg_keeps_k_loop(k):
    pc = tpc.res40_out_progcfg(k, 5120)
    ref = tpc.create_matmul_1d_decode_progcfg(32, k, 5120, num_cores=84, grid_w=11)
    assert pc.in0_block_w == ref.in0_block_w
    g = pc.compute_with_storage_grid_size
    assert (g.x, g.y) == (10, 4)
    assert pc.per_core_N == 4 and pc.per_core_M == 1 and pc.mcast_in0
    assert tpc.res40_out_progcfg(k, 5120 + 32) is None  # 161 tiles do not split over 40 cores


def test_flag_on_calls_linear_with_res40(monkeypatch):
    monkeypatch.setenv(tpc.VERIFY_OUT_RES40_ENV, "1")
    seen = {}

    def fake_linear(x, w, program_config=None, compute_kernel_config=None, memory_config=None):
        seen.update(x=x, pc=program_config, mc=memory_config)
        return "OUT"

    monkeypatch.setattr(ttnn, "linear", fake_linear)
    monkeypatch.setattr(ttnn, "reshape", lambda t, s: _T(list(s)))
    out = tpc.res40_out_matmul(_T([1, 32, 768]), _T([768, 5120]), "CKC", _dar())
    assert out == "OUT" and seen["mc"] == "RES40" and seen["x"].shape == [1, 1, 32, 768]
    assert seen["pc"].per_core_N == 4


class _S(_T):
    def __init__(self, shape, sharded=True):
        super().__init__(shape, sharded)

    def memory_config(self):
        return types.SimpleNamespace(memory_layout="WIDTH_SHARDED", shard_spec="40 cores [32, 128]")


def test_pack_flag_off_and_bad_shapes(monkeypatch):
    monkeypatch.setattr(ttnn, "to_memory_config", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.delenv(tpc.VERIFY_OUT_RES40_ENV, raising=False)
    assert tpc.res40_pack(_S([1, 1, 32, 5120]), _dar()) is None
    monkeypatch.setenv(tpc.VERIFY_OUT_RES40_ENV, "1")
    assert tpc.res40_pack(_S([1, 1, 32, 5120]), None) is None
    assert tpc.res40_pack(_S([1, 32, 5120]), _dar()) is None  # not 4D
    assert tpc.res40_pack(_S([1, 1, 64, 5120]), _dar()) is None  # > 1 tile of rows
    assert tpc.res40_pack(_S([1, 1, 32, 4096]), _dar()) is None  # n != dim
    assert tpc.res40_pack(_S([1, 1, 32, 5120], sharded=False), _dar()) is None  # interleaved: caller's path


def test_pack_flag_on_reshards_once_and_records(monkeypatch):
    monkeypatch.setenv(tpc.VERIFY_OUT_RES40_ENV, "1")
    monkeypatch.setattr(tpc, "_RES40_ENGAGED", {"x": 0})  # non-empty: no atexit hook from the test
    calls = []
    monkeypatch.setattr(ttnn, "to_memory_config", lambda t, mc: calls.append(mc) or "PACKED")
    assert tpc.res40_pack(_S([1, 1, 32, 5120]), _dar(), site="mlp_down_dsmc") == "PACKED"
    assert calls == ["RES40"] and tpc.res40_engaged()["mlp_down_dsmc"] == 1


def test_down_mode(monkeypatch):
    monkeypatch.delenv(tpc.VERIFY_OUT_RES40_DOWN_ENV, raising=False)
    assert tpc.verify_out_res40_down_mode() == "auto"
    for v in ("auto", "1d", "off", " 1D "):
        monkeypatch.setenv(tpc.VERIFY_OUT_RES40_DOWN_ENV, v)
        assert tpc.verify_out_res40_down_mode() == v.strip().lower()
    monkeypatch.setenv(tpc.VERIFY_OUT_RES40_DOWN_ENV, "dsmc")
    with pytest.raises(ValueError):
        tpc.verify_out_res40_down_mode()
