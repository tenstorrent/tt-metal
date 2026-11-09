# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) of QWEN36_PREFILL_SHORT_DECODE_MM (tp_common): flag parsing, the row window and site filter,
the short 1D progcfg shapes, and that the two AGMM funnels dispatch to all_gather_async + 1D matmul (+ split-SwiGLU)
only inside the window, with the AGMM untouched otherwise.

Run (worktree first on PYTHONPATH, ttnn from the built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_prefill_short_decode_mm_host.py -q
"""
import math
import types

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

KNOBS = ("QWEN36_PREFILL_SHORT_DECODE_MM", "QWEN36_PREFILL_SHORT_DECODE_MM_SITES", "QWEN36_PREFILL_AGMM_TUNED")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(tpc, "_short_mm_progcfgs", {})
    monkeypatch.setattr(tpc, "_short_swiglu", {})
    yield


# ------------------------------------------------------------------ flag parsing / window
@pytest.mark.parametrize("raw,cap", [("", 0), ("0", 0), ("off", 0), ("1", 256), ("on", 256), ("128", 128), ("64", 64)])
def test_cap(monkeypatch, raw, cap):
    monkeypatch.setenv("QWEN36_PREFILL_SHORT_DECODE_MM", raw)
    assert tpc.prefill_short_decode_mm_cap() == cap


@pytest.mark.parametrize("raw", ["32", "100", "512", "-64"])
def test_cap_rejects(monkeypatch, raw):
    monkeypatch.setenv("QWEN36_PREFILL_SHORT_DECODE_MM", raw)
    with pytest.raises(ValueError):
        tpc.prefill_short_decode_mm_cap()


def test_off_by_default():
    for rows in (64, 128, 256):
        for site in ("inproj", "gateup"):
            assert not tpc.short_decode_mm_ok(rows, site)


def test_window(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_SHORT_DECODE_MM", "128")
    assert not tpc.short_decode_mm_ok(32, "inproj")  # <= one tile: the existing decode-config path, unchanged
    assert tpc.short_decode_mm_ok(64, "inproj")
    assert tpc.short_decode_mm_ok(128, "gateup")
    assert not tpc.short_decode_mm_ok(160, "inproj")
    assert not tpc.short_decode_mm_ok(8192, "gateup")


def test_sites(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_SHORT_DECODE_MM", "1")
    monkeypatch.setenv("QWEN36_PREFILL_SHORT_DECODE_MM_SITES", "inproj")
    assert tpc.short_decode_mm_ok(128, "inproj")
    assert not tpc.short_decode_mm_ok(128, "gateup")
    monkeypatch.setenv("QWEN36_PREFILL_SHORT_DECODE_MM_SITES", "gateup,bogus")
    with pytest.raises(ValueError):
        tpc.short_decode_mm_ok(128, "gateup")


# ------------------------------------------------------------------ progcfg shapes
@pytest.fixture
def pc_recorder(monkeypatch):
    monkeypatch.setattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", lambda **kw: dict(kw), raising=False)


# Qwen3.8-27B TP8 per-device in-proj / gate-up shapes: (K, N)
SHAPES = {"gdn_qkvzab": (5120, 2080), "attn_qkv": (5120, 2048), "gateup_pair": (5120, 4352), "gdn_out_colpar": (6144, 640)}


@pytest.mark.parametrize("rows", [64, 128, 256])
@pytest.mark.parametrize("name", list(SHAPES))
def test_progcfg(pc_recorder, rows, name):
    k, n = SHAPES[name]
    pc = tpc.short_decode_mm_progcfg(rows, k, n, grid_w=12)
    cols, grid_rows = pc["compute_with_storage_grid_size"]
    assert cols * grid_rows <= 12 * 7
    assert pc["per_core_M"] == rows // 32
    assert pc["mcast_in0"] is True
    k_tiles = k // 32
    assert k_tiles % pc["in0_block_w"] == 0 and pc["in0_block_w"] <= 8
    assert pc["per_core_N"] * cols * grid_rows >= math.ceil(n / 32)  # covers N
    assert pc["out_subblock_h"] * pc["out_subblock_w"] <= 4  # fp32 dest acc
    assert pc["per_core_N"] % pc["out_subblock_w"] == 0 and pc["per_core_M"] % pc["out_subblock_h"] == 0
    # L1 budget of the in0 double buffer (bf16 tiles)
    assert pc["per_core_M"] * pc["in0_block_w"] * 2 * 2048 <= 256 * 1024


def test_progcfg_cached(pc_recorder):
    a = tpc.short_decode_mm_progcfg(128, 5120, 2080, grid_w=12)
    assert tpc.short_decode_mm_progcfg(128, 5120, 2080, grid_w=12) is a


# ------------------------------------------------------------------ funnel dispatch
class _T:
    def __init__(self, shape):
        self.shape = tuple(shape)


@pytest.fixture
def fake_ops(monkeypatch, pc_recorder):
    calls = []
    exp = types.SimpleNamespace(
        all_gather_minimal_matmul_async=lambda **kw: (calls.append(("agmm", kw)) or [_T((1, 1, kw["input_tensor"].shape[-2], 7))]),
        all_gather_async=lambda x, **kw: (calls.append(("ag", kw)) or _T((1, 1, x.shape[-2], x.shape[-1] * 8))),
    )
    monkeypatch.setattr(ttnn, "experimental", exp, raising=False)
    monkeypatch.setattr(ttnn, "reshape", lambda x, s, **kw: _T(s), raising=False)
    monkeypatch.setattr(ttnn, "deallocate", lambda x: calls.append(("free", x)), raising=False)
    monkeypatch.setattr(
        ttnn, "linear", lambda a, w, **kw: (calls.append(("linear", kw)) or _T((1, 1, a.shape[-2], w.shape[-1]))), raising=False
    )
    monkeypatch.setattr(tpc, "agmm_tuned_config", lambda *a, **k: None)
    mesh = types.SimpleNamespace(compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=12, y=10))
    ccl = types.SimpleNamespace(
        mesh_device=mesh,
        get_and_cycle_ag_semaphore_handles=lambda *a: "sem",
        get_and_cycle_barrier_semaphore_handle=lambda *a: "bar",
    )
    return calls, ccl


def _kinds(calls):
    return [c[0] for c in calls if c[0] != "free"]


@pytest.mark.parametrize("rows,short", [(32, False), (64, True), (128, True), (256, True), (288, False), (2048, False)])
def test_inproj_dispatch(monkeypatch, fake_ops, rows, short):
    calls, ccl = fake_ops
    monkeypatch.setenv("QWEN36_PREFILL_SHORT_DECODE_MM", "1")
    out = tpc.all_gather_matmul_prefill(_T((1, rows, 640)), _T((5120, 2080)), ccl, "ckc", "ring", out_memory_config="mc")
    if short:
        assert _kinds(calls) == ["ag", "linear"]
        lin = calls[[c[0] for c in calls].index("linear")][1]
        assert lin["program_config"]["per_core_M"] == rows // 32
        assert lin["compute_kernel_config"] == "ckc" and lin["memory_config"] == "mc"
        assert out.shape == (1, 1, rows, 2080)
        ag = calls[0][1]
        assert ag["dim"] == 3 and ag["memory_config"] == ttnn.L1_MEMORY_CONFIG
    else:
        assert _kinds(calls) == ["agmm"]


def test_inproj_flag_off_is_agmm(fake_ops):
    calls, ccl = fake_ops
    tpc.all_gather_matmul_prefill(_T((1, 128, 640)), _T((5120, 2080)), ccl, "ckc", "ring")
    assert _kinds(calls) == ["agmm"]


def test_inproj_fused_activation_rides_the_progcfg(monkeypatch, fake_ops):
    calls, ccl = fake_ops
    monkeypatch.setenv("QWEN36_PREFILL_SHORT_DECODE_MM", "1")
    tpc.all_gather_matmul_prefill(_T((1, 128, 640)), _T((5120, 2080)), ccl, "ckc", "ring", fused_activation="SILU")
    lin = [c for c in calls if c[0] == "linear"][0][1]
    assert lin["program_config"]["fused_activation"] == "SILU"


def test_gateup_dispatch(monkeypatch, fake_ops):
    calls, ccl = fake_ops
    monkeypatch.setenv("QWEN36_PREFILL_SHORT_DECODE_MM", "1")
    split_calls = []

    class FakeSplit:
        def __init__(self, mesh, pair=True, max_cores=None):
            assert pair is True

        def __call__(self, gu, memory_config=None):
            split_calls.append((gu.shape, memory_config))
            return _T((1, 1, gu.shape[-2], gu.shape[-1] // 2))

    from models.demos.blackhole.qwen36.tt import mlp_swiglu

    monkeypatch.setattr(mlp_swiglu, "SwiGLUSplit", FakeSplit)
    out = tpc.all_gather_swiglu_prefill(_T((1, 128, 640)), _T((5120, 4352)), ccl, "ckc", "ring", out_memory_config="dram")
    assert _kinds(calls) == ["ag", "linear"]
    assert split_calls == [((1, 1, 128, 4352), "dram")]
    assert out.shape == (1, 1, 128, 2176)
    # the split kernel instance is reused across calls on the same mesh
    tpc.all_gather_swiglu_prefill(_T((1, 128, 640)), _T((5120, 4352)), ccl, "ckc", "ring")
    assert len(tpc._short_swiglu) == 1


def test_gateup_site_off_keeps_agmm(monkeypatch, fake_ops):
    calls, ccl = fake_ops
    monkeypatch.setenv("QWEN36_PREFILL_SHORT_DECODE_MM", "1")
    monkeypatch.setenv("QWEN36_PREFILL_SHORT_DECODE_MM_SITES", "inproj")
    tpc.all_gather_swiglu_prefill(_T((1, 128, 640)), _T((5120, 4352)), ccl, "ckc", "ring")
    assert _kinds(calls) == ["agmm"]
