# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device) unit test of the TP=1 prefill matmul config builder (tp_common).

Covers the lane A review findings on c448dc16d0e: the measured 1D configs are emitted byte-identically for the P150
11x10 worker grid, the override is clamped to the device grid (F1), the TP gate is the `shape_overrides` feature key
resolved once by prefill_tuning(1) (F3/F4), and every other TP / shape / knob state takes the generic 2D config.

  pytest models/demos/blackhole/qwen36/tests/test_tp1_prefill_progcfg.py
"""

import types

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

S = 2048
# (K, N, weight tile bytes) of the three overridden shapes and their expected 1D configs on an 11x10 die.
SHAPES = {
    "attn_qkv": (
        5120,
        14336,
        tpc.TILE_BYTES_BFP8,
        dict(grid=(11, 9), per_core_N=5, in0_block_w=4, out_block_h=16, sub=(1, 1)),
    ),
    "gdn_qkvzab": (
        5120,
        16480,
        tpc.TILE_BYTES_BFP8,
        dict(grid=(11, 10), per_core_N=5, in0_block_w=4, out_block_h=16, sub=(1, 1)),
    ),
    "mlp_gate_up": (
        5120,
        17408,
        tpc.TILE_BYTES_BFP4,
        dict(grid=(11, 9), per_core_N=6, in0_block_w=8, out_block_h=16, sub=(1, 3)),
    ),
}


def _expected_1d(k, n, spec, act=None):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=spec["grid"],
        in0_block_w=spec["in0_block_w"],
        out_subblock_h=spec["sub"][0],
        out_subblock_w=spec["sub"][1],
        out_block_h=spec["out_block_h"],
        out_block_w=spec["per_core_N"],
        per_core_M=S // 32,
        per_core_N=spec["per_core_N"],
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=True,
    )


def _build(k, n, wtb, tuning, max_cols=11, act=None):
    return tpc.create_prefill_mlp_matmul_program_config(
        S, k, n, fused_activation=act, max_cols=max_cols, tuning=tuning, weight_tile_bytes=wtb
    )


def test_prefill_tuning_resolves_knob_once(monkeypatch):
    monkeypatch.setenv("QWEN36_TP1_PREFILL_OPT", "1")
    on = tpc.prefill_tuning(1)
    monkeypatch.setenv("QWEN36_TP1_PREFILL_OPT", "0")
    off = tpc.prefill_tuning(1)
    assert on["shape_overrides"] is True and off["shape_overrides"] is False
    # A copy per call: the module table never carries the key, and the resolved value does not follow later env changes.
    assert "shape_overrides" not in tpc._PREFILL_TUNING[1]
    monkeypatch.setenv("QWEN36_TP1_PREFILL_OPT", "1")
    assert off["shape_overrides"] is False
    for tp in (2, 4, 8):
        assert "shape_overrides" not in tpc.prefill_tuning(tp)
    # args-side accessor used by the GDN layer.
    assert tpc.tp1_prefill_opt(types.SimpleNamespace(prefill_tuning=on)) is True
    assert tpc.tp1_prefill_opt(types.SimpleNamespace(prefill_tuning=off)) is False
    assert tpc.tp1_prefill_opt(types.SimpleNamespace(prefill_tuning=tpc.prefill_tuning(4))) is False
    assert tpc.tp1_prefill_opt(types.SimpleNamespace()) is False


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_tp1_override_is_byte_identical_on_11x10(monkeypatch, name):
    monkeypatch.setenv("QWEN36_TP1_PREFILL_OPT", "1")
    monkeypatch.setattr(tpc, "prefill_grid_default", lambda: (8, 10))  # Blackhole P150 default prefill grid
    k, n, wtb, spec = SHAPES[name]
    act = ttnn.UnaryOpType.SILU if name == "mlp_gate_up" else None
    pc = _build(k, n, wtb, tpc.prefill_tuning(1), max_cols=11, act=act)
    assert isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig)
    assert pc.to_json() == _expected_1d(k, n, spec, act).to_json()
    # The 1D factory uses ceil(N tiles / per_core_N) cores; the entry's grid must hold them and fit the 11x10 die.
    cores = -(-(n // 32) // spec["per_core_N"])
    assert cores <= spec["grid"][0] * spec["grid"][1] <= 110


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_tp1_override_clamped_to_device_grid(monkeypatch, name):
    """F1: a narrower / shorter worker grid must fall back to the generic 2D config instead of a TT_FATAL at compile."""
    monkeypatch.setenv("QWEN36_TP1_PREFILL_OPT", "1")
    monkeypatch.setattr(tpc, "prefill_grid_default", lambda: (8, 10))
    k, n, wtb, _ = SHAPES[name]
    tuning_on = tpc.prefill_tuning(1)
    monkeypatch.setenv("QWEN36_TP1_PREFILL_OPT", "0")
    tuning_off = tpc.prefill_tuning(1)
    generic = _build(k, n, wtb, tuning_off, max_cols=10)
    assert isinstance(generic, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig)
    # 10 columns (a die exposing fewer than 11 compute columns): identical to the knob-off generic config.
    assert _build(k, n, wtb, tuning_on, max_cols=10).to_json() == generic.to_json()
    # Fewer rows than the entry needs (Wormhole-like 8-row prefill grid): generic as well.
    monkeypatch.setattr(tpc, "prefill_grid_default", lambda: (8, 8))
    assert isinstance(_build(k, n, wtb, tuning_on, max_cols=11), ttnn.MatmulMultiCoreReuseMultiCastProgramConfig)


def test_generic_everywhere_else(monkeypatch):
    monkeypatch.setenv("QWEN36_TP1_PREFILL_OPT", "1")
    monkeypatch.setattr(tpc, "prefill_grid_default", lambda: (8, 10))
    k, n, wtb, _ = SHAPES["attn_qkv"]
    # Knob off, TP=4 tuning, None tuning (frozen TP=4), other M: never the 1D override.
    monkeypatch.setenv("QWEN36_TP1_PREFILL_OPT", "0")
    off = tpc.prefill_tuning(1)
    monkeypatch.setenv("QWEN36_TP1_PREFILL_OPT", "1")
    for tuning in (off, tpc.prefill_tuning(4), tpc.prefill_tuning(2), None):
        assert isinstance(_build(k, n, wtb, tuning), ttnn.MatmulMultiCoreReuseMultiCastProgramConfig)
    on = tpc.prefill_tuning(1)
    for m in (128, 1024, 4096):
        pc = tpc.create_prefill_mlp_matmul_program_config(m, k, n, max_cols=11, tuning=on, weight_tile_bytes=wtb)
        assert isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig)
    # A non-overridden shape (MLP down) is generic with the knob on and identical to the knob-off config.
    assert (
        _build(17408, 5120, tpc.TILE_BYTES_BFP8, on).to_json()
        == _build(17408, 5120, tpc.TILE_BYTES_BFP8, off).to_json()
    )
