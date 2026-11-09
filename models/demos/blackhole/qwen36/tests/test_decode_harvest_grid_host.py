# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) of QWEN36_DECODE_HARVEST_GRID (tp_common.decode_harvest_grids / apply_decode_harvest_grids):
table selection per weight preset, explicit parsing, and that a re-grid keeps in0_block_w (bit-identical K loop) on
both the base config and the QWEN36_BFP4_LAYOUT 1D twin.

  PYTHONPATH=<worktree>:<ttnn or stub> python -m pytest models/demos/blackhole/qwen36/tests/test_decode_harvest_grid_host.py -q
"""
import types

import pytest

from models.demos.blackhole.qwen36.tt import tp_common as tpc

KNOBS = ("QWEN36_DECODE_HARVEST_GRID", "QWEN36_GPTQ_BFP4", "QWEN36_BFP4_LAYOUT")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    yield


def test_off_by_default():
    assert tpc.decode_harvest_grids() == {}


@pytest.mark.parametrize(
    "env,expect",
    [
        ({}, tpc.DECODE_HARVEST_GRID_BFP8),
        ({"QWEN36_GPTQ_BFP4": "1"}, tpc.DECODE_HARVEST_GRID_BFP4),
        ({"QWEN36_GPTQ_BFP4": "1", "QWEN36_BFP4_LAYOUT": "1"}, tpc.DECODE_HARVEST_GRID_R12),
    ],
)
def test_table_selection(monkeypatch, env, expect):
    monkeypatch.setenv("QWEN36_DECODE_HARVEST_GRID", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    want = {kv.split("=")[0]: tuple(int(t) for t in kv.split("=")[1].split("x")) for kv in expect.split(",")}
    assert tpc.decode_harvest_grids() == want


def test_r12_table_only_regriddable_sites():
    keys = {kv.split("=")[0] for kv in tpc.DECODE_HARVEST_GRID_R12.split(",")}
    assert keys == {"gdnin", "qkv"}  # down/gate-up are dsmc and wo/gdn_out RES40-out in the r12 production stack


def test_explicit_and_unknown(monkeypatch):
    monkeypatch.setenv("QWEN36_DECODE_HARVEST_GRID", "gdnin=8x5, qkv=11x3")
    assert tpc.decode_harvest_grids() == {"gdnin": (8, 5), "qkv": (11, 3)}
    monkeypatch.setenv("QWEN36_DECODE_HARVEST_GRID", "lmhead=8x5")
    with pytest.raises(ValueError):
        tpc.decode_harvest_grids()


def _pc(ibw, cores=44):
    return types.SimpleNamespace(in0_block_w=ibw, cores=cores)


def test_apply_keeps_ibw_and_regrids_bfp4_twin(monkeypatch):
    calls = []

    def fake(m, k, n, num_cores, fused_activation=None, grid_w=8, max_in0_block_w=8, **_):
        calls.append((k, n, num_cores, grid_w, max_in0_block_w))
        kt = k // 32
        ibw = max(d for d in range(1, max_in0_block_w + 1) if kt % d == 0)
        return types.SimpleNamespace(in0_block_w=ibw, cores=num_cores, grid_w=grid_w)

    monkeypatch.setattr(tpc, "create_matmul_1d_decode_progcfg", fake)
    args = types.SimpleNamespace(gdn_qkvz_decode_1d_progcfg=_pc(8, 84), gdn_qkvz_decode_1d_progcfg_bfp4=_pc(20, 44),
                                 attn_qkv_decode_1d_progcfg=_pc(8, 60))
    spec = {"gdnin": ("gdn_qkvz_decode_1d_progcfg", 5120, 2080, None, 8),
            "qkv": ("attn_qkv_decode_1d_progcfg", 5120, 2048, None, 8)}
    done = tpc.apply_decode_harvest_grids(args, 32, {"gdnin": (8, 5), "qkv": (11, 3)}, spec, 12)
    assert done == {"gdn_qkvz_decode_1d_progcfg": (8, 5), "gdn_qkvz_decode_1d_progcfg_bfp4": (8, 5),
                    "attn_qkv_decode_1d_progcfg": (11, 3)}
    assert args.gdn_qkvz_decode_1d_progcfg_bfp4.in0_block_w == 20 and args.gdn_qkvz_decode_1d_progcfg_bfp4.cores == 40
    assert args.gdn_qkvz_decode_1d_progcfg.in0_block_w == 8
    assert args.attn_qkv_decode_1d_progcfg.cores == 33 and args.attn_qkv_decode_1d_progcfg.in0_block_w == 8


def test_apply_skips_wide_and_missing(monkeypatch):
    monkeypatch.setattr(tpc, "create_matmul_1d_decode_progcfg", lambda *a, **k: pytest.fail("should not build"))
    args = types.SimpleNamespace()
    spec = {"qkv": ("attn_qkv_decode_1d_progcfg", 5120, 2048, None, 8)}
    assert tpc.apply_decode_harvest_grids(args, 32, {"qkv": (11, 3)}, spec, 12) == {}  # attr absent
    args.attn_qkv_decode_1d_progcfg = _pc(8, 60)
    assert tpc.apply_decode_harvest_grids(args, 32, {"qkv": (13, 3)}, spec, 12) == {}  # wider than the grid


def test_apply_rejects_ibw_change(monkeypatch):
    monkeypatch.setattr(tpc, "create_matmul_1d_decode_progcfg",
                        lambda m, k, n, num_cores, **kw: types.SimpleNamespace(in0_block_w=4, cores=num_cores))
    args = types.SimpleNamespace(attn_qkv_decode_1d_progcfg=_pc(8, 60))
    spec = {"qkv": ("attn_qkv_decode_1d_progcfg", 5120, 2048, None, 8)}
    with pytest.raises(ValueError):
        tpc.apply_decode_harvest_grids(args, 32, {"qkv": (11, 3)}, spec, 12)
