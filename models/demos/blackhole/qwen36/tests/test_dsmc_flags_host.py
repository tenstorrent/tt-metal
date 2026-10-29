# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host tests (ttnn importable, no device) for QWEN36_DSHARD_MC: flag parsing, the multi-core DRAM-sharded decode
configs on the Qwen3.8-27B TP=8 per-device shapes (output shard width divides N, output core counts, the factory's
workers-per-bank layout rules, activation coverage of K), the init guards (dtype, dsmc + weight prefetch on one
target) and the MLP routing helper (off == today's path). Also pins the gdn_out prefetch twin dtype fix.

Run: pytest models/demos/blackhole/qwen36/tests/test_dsmc_flags_host.py -q
"""
import inspect
import math
import os
from types import SimpleNamespace

import pytest

ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.qwen36.tt import tp_common as tpc  # noqa: E402
from models.demos.blackhole.qwen36.tt import weight_prefetch as wp  # noqa: E402
from models.demos.blackhole.qwen36.tt.mlp import mlp_dsmc_targets  # noqa: E402

TILE = 32
# Qwen3.8-27B per-device (TP=8) decode projections: target -> (K, N)
DIM, HIDDEN_TP, QKVZAB_TP, VALUE_TP = 5120, 17408 // 8, 2060, 6144 // 8
SHAPES = {
    "gateup": (DIM, HIDDEN_TP),
    "down": (HIDDEN_TP, DIM),
    "gdn_in": (DIM, QKVZAB_TP),
    "gdn_out": (VALUE_TP, DIM),
}
# (output per_core_N, output cores, weight tiles per DRAM bank, default workers per bank)
EXPECTED = {
    "gateup": (2, 34, 9, 2),
    "down": (4, 40, 20, 3),
    "gdn_in": (5, 13, 9, 2),
    "gdn_out": (4, 40, 20, 3),
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith(tpc.DSHARD_MC_ENV):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv(wp.ENV, raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, set()),
        ("", set()),
        ("0", set()),
        ("off", set()),
        ("all", set(tpc.DSHARD_MC_TARGETS)),
        ("1", set(tpc.DSHARD_MC_TARGETS)),
        ("gateup", {"gateup"}),
        ("gateup, down", {"gateup", "down"}),
        ("gdn_in,gdn_out", {"gdn_in", "gdn_out"}),
    ],
)
def test_parse(value, expected):
    assert tpc.parse_dshard_mc_targets(value) == frozenset(expected)


def test_parse_rejects_unknown():
    with pytest.raises(ValueError, match="unknown target"):
        tpc.parse_dshard_mc_targets("gateup,attn_qkv")


def test_default_off(_clean):
    assert tpc.dshard_mc_targets() == frozenset()
    assert not any(tpc.dshard_mc_enabled(t) for t in tpc.DSHARD_MC_TARGETS)


@pytest.mark.parametrize("target", sorted(SHAPES))
def test_target_progcfg(target):
    k, n = SHAPES[target]
    pcn, cores, bank_tiles, workers = EXPECTED[target]
    n_tiles = math.ceil(n / TILE)
    assert tpc.dshard_mc_bank_tiles(n) == bank_tiles
    assert tpc.dshard_mc_out_per_core_n(n_tiles) == pcn
    assert n_tiles % pcn == 0 and n_tiles // pcn == cores  # the op builds ceil(N / per_core_N) output cores
    assert tpc.dshard_mc_workers(target, bank_tiles) == workers
    pc = tpc.create_dshard_mc_target_progcfg(target, k, n)
    assert pc.per_core_N == pcn
    assert pc.per_core_M == 1
    assert pc.in0_block_w == tpc.DSHARD_MC_DEFAULT_IBW
    assert pc.num_workers_per_dram_bank == workers
    assert pc.use_multicore_pipeline


@pytest.mark.parametrize("bank_tiles", [9, 20])
@pytest.mark.parametrize("workers", [2, 3, 4])
def test_swept_workers_satisfy_factory_rules(bank_tiles, workers):
    cn, ck, passes = tpc.dshard_mc_layout(bank_tiles, workers)
    assert ck == 1 or passes == 1
    assert bank_tiles // cn >= passes
    assert tpc.dshard_mc_workers_ok(bank_tiles, workers)


def test_one_worker_is_not_the_pipeline():
    assert not tpc.dshard_mc_workers_ok(9, 1)


def test_worker_and_block_overrides(_clean):
    _clean.setenv("QWEN36_DSHARD_MC_W_DOWN", "4")
    _clean.setenv("QWEN36_DSHARD_MC_IBW_DOWN", "8")
    pc = tpc.create_dshard_mc_target_progcfg("down", *SHAPES["down"])
    assert pc.num_workers_per_dram_bank == 4 and pc.in0_block_w == 8
    _clean.setenv("QWEN36_DSHARD_MC_W_GATEUP", "1")
    with pytest.raises(ValueError, match="not a valid multi-core layout"):
        tpc.create_dshard_mc_target_progcfg("gateup", *SHAPES["gateup"])
    _clean.setenv("QWEN36_DSHARD_MC_IBW_GDN_IN", "0")
    with pytest.raises(ValueError, match="at least 1"):
        tpc.dshard_mc_ibw("gdn_in")


def _shard(memcfg):
    spec = memcfg.shard_spec
    return spec.num_cores(), spec.shape[1] // TILE


def test_activation_shards_cover_k():
    # gate/up and gdn_in: act_shard_hidden (the decode ff/attn norm output)
    cores, width = _shard(tpc.create_activation_shard_config(DIM))
    assert (cores, width) == (32, 5) and cores * width == DIM // TILE
    # gdn_out: act_shard_gdn_value
    cores, width = _shard(tpc.create_activation_shard_config(VALUE_TP))
    assert cores * width == VALUE_TP // TILE
    # down fed by the 1D gate/up (interleaved hidden): mlp_w2_dsmc_act_memcfg
    cores, width = _shard(tpc.create_activation_shard_config(HIDDEN_TP))
    assert (cores, width) == (4, 17)
    # down fed by the dsmc gate/up: the gate/up output shard (2 tiles x 34 cores) covers down's K
    pcn, out_cores, *_ = EXPECTED["gateup"]
    assert pcn * out_cores == HIDDEN_TP // TILE and (HIDDEN_TP // TILE) % pcn == 0


def test_check_weight_guards(_clean):
    for t in tpc.DSHARD_MC_TARGETS:
        tpc.dshard_mc_check_weight(t, ttnn.bfloat4_b)
        tpc.dshard_mc_check_weight(t, ttnn.bfloat8_b)
    with pytest.raises(ValueError, match="bfloat4_b"):
        tpc.dshard_mc_check_weight("down", ttnn.bfloat16)
    _clean.setenv(wp.ENV, "down")
    with pytest.raises(ValueError, match="both name"):
        tpc.dshard_mc_check_weight("down", ttnn.bfloat4_b)
    tpc.dshard_mc_check_weight("gateup", ttnn.bfloat4_b)  # a different prefetch target is fine


def _args(**kw):
    base = dict(num_devices=8, mlp_1d_decode=True, mlp_w1_dsmc_progcfg=None, mlp_w2_dsmc_progcfg=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_mlp_routing():
    assert mlp_dsmc_targets(_args()) == (False, False)  # flag off: today's path
    assert mlp_dsmc_targets(None) == (False, False)
    on = _args(mlp_w1_dsmc_progcfg=object(), mlp_w2_dsmc_progcfg=object())
    assert mlp_dsmc_targets(on) == (True, True)
    assert mlp_dsmc_targets(on, dense=False) == (False, False)  # MoE shared expert keeps its layout
    assert mlp_dsmc_targets(_args(mlp_w2_dsmc_progcfg=object())) == (False, True)
    assert mlp_dsmc_targets(_args(mlp_w1_dsmc_progcfg=object(), mlp_1d_decode=False)) == (False, False)
    assert mlp_dsmc_targets(_args(mlp_w1_dsmc_progcfg=object(), num_devices=1)) == (False, False)


def test_gdn_out_prefetch_twin_uses_out_dtype():
    from models.demos.blackhole.qwen36.tt.gdn import tp as gdn_tp

    src = inspect.getsource(gdn_tp.load_gdn_weights_tp)
    assert '"gdn_out", "gdn.out_proj", mesh, sd[P + "out_proj.weight"], 0, _out_dtype,' in src
