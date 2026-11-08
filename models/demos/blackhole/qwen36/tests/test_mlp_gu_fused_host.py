# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MLP_GU_FUSED host checks (no device): flag parsing, the split kernel's tile index math against the prefill
AGMM weight's column order, the per-core work split, the DRAM-sharded pipeline's layout rules, and which MLPs take the
fused decode path."""
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.mlp import mlp_gu_fused_decode_mode
from models.demos.blackhole.qwen36.tt.mlp_swiglu import pair_tile_ids, split_work
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu


def test_parse_flag():
    for v in (None, "", "0", "off", "false", "no", " OFF "):
        assert tpc.parse_mlp_gu_fused(v) is None
    for v in ("1", "on", "true", "1d", "1D", "yes"):
        assert tpc.parse_mlp_gu_fused(v) == "1d"
    assert tpc.parse_mlp_gu_fused("dsmc") == "dsmc"
    for bad in ("2d", "mc", "dram"):
        with pytest.raises(ValueError):
            tpc.parse_mlp_gu_fused(bad)


def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv(tpc.MLP_GU_FUSED_ENV, raising=False)
    assert tpc.mlp_gu_fused_mode() is None


@pytest.mark.parametrize("tp", [8, 4])
def test_pair_ids_follow_the_agmm_weight(tp):
    """Device d's output tile o of the split must be hidden columns [d*H/tp + 32*o, +32): exactly the rows of w2's
    row-parallel shard on device d, so the down projection reads the same hidden as r5."""
    hidden = 17408
    gate = torch.arange(hidden, dtype=torch.float64).repeat(2, 1)  # value = hidden column index
    up = gate + 1e6
    wgu = prepare_for_fused_swiglu(torch.cat([gate, up], dim=-1), ndev=tp, gate_is_first=True)
    per_dev = 2 * hidden // tp
    n_pairs = per_dev // 64
    cols = torch.arange(32, dtype=torch.float64)
    for d in range(tp):
        blk = wgu[0, d * per_dev : (d + 1) * per_dev]  # ShardTensorToMesh(dim=-1) slice
        for o in range(n_pairs):
            g, u = pair_tile_ids(o, n_pairs)
            want = cols + d * (hidden // tp) + 32 * o
            assert torch.equal(blk[32 * g : 32 * g + 32], want)
            assert torch.equal(blk[32 * u : 32 * u + 32], want + 1e6)


def test_pair_ids_multirow_and_halves():
    n = 5
    for o in range(3 * n):
        row, p = divmod(o, n)
        assert pair_tile_ids(o, n) == (row * 2 * n + 2 * p, row * 2 * n + 2 * p + 1)
        assert pair_tile_ids(o, n, pair=False) == (row * 2 * n + p, row * 2 * n + p + n)


def test_split_silu_mul_reproduces_swiglu():
    """torch model of the fused decode MLP on one device: x @ w_gate_up (AGMM layout) -> per-pair silu(g) * u == the
    r5 math silu(x @ w1_d) * (x @ w3_d) on that device's columns."""
    torch.manual_seed(0)
    tp, dim, hidden = 4, 64, 512
    w1 = torch.randn(hidden, dim, dtype=torch.float64)  # HF [out, in]
    w3 = torch.randn(hidden, dim, dtype=torch.float64)
    x = torch.randn(3, dim, dtype=torch.float64)
    wgu = prepare_for_fused_swiglu(torch.cat([w1.T, w3.T], dim=-1), ndev=tp, gate_is_first=True)
    per_dev, hd = 2 * hidden // tp, hidden // tp
    for d in range(tp):
        gu = x @ wgu[:, d * per_dev : (d + 1) * per_dev]
        n_pairs = per_dev // 64
        h = torch.empty(3, hd, dtype=torch.float64)
        for o in range(n_pairs):
            g, u = pair_tile_ids(o, n_pairs)
            h[:, 32 * o : 32 * o + 32] = torch.nn.functional.silu(gu[:, 32 * g : 32 * g + 32]) * gu[:, 32 * u : 32 * u + 32]
        ref = torch.nn.functional.silu(x @ w1[d * hd : (d + 1) * hd].T) * (x @ w3[d * hd : (d + 1) * hd].T)
        assert torch.allclose(h, ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("total, cores", [(68, 110), (68, 34), (68, 68), (136, 88), (5, 3), (1, 120)])
def test_split_work(total, cores):
    work = split_work(total, cores)
    assert len(work) == min(total, cores)
    assert sum(c for _, c in work) == total
    assert all(c >= 1 for _, c in work)
    assert max(c for _, c in work) - min(c for _, c in work) <= 1
    start = 0
    for s, c in work:
        assert s == start
        start += c


def test_dsmc_layout_rules_and_progcfg():
    # Qwen3.8 TP=8 fused gate|up: 136 tiles = 8 banks x 17, no padding.
    assert tpc._roundup(4352, 32 * tpc.DRAM_CORES) == 4352
    assert not tpc.dram_sharded_mc_workers_ok(17, 1)
    for w in (2, 3, 4):
        assert tpc.dram_sharded_mc_workers_ok(17, w), w
    assert tpc.dram_sharded_mc_layout(17, 4) == (4, 1, 1)
    cfg = tpc.create_dram_sharded_mc_progcfg(5120, 4352, 4, 4, 8)
    assert (cfg.in0_block_w, cfg.per_core_M, cfg.per_core_N) == (8, 1, 4)
    assert cfg.num_workers_per_dram_bank == 4 and cfg.use_multicore_pipeline
    with pytest.raises(ValueError):
        tpc.create_dram_sharded_mc_progcfg(5120, 4352, 3, 4, 8)  # 3 does not divide 136
    with pytest.raises(ValueError):
        tpc.create_dram_sharded_mc_progcfg(5120, 4352, 4, 1, 8)  # one worker per bank is not the pipeline
    with pytest.raises(ValueError):
        tpc.create_dram_sharded_mc_progcfg(5120, 4352, 4, 4, 0)


def test_decode_mode_gate():
    args = SimpleNamespace(mlp_gu_fused="1d", mlp_1d_decode=True)
    wgu = object()
    assert mlp_gu_fused_decode_mode(args, wgu) == "1d"
    assert mlp_gu_fused_decode_mode(SimpleNamespace(mlp_gu_fused="dsmc", mlp_1d_decode=True), wgu) == "dsmc"
    assert mlp_gu_fused_decode_mode(args, None) is None  # no AGMM weight (MoE shared expert): r5 path
    assert mlp_gu_fused_decode_mode(SimpleNamespace(mlp_gu_fused="1d", mlp_1d_decode=False), wgu) is None
    assert mlp_gu_fused_decode_mode(SimpleNamespace(mlp_gu_fused=None, mlp_1d_decode=True), wgu) is None
    assert mlp_gu_fused_decode_mode(SimpleNamespace(), wgu) is None  # args without the attribute (flag off)
    assert mlp_gu_fused_decode_mode(None, wgu) is None
