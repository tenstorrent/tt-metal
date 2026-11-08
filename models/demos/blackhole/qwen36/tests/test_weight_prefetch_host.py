# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of tt/weight_prefetch.py (QWEN36_WEIGHT_PREFETCH): flag parsing, the weight <-> 1D-matmul geometry on
the production TP=8 Blackhole decode shapes, the receiver/bank pairing, and the linear() routing (off == plain
ttnn.linear with the same arguments). No device: the ttnn calls that need one are faked.

Run: pytest models/demos/blackhole/qwen36/tests/test_weight_prefetch_host.py -q
"""
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt import weight_prefetch as wp

BH_GRID_W = 11  # Blackhole worker-grid width the TP=8 decode configs are shaped for (model_config.decode_grid_w)
NUM_BANKS = 8


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(wp.ENV, raising=False)
    monkeypatch.delenv(wp.GCB_KB_ENV, raising=False)
    wp._reset_for_tests()
    yield
    wp._reset_for_tests()


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, set()),
        ("", set()),
        ("0", set()),
        ("off", set()),
        ("1", {"gateup"}),
        ("gateup", {"gateup"}),
        ("down, gdn_in", {"down", "gdn_in"}),
        ("all", set(wp.TARGETS)),
        ("gateup,all", set(wp.TARGETS)),
    ],
)
def test_parse_targets(value, expected):
    assert wp.parse_targets(value) == frozenset(expected)


def test_parse_targets_rejects_unknown():
    with pytest.raises(ValueError, match="unknown target"):
        wp.parse_targets("gateup,lm_head")


def test_enabled_reads_env(monkeypatch):
    assert not wp.enabled("gateup")
    monkeypatch.setenv(wp.ENV, "1")
    assert wp.enabled("gateup") and not wp.enabled("down")
    monkeypatch.setenv(wp.ENV, "all")
    assert all(wp.enabled(t) for t in wp.TARGETS)


# Production TP=8 per-device decode shapes (Qwen3.8-27B: dim 5120, hidden 17408, GDN in 2080, attn qkv 2048,
# GDN/attn out K 768) with the exact model_config.py 1D decode progcfg calls.
_PROD = {
    # name: (k, n, num_cores, max_in0_block_w, fused_silu) -> expected (receivers, per_core_N, in0_block_w)
    "gateup": ((5120, 2176, 44, 16, True), (34, 2, 16)),
    "down": ((2176, 5120, 84, 8, False), (80, 2, 4)),
    "gdn_in": ((5120, 2080, 84, 8, False), (65, 1, 8)),
    "attn_qkv": ((5120, 2048, 60, 8, False), (64, 1, 8)),
    "out_proj": ((768, 5120, 84, 8, False), (80, 2, 8)),
}


@pytest.mark.parametrize("name", list(_PROD))
def test_production_shapes_fit_the_transport(name):
    (k, n, num_cores, ibw, silu), (receivers, per_core_N, in0_block_w) = _PROD[name]
    pc = tpc.create_matmul_1d_decode_progcfg(
        1,
        k,
        n,
        num_cores=num_cores,
        fused_activation=ttnn.UnaryOpType.SILU if silu else None,
        grid_w=BH_GRID_W,
        max_in0_block_w=ibw,
    )
    assert wp.progcfg_problem(pc) is None  # out_block_h/w default to per_core_M/N: one output block per worker
    geom, why = wp.geometry_for(pc, k, n, ttnn.bfloat8_b)
    assert why is None, why
    assert (geom.receivers, geom.per_core_N, geom.in0_block_w) == (receivers, per_core_N, in0_block_w)
    assert geom.receivers * geom.per_core_N == geom.n_tiles  # every worker owns a whole shard
    assert geom.block_count * geom.in0_block_w == geom.k_tiles
    assert geom.page_bytes == in0_block_w * per_core_N * 1088
    assert geom.receivers <= geom.cols * geom.rows


def test_plan_geometry_rejects_ragged_and_unaligned():
    tb = 1088
    assert wp.plan_geometry(5120, 65 * 32, 11, 8, 2, 8, tb)[0] is None  # 65 tiles not a multiple of per_core_N=2
    assert wp.plan_geometry(5120, 64 * 32, 11, 8, 2, 7, tb)[0] is None  # 160 K tiles not a multiple of 7
    assert wp.plan_geometry(5120 + 16, 2048, 11, 8, 2, 8, tb)[0] is None  # K not tile aligned
    assert wp.plan_geometry(5120, 100 * 32, 4, 4, 2, 8, tb)[0] is None  # 50 workers > 16-core grid
    geom, why = wp.plan_geometry(5120, 2176, 11, 4, 2, 16, tb)
    assert why is None and geom.receivers == 34


@pytest.mark.parametrize("receivers", [34, 64, 65, 80])
def test_bank_pairing_is_round_robin_and_covers_every_worker_once(receivers):
    seen = []
    for b in range(NUM_BANKS):
        pos = wp.bank_ring_positions(b, receivers, NUM_BANKS)
        assert all(p % NUM_BANKS == b for p in pos)  # ROUND_ROBIN_1D: shard s lives on bank s % num_banks
        seen += pos
    assert sorted(seen) == list(range(receivers))
    cores = [wp.ring_core(p, BH_GRID_W) for p in range(receivers)]
    assert cores[0] == (0, 0) and cores[BH_GRID_W] == (0, 1)  # row-major, as the matmul places output blocks
    assert len(set(cores)) == receivers


def test_gcb_size_holds_the_two_page_window():
    assert wp.gcb_size_bytes(34816, 72 * 1024) == 2 * 34816
    assert wp.gcb_size_bytes(8704, 72 * 1024) == 8 * 8704
    assert wp.gcb_size_bytes(34816, 0) == 2 * 34816  # never below the reader's window
    assert wp.gcb_size_bytes(100_000, 72 * 1024) == 200_000


def test_per_device_kn():
    # shard_w: torch [out, in] -> [in, out], split on dim (-1 column-parallel, 0 row-parallel)
    assert wp.per_device_kn((17408, 5120), -1, 8) == (5120, 2176)
    assert wp.per_device_kn((5120, 17408), 0, 8) == (2176, 5120)


def test_tile_bytes():
    assert wp.tile_bytes(ttnn.bfloat8_b) == 1088
    assert wp.tile_bytes(ttnn.bfloat4_b) == 576
    assert wp.tile_bytes(ttnn.bfloat16) == 2048


class _FakeTtnn(SimpleNamespace):
    """Records ttnn.linear / prefetch_and_linear calls; everything else forwards to the real module."""

    def __init__(self, supported=True, block_count_raises=False):
        calls = []
        self_ref = self

        def linear(x, w, **kw):
            calls.append(("linear", x, w, kw))
            return "plain_out"

        def prefetch_and_linear(x, w, **kw):
            calls.append(("prefetch", x, w, kw))
            return "pf_out"

        def block_count(pc, w, gcb):
            if block_count_raises:
                raise RuntimeError("weight K must be divisible")
            return self_ref.expect_block_count

        experimental = SimpleNamespace(
            tensor_prefetcher_matmul=SimpleNamespace(prefetch_and_linear=prefetch_and_linear),
            is_tensor_prefetcher_supported=lambda mesh: supported,
            start_tensor_prefetcher=lambda mesh: calls.append(("start", mesh)),
            create_global_circular_buffer_for_tensor_prefetcher=lambda mesh, pairs, size, **kw: ("gcb", size),
            tensor_prefetcher_block_count_for_matmul_1d=block_count,
        )
        super().__init__(linear=linear, experimental=experimental, deallocate=lambda t: calls.append(("dealloc", t)))
        self.calls = calls
        self.expect_block_count = None

    def __getattr__(self, name):  # dtypes, CoreRangeSet, ... from the real ttnn
        return getattr(ttnn, name)


class _FakeMesh:
    def dram_grid_size(self):
        return SimpleNamespace(x=NUM_BANKS, y=1)


def _gateup_progcfg():
    return tpc.create_matmul_1d_decode_progcfg(
        1, 5120, 2176, num_cores=44, fused_activation=ttnn.UnaryOpType.SILU, grid_w=BH_GRID_W, max_in0_block_w=16
    )


def test_linear_is_plain_ttnn_linear_when_nothing_is_registered(monkeypatch):
    fake = _FakeTtnn()
    monkeypatch.setattr(wp, "ttnn", fake)
    pc = object()
    out = wp.linear("x", "w", pc, compute_kernel_config="ckc", memory_config="mc")
    assert out == "plain_out"
    assert fake.calls == [
        ("linear", "x", "w", {"program_config": pc, "compute_kernel_config": "ckc", "memory_config": "mc"})
    ]


def test_load_prefetched_is_inert_when_target_off(monkeypatch):
    fake = _FakeTtnn()
    monkeypatch.setattr(wp, "ttnn", fake)
    monkeypatch.setenv(wp.ENV, "down")  # gateup not selected
    shard_calls = []
    w = wp.load_prefetched(
        "gateup", "mlp.gate_proj", _FakeMesh(), SimpleNamespace(shape=(17408, 5120)), -1, ttnn.bfloat8_b, "c",
        _gateup_progcfg(), 8, replaces_primary=True, shard_fn=lambda *a, **k: shard_calls.append(1),
    )  # fmt: skip
    assert w is None and not shard_calls and fake.calls == []  # never started the prefetcher


def _load_gateup(monkeypatch, fake, **kw):
    monkeypatch.setattr(wp, "ttnn", fake)
    monkeypatch.setenv(wp.ENV, "1")
    fake.expect_block_count = 10  # 160 K tiles / in0_block_w 16
    shard_args = {}

    def shard_fn(t, mesh, dim, memory_config, cache_path, dtype):
        shard_args.update(dim=dim, memory_config=memory_config, cache_path=cache_path, dtype=dtype)
        return SimpleNamespace(name="w_rc")

    pc = _gateup_progcfg()
    w = wp.load_prefetched(
        "gateup", "mlp.gate_proj", _FakeMesh(), SimpleNamespace(shape=(17408, 5120)), -1, ttnn.bfloat8_b,
        "cache/mlp.gate_proj.weight.tp", pc, 8, replaces_primary=True, shard_fn=shard_fn, **kw,
    )  # fmt: skip
    return w, pc, shard_args


def test_gateup_loads_receiver_contiguous_and_routes_through_the_prefetcher(monkeypatch):
    fake = _FakeTtnn()
    w, pc, shard_args = _load_gateup(monkeypatch, fake)
    assert w is not None and wp.is_registered(w)
    assert shard_args["cache_path"] == "cache/mlp.gate_proj.weight.tp.pf34"  # distinct from the interleaved cache
    assert shard_args["dim"] == -1 and shard_args["dtype"] == ttnn.bfloat8_b
    assert fake.calls[0][0] == "start"
    out = wp.linear("x", w, pc, compute_kernel_config="ckc", memory_config="mc")
    assert out == "pf_out"
    kind, x, wt, kw = fake.calls[-1]
    assert (kind, x, wt) == ("prefetch", "x", w)
    assert kw["program_config"] is pc  # the model's own progcfg: identical compute
    assert kw["global_cb"] == ("gcb", 2 * 34816)
    assert kw["compute_kernel_config"] == "ckc" and kw["memory_config"] == "mc"


def test_replaced_primary_with_another_progcfg_raises(monkeypatch):
    w, pc, _ = _load_gateup(monkeypatch, _FakeTtnn())
    with pytest.raises(RuntimeError, match="unregistered program config"):
        wp.linear("x", w, None)


def test_contract_failure_falls_back_at_load(monkeypatch):
    fake = _FakeTtnn(block_count_raises=True)
    w, _, _ = _load_gateup(monkeypatch, fake)
    assert w is None  # caller then loads the interleaved weight
    assert ("dealloc", SimpleNamespace(name="w_rc")) in fake.calls
    assert not wp._REGISTRY


def test_unsupported_mesh_disables_without_loading(monkeypatch):
    fake = _FakeTtnn(supported=False)
    w, _, shard_args = _load_gateup(monkeypatch, fake)
    assert w is None and not shard_args
    assert not any(c[0] == "start" for c in fake.calls)


def test_twin_routes_only_the_registered_progcfg(monkeypatch):
    fake = _FakeTtnn()
    monkeypatch.setattr(wp, "ttnn", fake)
    monkeypatch.setenv(wp.ENV, "down")
    fake.expect_block_count = 17  # 68 K tiles / in0_block_w 4
    pc = tpc.create_matmul_1d_decode_progcfg(1, 2176, 5120, num_cores=84, grid_w=BH_GRID_W)
    primary = SimpleNamespace(name="w2_interleaved")
    twin = wp.load_prefetched(
        "down", "mlp.down_proj", _FakeMesh(), SimpleNamespace(shape=(5120, 17408)), 0, ttnn.bfloat8_b, "c", pc, 8,
        shard_fn=lambda *a, **k: SimpleNamespace(name="w2_rc"),
    )  # fmt: skip
    assert twin is not None and not wp.is_registered(twin)
    assert wp.attach_twin(_FakeMesh(), primary, twin, pc)
    assert wp.linear("h", primary, pc) == "pf_out"
    assert fake.calls[-1][2] is twin  # the receiver-contiguous copy is what streams
    assert wp.linear("h", primary, "prefill_pc") == "plain_out"  # prefill keeps the interleaved weight
    assert fake.calls[-1][2] is primary


def test_twin_is_freed_when_registration_fails(monkeypatch):
    fake = _FakeTtnn(block_count_raises=True)
    monkeypatch.setattr(wp, "ttnn", fake)
    monkeypatch.setenv(wp.ENV, "down")
    pc = tpc.create_matmul_1d_decode_progcfg(1, 2176, 5120, num_cores=84, grid_w=BH_GRID_W)
    primary = SimpleNamespace(name="w2")
    twin = wp.load_prefetched(
        "down", "mlp.down_proj", _FakeMesh(), SimpleNamespace(shape=(5120, 17408)), 0, ttnn.bfloat8_b, "c", pc, 8,
        shard_fn=lambda *a, **k: SimpleNamespace(name="w2_rc"),
    )  # fmt: skip
    assert not wp.attach_twin(_FakeMesh(), primary, twin, pc)
    assert ("dealloc", twin) in fake.calls
    assert wp.linear("h", primary, pc) == "plain_out"


def test_attach_twin_none_is_a_noop():
    assert wp.attach_twin(_FakeMesh(), SimpleNamespace(), None, None) is False
