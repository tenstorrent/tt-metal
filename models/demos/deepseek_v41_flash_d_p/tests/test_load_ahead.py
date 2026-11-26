# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU: tt/load_ahead.py (LOAD_THREADS) - the pool reads every layer's files, in layer order, within LOAD_AHEAD_GB
past the layer being built, stops cleanly; dense_dtype's statistics on the pool equal the inline ones, in the same
BFP8_STATS order (no device: the bfp8 round trip is stubbed, since a host ttnn.from_torch opens the cluster)."""

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from models.demos.deepseek_v41_flash_d_p.tt import common, load_ahead

LAYERS = list(range(6))
SIZE = 3 * load_ahead.PIECE + 12345  # bytes of each layer's fake expert file (4 pieces)


class _Ck:
    """The Checkpoint interface dense_ranges reads: weight_map, info, path (two small tensors a layer + an expert)."""

    def __init__(self, path, fname):
        self.path, self.weight_map, self._info = str(path), {}, {}
        off = 0
        for i in LAYERS:
            for name, n in (("attn.wq_a.weight", 4096), ("attn.wo_b.weight", 8192), ("ffn.experts.0.w1.weight", 999)):
                self.weight_map[f"layers.{i}.{name}"] = fname
                self._info[f"layers.{i}.{name}"] = (torch.uint8, [n], off, n)
                off += n
        (path / fname).write_bytes(bytes(off))

    def info(self, name):
        return self._info[name]


@pytest.fixture
def model(tmp_path, monkeypatch):
    files = {}
    for i in LAYERS:
        f = tmp_path / f"L{i:02d}.moe_blk.k.w_gu.tensorbin"
        f.write_bytes(bytes(SIZE))
        files[i] = [f]
    monkeypatch.setattr(load_ahead, "expert_files", lambda m, i: files[i])
    return SimpleNamespace(ck=_Ck(tmp_path, "shard.safetensors"), weight_cache=tmp_path)


def test_dense_ranges_skip_experts_and_merge(model):
    spans = load_ahead.dense_ranges(model.ck, 2)
    # (layer 2's wq_a and wo_b are adjacent in the shard: one span; its expert tensor is not read)
    _, _, off_a, _ = model.ck.info("layers.2.attn.wq_a.weight")
    assert spans == [(f"{model.ck.path}/shard.safetensors", off_a, 4096 + 8192)]


@pytest.mark.parametrize("budget_pieces", [2, 100])
def test_reads_every_layer_in_order_within_budget(model, monkeypatch, budget_pieces):
    read_order, peak = [], {"ahead": 0}
    real = load_ahead.LoadAhead._read
    lock = threading.Lock()

    def spy(self, k, path, off, n):
        with lock:
            read_order.append(k)
            with self.cv:
                past = sum(b for j, b in self.ahead.items() if j > self.at + 1)
            peak["ahead"] = max(peak["ahead"], past)
        real(self, k, path, off, n)

    monkeypatch.setattr(load_ahead.LoadAhead, "_read", spy)
    budget = budget_pieces * load_ahead.PIECE
    ahead = load_ahead.LoadAhead(model, LAYERS, 4, budget)
    for i in LAYERS:
        ahead.building(i)
        time.sleep(0.05)  # (the layer's device build)
    t0 = time.time()
    while len(ahead.read_layers) < len(LAYERS) and time.time() - t0 < 10:
        time.sleep(0.01)
    ahead.close()
    assert ahead.read_layers == set(LAYERS)
    assert read_order == sorted(read_order), "reads start in layer order"
    assert peak["ahead"] <= budget, (peak, budget)
    # (a layer the build has passed is not read on: what it skipped it read itself)
    assert ahead.read_bytes <= len(LAYERS) * PER_LAYER


PER_LAYER = SIZE + 4096 + 8192


def _wait_bytes(ahead, n, timeout=20.0):
    t0 = time.time()
    while ahead.read_bytes < n and time.time() - t0 < timeout:
        time.sleep(0.01)
    time.sleep(0.2)  # (anything past n would have been read by now)
    return ahead.read_bytes


def test_reads_all_while_build_is_slow(model):
    ahead = load_ahead.LoadAhead(model, LAYERS, 4, 100 * load_ahead.PIECE)
    ahead.building(0)
    got = _wait_bytes(ahead, len(LAYERS) * PER_LAYER)
    ahead.close()
    assert got == len(LAYERS) * PER_LAYER


def test_stops_at_budget(model):
    # the budget counts every byte past the layer being built, the next layer's too (which is read regardless)
    extra = 2 * load_ahead.PIECE
    ahead = load_ahead.LoadAhead(model, LAYERS, 4, PER_LAYER + extra)
    ahead.building(0)
    # layers 0 and 1; then layer 2's pieces while layer 1 + them stay within the budget: two of its 64 MiB pieces
    got = _wait_bytes(ahead, 2 * PER_LAYER + extra)
    assert got == 2 * PER_LAYER + extra, got
    ahead.building(1)  # (layer 2 is now the next layer: the rest of it, then two pieces of layer 3)
    got = _wait_bytes(ahead, 3 * PER_LAYER + extra)
    ahead.close()
    assert got == 3 * PER_LAYER + extra, got


def test_close_before_reads_finish(model):
    ahead = load_ahead.LoadAhead(model, LAYERS, 2, load_ahead.PIECE)
    ahead.building(0)
    ahead.close()  # (stops at once: no hang, no error)
    assert not ahead.sched.is_alive()


def test_off_when_no_threads(model, monkeypatch):
    monkeypatch.setenv("DS41_LOAD_THREADS", "0")
    assert load_ahead.LoadAhead.start(model, LAYERS) is None


def _stub_roundtrip(t):
    return t.bfloat16().float()  # (a lossy round trip: the statistics are nonzero)


def test_stats_on_pool_equal_inline(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("DS41_DENSE_WEIGHTS", "bfp8")
    monkeypatch.setattr(common, "bfp8_roundtrip", _stub_roundtrip)
    g = torch.Generator().manual_seed(0)
    ws = [(torch.randn(64 + 32 * k, 128, generator=g), f"m{k}") for k in range(40)]

    def run(pool):
        common.BFP8_STATS.clear()
        common.set_stats_pool(pool)
        try:
            common.set_load_layer(3)
            dts = [common.dense_dtype(w, label) for w, label in ws]
            common.drain_stats()
        finally:
            common.set_stats_pool(None)
            common.set_load_layer(None)
        return dts, [dict(s) for s in common.BFP8_STATS]

    inline = run(None)
    with ThreadPoolExecutor(8) as pool:
        pooled = run(pool)
    assert pooled == inline
    assert [s["matrix"] for s in pooled[1]] == [f"L03 m{k}" for k in range(40)]
    assert all(s["rel"] > 0 for s in pooled[1])
