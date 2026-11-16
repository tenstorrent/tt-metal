# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The recycled readback buffers: reused only once released, and byte-identical output."""

import gc

import numpy as np
import pytest
import torch

from models.tt_dit.utils import host_buffers
from models.tt_dit.utils.host_buffers import empty_uint8

SHAPE = (4, 1 << 19)  # 2 MiB, above the pool's floor


def _addr(a: np.ndarray) -> int:
    return a.__array_interface__["data"][0]


def test_released_buffer_is_reused():
    a = empty_uint8(SHAPE)
    addr = _addr(a)
    del a
    gc.collect()
    b = empty_uint8(SHAPE)
    assert _addr(b) == addr and b.shape == SHAPE and b.dtype == np.uint8


@pytest.mark.parametrize("hold", ["view", "torch", "memoryview"])
def test_live_buffer_is_never_handed_out_again(hold):
    a = empty_uint8(SHAPE)
    a[:] = 7
    kept = {"view": lambda: a[1:3, ::2], "torch": lambda: torch.from_numpy(a), "memoryview": lambda: memoryview(a)}[hold]()
    addr = _addr(a)
    del a
    b = empty_uint8(SHAPE)
    assert _addr(b) != addr
    b[:] = 0
    assert np.asarray(kept).min() == 7


def test_pool_off_is_plain_empty(monkeypatch):
    monkeypatch.setenv("TT_DIT_HOST_BUFFER_POOL_MB", "0")
    before = len(host_buffers._slabs)
    empty_uint8(SHAPE)
    assert len(host_buffers._slabs) == before


def test_idle_bytes_stay_under_the_cap(monkeypatch):
    monkeypatch.setenv("TT_DIT_HOST_BUFFER_POOL_MB", "8")
    for rows in range(1, 12):
        empty_uint8((rows, 1 << 20))
    gc.collect()
    idle = [s for s in host_buffers._slabs if host_buffers._refs(s) <= host_buffers._IDLE_REFS]
    assert sum(s.size for s in idle) <= 8 << 20


def test_eviction_behind_a_live_slab_of_another_size(monkeypatch):
    # list.remove compared arrays element by element and raised once the slab to drop
    # was not first in the pool.
    monkeypatch.setattr(host_buffers, "_slabs", [])
    monkeypatch.setenv("TT_DIT_HOST_BUFFER_POOL_MB", "10")
    live = empty_uint8((3, 1 << 20))
    idle = empty_uint8((2, 1 << 20))
    del idle
    gc.collect()
    big = empty_uint8((6, 1 << 20))
    assert big.shape == (6, 1 << 20)
    assert [s.size for s in host_buffers._slabs] == [3 << 20, 6 << 20]
    del live


def test_live_and_idle_bytes_stay_under_the_cap(monkeypatch):
    monkeypatch.setattr(host_buffers, "_slabs", [])
    monkeypatch.setenv("TT_DIT_HOST_BUFFER_POOL_MB", "8")
    held = [empty_uint8((rows, 1 << 20)) for rows in (3, 4, 5)]
    assert all(h.shape[0] == r for h, r in zip(held, (3, 4, 5)))
    assert sum(s.size for s in host_buffers._slabs) <= 8 << 20
    del held
    gc.collect()
    assert sum(s.size for s in host_buffers._slabs) <= 8 << 20


def test_idle_slabs_stay_when_dropping_them_would_not_make_room(monkeypatch):
    monkeypatch.setattr(host_buffers, "_slabs", [])
    monkeypatch.setenv("TT_DIT_HOST_BUFFER_POOL_MB", "8")
    live = empty_uint8((5, 1 << 20))
    idle = empty_uint8((2, 1 << 20))
    del idle
    gc.collect()
    empty_uint8((4, 1 << 20))  # 5 live + 4 > 8 even with the idle 2 dropped
    assert [s.size for s in host_buffers._slabs] == [5 << 20, 2 << 20]
    del live

def test_malformed_cap_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("TT_DIT_HOST_BUFFER_POOL_MB", "3G")
    assert host_buffers._max_idle_bytes() == 3072 << 20

def test_planar_concat_into_a_recycled_buffer_matches_a_fresh_one():
    pc = pytest.importorskip("models.tt_dit.utils.planar_concat")
    if not pc.HAS_CPP_PLANAR_CONCAT:
        pytest.skip("C++ planar concat not built here")
    T, TP, SP, H, W = 28, 4, 8, 768, 1344
    rng = np.random.default_rng(0)
    y = [rng.integers(0, 256, (1, H // TP, W // SP, T), dtype=np.uint8) for _ in range(TP * SP)]
    u = [rng.integers(0, 256, (1, H // TP // 2, W // SP // 2, T), dtype=np.uint8) for _ in range(TP * SP)]
    v = [rng.integers(0, 256, (1, H // TP // 2, W // SP // 2, T), dtype=np.uint8) for _ in range(TP * SP)]
    row = H * W + 2 * (H // 2) * (W // 2)
    fresh = np.empty((T, row), dtype=np.uint8)
    pc.planar_concat_cpp(y, u, v, "CHWT", (TP, SP), out=fresh, out_H=H, out_W=W)
    stale = empty_uint8((T, row))
    stale[:] = 0xA5  # what a released buffer still holds
    addr = _addr(stale)
    del stale
    got = pc.planar_concat_cpp(y, u, v, "CHWT", (TP, SP), out_H=H, out_W=W)
    assert _addr(got) == addr
    assert np.array_equal(got, fresh)
