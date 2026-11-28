# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the YUV readback records its readback and host-assembly spans under TT_DIT_STAGE_TIMING."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn
from models.tt_dit.utils import timing_tree as dt
from models.tt_dit.utils import yuv_d2h

TP, SP, H, W, T = 2, 2, 8, 8, 3


class _Shard:
    def __init__(self, arr):
        self.arr = arr
        self.shape = arr.shape


class _Host:
    def __init__(self, shards):
        self.shards = shards


class _DevTensor:
    dtype = layout = None

    def __init__(self, shards):
        self.shards = self.spec = shards
        self.shape = shards[0].shape

    def cpu(self, blocking=True):
        return _Host(self.shards)

    def tensor_topology(self):
        coords = [(r, c) for r in range(TP) for c in range(SP)]
        return type("Topo", (), {"mesh_coords": lambda self: coords})()


def _planes(seed=0):
    """Per-shard (1, h_per, w_per, T) uint8 planes with distinct bytes, plus the assembled reference."""
    rng = np.random.default_rng(seed)
    ref = [
        rng.integers(0, 256, size=(T, h, w), dtype=np.uint8) for h, w in ((H, W), (H // 2, W // 2), (H // 2, W // 2))
    ]
    tensors = []
    for plane in ref:
        h_per, w_per = plane.shape[1] // TP, plane.shape[2] // SP
        shards = [
            _Shard(
                np.ascontiguousarray(
                    plane[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per].transpose(1, 2, 0)
                )[None]
            )
            for r in range(TP)
            for c in range(SP)
        ]
        tensors.append(_DevTensor(shards))
    expected = np.concatenate([p.reshape(T, -1) for p in ref], axis=1)
    return tensors, expected


@pytest.fixture
def fake_device(monkeypatch):
    syncs = []
    monkeypatch.setattr(ttnn, "synchronize_device", lambda dev: syncs.append(dev))
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda host: host.shards)
    monkeypatch.setattr(yuv_d2h, "_to_torch_zero_copy", lambda s: torch.from_numpy(s.arr))
    dt.reset()
    yield syncs
    dt.reset()


def _run():
    (y, cb, cr), expected = _planes()
    mesh = type("Mesh", (), {"shape": (TP, SP)})()
    with dt.span(mesh, "decode TOTAL", root=True):
        out = yuv_d2h._yuv_planar_d2h(y, cb, cr, mesh, H, W, T)
    np.testing.assert_array_equal(out, expected)


def test_spans_name_readback_and_assembly_path(monkeypatch, fake_device):
    monkeypatch.setattr(dt, "ENABLED", True)
    _run()
    root = dt.roots()[-1]
    path = "C++ planar concat" if yuv_d2h.HAS_CPP_PLANAR_CONCAT else "torch fallback"
    assert [c.label for c in root.children] == ["yuv readback (3 planes)", f"yuv host assemble ({path})"]
    assert [c.category for c in root.children] == [dt.HOST_XFER, dt.HOST_COMPUTE]


def test_untimed_run_records_nothing(monkeypatch, fake_device):
    monkeypatch.setattr(dt, "ENABLED", False)
    _run()
    assert dt.root_count() == 0
    assert len(fake_device) == 1  # only the readback's own sync


@pytest.mark.parametrize("defer", [False, True])
def test_planes_to_host_defers_assembly(monkeypatch, fake_device, defer):
    """The fused-path entry point returns the same bytes eagerly or deferred; deferred skips the host assembly."""
    monkeypatch.setattr(dt, "ENABLED", True)
    monkeypatch.setattr(ttnn, "using_distributed_env", lambda: False)
    events = []
    monkeypatch.setattr(ttnn, "record_event", lambda dev, cq, device_range=None: events.append(dev) or "event")
    monkeypatch.setattr(ttnn, "event_synchronize", lambda event: events.remove(mesh))
    planes, expected = _planes()
    mesh = type("Mesh", (), {"shape": (TP, SP)})()
    with dt.span(mesh, "decode TOTAL", root=True):
        out = yuv_d2h.yuv_planes_to_host(planes, mesh, logical_h=H, logical_w=W, defer=defer)
    assert isinstance(out, yuv_d2h.DeferredYuvPlanar) == defer
    labels = [c.label for c in dt.roots()[-1].children]
    assert len(labels) == (1 if defer else 2)
    np.testing.assert_array_equal(out.result() if defer else out, expected)
    assert events == []  # a deferred readback waits on its own event, not the whole device


def test_kept_host_planes_take_the_next_read(monkeypatch, fake_device):
    """With ``host_planes`` a read lands in host tensors a finished read handed back, never in ones still in flight."""
    allocated = []

    def allocate(spec, mesh):
        host = _Host([_Shard(np.zeros_like(shard.arr)) for shard in spec])
        host.update_tensor_topology = lambda topology: None
        allocated.append(host)
        return host

    def copy(plane, host, blocking=True):
        assert not blocking
        for src, dst in zip(plane.shards, host.shards):
            np.copyto(dst.arr, src.arr)

    monkeypatch.setattr(ttnn, "allocate_tensor_on_host", allocate)
    monkeypatch.setattr(ttnn, "copy_device_to_host_tensor", copy)
    monkeypatch.setattr(ttnn, "record_event", lambda dev, cq, device_range=None: "event")
    monkeypatch.setattr(ttnn, "event_synchronize", lambda event: None)
    monkeypatch.setattr(_DevTensor, "cpu", None)  # no new host tensors per read
    mesh = type("Mesh", (), {"shape": (TP, SP)})()
    kept = {}

    def read(seed):
        planes, expected = _planes(seed)
        return yuv_d2h._yuv_planar_d2h(*planes, mesh, H, W, T, defer=True, host_planes=kept), expected

    (first, first_expected), (second, second_expected) = read(0), read(1)
    assert len(allocated) == 6  # two reads in flight: a set of three each
    np.testing.assert_array_equal(first.result(), first_expected)
    np.testing.assert_array_equal(second.result(), second_expected)
    third, third_expected = read(2)
    assert len(allocated) == 6
    np.testing.assert_array_equal(third.result(), third_expected)
    assert [len(free) for free in kept.values()] == [2]
