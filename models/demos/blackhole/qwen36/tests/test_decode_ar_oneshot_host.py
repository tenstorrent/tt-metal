# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_DECODE_AR_ONESHOT (DecodeResidualAllReduce -> all_reduce_async one_shot).

Host-checkable parts only, over a recording fake of the ttnn calls involved:
* flag off (and an instance built without __init__) calls all_reduce_async with exactly the default arguments;
* flag on passes one_shot=True for every bf16 all-reduce, with or without a fused residual, and never live-rows packing;
* an fp32 GDN partial (QWEN36_DECODE_AR_GDN_FP32=1) keeps the default op;
* a tp that does not divide the 40 RES40 cores turns the flag off at construction.
The device numerics are in test_decode_ar_oneshot.py (and the op test
tests/ttnn/unit_tests/operations/ccl/test_all_reduce_async_one_shot.py). No device. Run:
pytest models/demos/blackhole/qwen36/tests/test_decode_ar_oneshot_host.py -q
"""

import types

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc


class FT:
    """Fake tensor: a shape, a dtype and (for armed residuals) a memory config."""

    def __init__(self, shape, dtype, mem="RES40"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._mem = mem

    def memory_config(self):
        return self._mem


@pytest.fixture
def rec(monkeypatch):
    calls = []

    def to_memory_config(t, mem, *a, **k):
        calls.append(("to_memory_config", (t, mem), k))
        return FT(t.shape, t.dtype)

    def typecast(t, dtype, *a, **k):
        calls.append(("typecast", (t, dtype), k))
        return FT(t.shape, dtype)

    def all_reduce_async(packed, buffer, **k):
        calls.append(("all_reduce_async", (packed, buffer), k))
        return FT(packed.shape, packed.dtype)

    monkeypatch.setattr(ttnn, "to_memory_config", to_memory_config)
    monkeypatch.setattr(ttnn, "typecast", typecast)
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    monkeypatch.setattr(ttnn.experimental, "all_reduce_async", all_reduce_async, raising=False)
    return calls


class _CCL:
    def get_and_cycle_barrier_semaphore_handle(self, axis=None):
        return "sem"


def _dar(one_shot=None, gdn_fp32=False, live_rows=False, fuse=False):
    d = tpc.DecodeResidualAllReduce.__new__(tpc.DecodeResidualAllReduce)
    d.mesh, d.tt_ccl, d.dim, d.tp = "mesh", _CCL(), 5120, 8
    d.topology, d.num_links = "ring", 2
    d.gdn_fp32, d.live_rows = gdn_fp32, live_rows
    d.fuse_residual, d.fuse_live_rows, d._armed, d._fused_out = fuse, False, None, None
    d.res_mem, d.buffer, d.buffer_fp32 = "RES40", "buf_bf16", ("buf_fp32" if gdn_fp32 else None)
    if one_shot is not None:  # None: as built without __init__ (class default)
        d.one_shot = one_shot
    return d


def _ar_kwargs(calls):
    (k,) = [k for name, _, k in calls if name == "all_reduce_async"]
    return k


DEFAULT_KW = {"cluster_axis", "mesh_device", "multi_device_global_semaphore", "topology", "num_links", "memory_config"}


@pytest.mark.parametrize("one_shot", [None, False])
@pytest.mark.parametrize("rows", [8, 32])
def test_flag_off_is_default_call(rec, one_shot, rows):
    _dar(one_shot).all_reduce(FT((1, 1, rows, 5120), ttnn.bfloat16, mem="DRAM"))
    assert set(_ar_kwargs(rec)) == DEFAULT_KW


@pytest.mark.parametrize("rows", [1, 8, 16, 32])
def test_flag_on_bf16(rec, rows):
    d = _dar(True, live_rows=True)  # live rows would engage at <= 16 rows: one_shot takes precedence
    d.all_reduce(FT((1, 1, rows, 5120), ttnn.bfloat16, mem="DRAM"))
    k = _ar_kwargs(rec)
    assert k.get("one_shot") is True and "live_rows_packing" not in k
    assert set(k) == DEFAULT_KW | {"one_shot"}


def test_flag_on_with_fused_residual(rec):
    d = _dar(True, fuse=True)
    res = FT((1, 1, 32, 5120), ttnn.bfloat16)
    d.arm_residual(res)
    out = d.all_reduce(FT((1, 1, 32, 5120), ttnn.bfloat16, mem="DRAM"))
    k = _ar_kwargs(rec)
    assert k.get("one_shot") is True and k.get("residual_tensor") is res
    assert d.add_residual(res, out) is out


def test_fp32_gdn_partial_keeps_default(rec):
    d = _dar(True, gdn_fp32=True)
    d.all_reduce(FT((1, 1, 32, 5120), ttnn.float32, mem="DRAM"), keep_fp32=True)
    k = _ar_kwargs(rec)
    assert "one_shot" not in k and k.get("fp32_dest_acc") is True


def test_tp_not_dividing_res40_disables(monkeypatch):
    # Construction-time guard only: stop before any device allocation.
    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop

    monkeypatch.setattr(ttnn, "CoreRangeSet", stop)
    mesh = types.SimpleNamespace(shape=(1, 3))
    tt_ccl = types.SimpleNamespace(get_num_links=lambda axis=None: 1)
    args = types.SimpleNamespace(dim=5120, ccl_topology=lambda: "ring")
    d = tpc.DecodeResidualAllReduce.__new__(tpc.DecodeResidualAllReduce)
    with pytest.raises(Stop):
        tpc.DecodeResidualAllReduce.__init__(d, mesh, args, tt_ccl, one_shot=True)
    assert d.one_shot is False
    d2 = tpc.DecodeResidualAllReduce.__new__(tpc.DecodeResidualAllReduce)
    with pytest.raises(Stop):
        tpc.DecodeResidualAllReduce.__init__(d2, types.SimpleNamespace(shape=(1, 8)), args, tt_ccl, one_shot=True)
    assert d2.one_shot is True
