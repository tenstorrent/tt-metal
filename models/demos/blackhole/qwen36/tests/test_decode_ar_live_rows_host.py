# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_DECODE_AR_LIVE_ROWS (DecodeResidualAllReduce -> all_reduce_async live_rows_packing).

Host-checkable parts only, over a recording fake of the ttnn calls involved:
* the flag (default off, "1" on) and the eligibility bound (16 rows = faces 0/1 of a 32-row tile);
* all_reduce passes live_rows_packing=True only when the flag is on and the packed partial is bf16 with <= 16 rows,
  and otherwise calls all_reduce_async with exactly the default arguments (flag off is a no-op);
* the GDN fp32 path (QWEN36_DECODE_AR_GDN_FP32=1) never asks for packing.
The device numerics and the latency gate are in test_decode_ar_live_rows.py. No device. Run:
pytest models/demos/blackhole/qwen36/tests/test_decode_ar_live_rows_host.py -q
"""

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc


class FT:
    """Fake tensor: a shape and a dtype."""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


@pytest.fixture
def rec(monkeypatch):
    calls = []

    def to_memory_config(t, mem, *a, **k):
        calls.append(("to_memory_config", (t, mem), k))
        return FT(t.shape, t.dtype)

    def typecast(t, dtype, *a, **k):
        calls.append(("typecast", (t, dtype), k))
        return FT(t.shape, dtype)

    def reshape(t, s, *a, **k):
        calls.append(("reshape", (t, s), k))
        return FT(tuple(s), t.dtype)

    def all_reduce_async(packed, buffer, **k):
        calls.append(("all_reduce_async", (packed, buffer), k))
        return FT(packed.shape, packed.dtype)

    monkeypatch.setattr(ttnn, "to_memory_config", to_memory_config)
    monkeypatch.setattr(ttnn, "typecast", typecast)
    monkeypatch.setattr(ttnn, "reshape", reshape)
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    monkeypatch.setattr(ttnn.experimental, "all_reduce_async", all_reduce_async, raising=False)
    return calls


class _CCL:
    def get_and_cycle_barrier_semaphore_handle(self, axis=None):
        return "sem"


def _dar(live_rows, gdn_fp32=False):
    d = tpc.DecodeResidualAllReduce.__new__(tpc.DecodeResidualAllReduce)
    d.mesh, d.tt_ccl, d.dim, d.tp = "mesh", _CCL(), 5120, 8
    d.topology, d.num_links = "ring", 2
    d.gdn_fp32, d.live_rows = gdn_fp32, live_rows
    d.fuse_residual, d.fuse_live_rows, d._armed, d._fused_out = False, False, None, None  # ar-fuse off
    d.res_mem, d.buffer, d.buffer_fp32 = "RES40", "buf_bf16", ("buf_fp32" if gdn_fp32 else None)
    return d


def _ar_kwargs(calls):
    (k,) = [k for name, _, k in calls if name == "all_reduce_async"]
    return k


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(tpc.DECODE_AR_LIVE_ROWS_ENV, raising=False)
    assert not tpc.decode_ar_live_rows_enabled()
    monkeypatch.setenv(tpc.DECODE_AR_LIVE_ROWS_ENV, "0")
    assert not tpc.decode_ar_live_rows_enabled()
    monkeypatch.setenv(tpc.DECODE_AR_LIVE_ROWS_ENV, "1")
    assert tpc.decode_ar_live_rows_enabled()
    assert tpc.DECODE_AR_LIVE_ROWS_MAX == 16


@pytest.mark.parametrize("rows", [1, 4, 8, 12, 16])
def test_on_eligible_rows_pass_packing(rec, rows):
    _dar(live_rows=True).all_reduce(FT((1, 1, rows, 5120), ttnn.bfloat16))
    k = _ar_kwargs(rec)
    assert k.get("live_rows_packing") is True
    assert "fp32_dest_acc" not in k


@pytest.mark.parametrize("rows", [17, 24, 32])
def test_on_full_rows_keep_default_call(rec, rows):
    _dar(live_rows=True).all_reduce(FT((1, 1, rows, 5120), ttnn.bfloat16))
    assert "live_rows_packing" not in _ar_kwargs(rec)


@pytest.mark.parametrize("rows", [1, 8, 16, 32])
def test_off_never_passes_packing(rec, rows):
    _dar(live_rows=False).all_reduce(FT((1, 1, rows, 5120), ttnn.bfloat16))
    k = _ar_kwargs(rec)
    assert "live_rows_packing" not in k
    assert set(k) == {
        "cluster_axis",
        "mesh_device",
        "multi_device_global_semaphore",
        "topology",
        "num_links",
        "memory_config",
    }


def test_partial_cast_to_bf16_is_eligible(rec):
    """A fp32 partial (GDN out-proj) is cast to bf16 before the all-reduce when GDN fp32 is off: packing applies."""
    _dar(live_rows=True).all_reduce(FT((1, 1, 8, 5120), ttnn.float32), keep_fp32=True)
    assert any(name == "typecast" for name, _, _ in rec)
    assert _ar_kwargs(rec).get("live_rows_packing") is True


def test_gdn_fp32_never_packs(rec):
    _dar(live_rows=True, gdn_fp32=True).all_reduce(FT((1, 1, 8, 5120), ttnn.float32), keep_fp32=True)
    k = _ar_kwargs(rec)
    assert k.get("fp32_dest_acc") is True
    assert "live_rows_packing" not in k
    assert [a[1] for name, a, _ in rec if name == "all_reduce_async"] == ["buf_fp32"]


def test_rank3_partial_reshaped_rows_count(rec):
    """[1,R,1,dim]-style partials are reshaped to [1,1,R,dim] first; eligibility reads the reshaped row count."""
    _dar(live_rows=True).all_reduce(FT((2, 4, 2, 5120), ttnn.bfloat16))
    assert _ar_kwargs(rec).get("live_rows_packing") is True
    _dar(live_rows=True).all_reduce(FT((2, 4, 4, 5120), ttnn.bfloat16))
    assert "live_rows_packing" not in [k for name, _, k in rec if name == "all_reduce_async"][-1]
