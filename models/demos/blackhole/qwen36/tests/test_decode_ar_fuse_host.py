# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_DECODE_AR_FUSE_RESIDUAL (DecodeResidualAllReduce.arm_residual / all_reduce / add_residual).

Over a recording fake of the ttnn calls involved:
* flag off: arm is a no-op, all_reduce_async gets no residual_tensor, add_residual adds (today's path, unchanged);
* flag on: the armed residual reaches all_reduce_async as residual_tensor and add_residual returns that output
  without another add;
* fallbacks keep the separate add: fp32 GDN partial (QWEN36_DECODE_AR_GDN_FP32=1), residual of another
  shape / dtype / memory config, nothing armed;
* one armed residual feeds at most one all_reduce, and a module output that is not the fused all_reduce output
  raises instead of silently dropping the residual.
The device numerics are in test_decode_ar_fuse.py. No device. Run:
pytest models/demos/blackhole/qwen36/tests/test_decode_ar_fuse_host.py -q
"""

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

RES_MEM = object()  # stands in for the RES40 MemoryConfig
OTHER_MEM = object()


class FT:
    """Fake tensor: shape, dtype, memory config."""

    def __init__(self, shape=(1, 1, 8, 5120), dtype=None, mem=RES_MEM, name="t"):
        self.shape = tuple(shape)
        self.dtype = ttnn.bfloat16 if dtype is None else dtype
        self._mem = mem
        self.name = name

    def memory_config(self):
        return self._mem


@pytest.fixture
def rec(monkeypatch):
    calls = []

    def to_mem(t, mem, *a, **k):
        calls.append(("to_memory_config", (t, mem), k))
        return FT(t.shape, t.dtype, mem, "packed")

    def typecast(t, dtype, *a, **k):
        calls.append(("typecast", (t, dtype), k))
        return FT(t.shape, dtype, t.memory_config(), "cast")

    def ar(packed, buf, **k):
        calls.append(("all_reduce_async", (packed, buf), k))
        return FT(packed.shape, packed.dtype, RES_MEM, "ar_out")

    def add(a, b, **k):
        calls.append(("add", (a, b), k))
        return FT(a.shape, a.dtype, RES_MEM, "sum")

    monkeypatch.setattr(ttnn, "to_memory_config", to_mem)
    monkeypatch.setattr(ttnn, "typecast", typecast)
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: calls.append(("deallocate", a, k)))
    monkeypatch.setattr(ttnn, "add", add)
    monkeypatch.setattr(ttnn.experimental, "all_reduce_async", ar, raising=False)
    return calls


def _dar(fuse, gdn_fp32=False):
    d = tpc.DecodeResidualAllReduce.__new__(tpc.DecodeResidualAllReduce)
    d.mesh = "mesh"
    d.res_mem = RES_MEM
    d.topology = "ring"
    d.num_links = 2
    d.gdn_fp32 = gdn_fp32
    d.fuse_residual = fuse
    d._armed = None
    d._fused_out = None
    d.buffer = "buf"
    d.buffer_fp32 = "buf32" if gdn_fp32 else None

    class _CCL:
        def get_and_cycle_barrier_semaphore_handle(self, axis):
            return f"sem{axis}"

    d.tt_ccl = _CCL()
    return d


def _names(calls):
    return [c[0] for c in calls]


def _ar_kwargs(calls):
    (k,) = [c[2] for c in calls if c[0] == "all_reduce_async"]
    return k


def test_flag_off_is_unchanged(rec):
    d = _dar(fuse=False)
    x = FT(name="x")
    d.arm_residual(x)
    assert d._armed is None
    out = d.all_reduce(FT(mem=OTHER_MEM, name="partial"))
    assert "residual_tensor" not in _ar_kwargs(rec)
    h = d.add_residual(x, out)
    assert h is not out and h.name == "sum"
    assert _names(rec).count("add") == 1


def test_flag_on_fuses_and_skips_add(rec):
    d = _dar(fuse=True)
    x = FT(name="x")
    d.arm_residual(x)
    out = d.all_reduce(FT(mem=OTHER_MEM, name="partial"))
    k = _ar_kwargs(rec)
    assert k["residual_tensor"] is x
    assert k["memory_config"] is RES_MEM and "fp32_dest_acc" not in k
    h = d.add_residual(x, out)
    assert h is out
    assert "add" not in _names(rec)
    assert d._armed is None and d._fused_out is None


def test_fp32_gdn_partial_keeps_separate_add(rec):
    d = _dar(fuse=True, gdn_fp32=True)
    x = FT(name="x")
    d.arm_residual(x)
    out = d.all_reduce(FT(dtype=ttnn.float32, mem=OTHER_MEM, name="partial"), keep_fp32=True)
    k = _ar_kwargs(rec)
    assert "residual_tensor" not in k and k["fp32_dest_acc"] is True
    h = d.add_residual(x, out)
    assert h is not out and _names(rec).count("add") == 1


def test_keep_fp32_without_gdn_fp32_still_fuses(rec):
    # QWEN36_DECODE_AR_GDN_FP32=0 (default): the GDN partial is cast to bf16, so it fuses like the others.
    d = _dar(fuse=True, gdn_fp32=False)
    x = FT(name="x")
    d.arm_residual(x)
    out = d.all_reduce(FT(dtype=ttnn.float32, mem=OTHER_MEM, name="partial"), keep_fp32=True)
    assert _ar_kwargs(rec)["residual_tensor"] is x
    assert d.add_residual(x, out) is out


@pytest.mark.parametrize(
    "residual",
    [
        FT(shape=(1, 1, 32, 5120), name="x_other_rows"),
        FT(dtype=ttnn.float32, name="x_fp32"),
        FT(mem=OTHER_MEM, name="x_dram"),
    ],
    ids=["shape", "dtype", "memcfg"],
)
def test_incompatible_residual_falls_back(rec, residual):
    d = _dar(fuse=True)
    d.arm_residual(residual)
    out = d.all_reduce(FT(mem=OTHER_MEM, name="partial"))
    assert "residual_tensor" not in _ar_kwargs(rec)
    assert d._armed is None
    h = d.add_residual(residual, out)
    assert h is not out and _names(rec).count("add") == 1


def test_unarmed_all_reduce_never_fuses(rec):
    d = _dar(fuse=True)
    d.all_reduce(FT(mem=OTHER_MEM, name="partial"))
    assert "residual_tensor" not in _ar_kwargs(rec)


def test_armed_residual_is_consumed_once(rec):
    d = _dar(fuse=True)
    x = FT(name="x")
    d.arm_residual(x)
    d.all_reduce(FT(mem=OTHER_MEM, name="p0"))
    out2 = d.all_reduce(FT(mem=OTHER_MEM, name="p1"))
    ks = [c[2] for c in rec if c[0] == "all_reduce_async"]
    assert ks[0].get("residual_tensor") is x and "residual_tensor" not in ks[1]
    # The module returned the second (unfused) output: the residual was folded into the first -> loud failure.
    with pytest.raises(RuntimeError, match="QWEN36_DECODE_AR_FUSE_RESIDUAL"):
        d.add_residual(x, out2)


def test_rearm_clears_stale_fused_output(rec):
    d = _dar(fuse=True)
    x = FT(name="x")
    d.arm_residual(x)
    d.all_reduce(FT(mem=OTHER_MEM, name="p0"))
    d.arm_residual(x)  # a new module starts: the previous fused output must not leak into its add
    assert d._fused_out is None
