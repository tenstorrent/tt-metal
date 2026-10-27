# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host semantics of the KDA chronology scalars (no device): ``chrono_scalar`` keeps one persistent start/end pair per
mesh and rewrites it in place only when the value changes; ``scalar`` refuses to allocate a new constant once the mesh
is frozen by a trace capture (the 2026-10-27 served hang: scalar(49152) allocated after the captures, clobbered by the
decode replays, ttKDA then waited forever on an interval longer than its capacity). Run with ``--noconftest``."""


from models.demos.glm53_flash.tt import kda as kda_mod
from models.demos.glm53_flash.tt.kda import TtKDA


class _FakeMesh:
    def __init__(self):
        self.chrono = {}
        self.allocations_frozen = [False]
        self.uploads = []

    def upload(self, t, dtype, layout=None, **kw):
        self.uploads.append(int(t.reshape(-1)[0]))
        return {"value": int(t.reshape(-1)[0])}

    def shard_2d(self, rows, cols):
        return ("replicate", rows, cols)


class _Attn:
    def __init__(self, mesh):
        self.mesh = mesh
        self._scalars = {}


def _patch_writes(monkeypatch):
    writes = []
    monkeypatch.setattr(kda_mod.ttnn, "from_torch", lambda t, **kw: {"host": int(t.reshape(-1)[0])}, raising=False)
    monkeypatch.setattr(
        kda_mod.ttnn,
        "copy_host_to_device_tensor",
        lambda host, dev: writes.append((host["host"], id(dev))),
        raising=False,
    )
    return writes


def test_chrono_pair_is_allocated_once_and_written_on_change(monkeypatch):
    writes = _patch_writes(monkeypatch)
    mesh = _FakeMesh()
    a, b = _Attn(mesh), _Attn(mesh)  # two layers of the same mesh share the pair
    s0 = TtKDA.chrono_scalar(a, 0, "start")
    e0 = TtKDA.chrono_scalar(a, 16384, "end")
    assert mesh.uploads == [0, 0]  # the pair, allocated once, zero-initialised
    assert writes == [(16384, id(e0))]  # start stayed 0 (no write), end changed
    assert TtKDA.chrono_scalar(b, 16384, "end") is e0 and len(writes) == 1  # same value: no write
    assert TtKDA.chrono_scalar(b, 16384, "start") is s0 and writes[-1] == (16384, id(s0))
    assert TtKDA.chrono_scalar(a, 32768, "end") is e0 and writes[-1] == (32768, id(e0))
    assert mesh.uploads == [0, 0]  # never a third allocation
    assert mesh.chrono["values"] == {"start": 16384, "end": 32768}


def test_scalar_refuses_new_constants_after_a_capture(monkeypatch):
    _patch_writes(monkeypatch)
    mesh = _FakeMesh()
    attn = _Attn(mesh)
    z = TtKDA.scalar(attn, 0)
    TtKDA.chrono_scalar(attn, 0, "start")  # the pair exists before the capture (construction allocates it)
    mesh.allocations_frozen[0] = True
    assert TtKDA.scalar(attn, 0) is z  # cached constants stay available
    raised = None
    try:
        TtKDA.scalar(attn, 49152)
    except RuntimeError as exc:
        raised = str(exc)
    assert raised and "after a trace capture" in raised
    # the persistent pair keeps working after the freeze (in-place writes, no allocation)
    TtKDA.chrono_scalar(attn, 49152, "end")
    assert mesh.uploads == [0, 0, 0]  # scalar(0) + the pair; nothing after the freeze


def test_chrono_pair_must_exist_before_the_freeze(monkeypatch):
    _patch_writes(monkeypatch)
    mesh = _FakeMesh()
    mesh.allocations_frozen[0] = True
    raised = False
    try:
        TtKDA.chrono_scalar(_Attn(mesh), 5, "end")
    except RuntimeError:
        raised = True
    assert raised and mesh.uploads == []


def test_eager_pad_length_buckets():
    from models.demos.glm53_flash.tt.mesh_model import DEFAULT_PREFILL_BUCKETS, eager_pad_length, prefill_buckets

    unit = 256  # 32 * 8 rows
    assert eager_pad_length(128, unit, DEFAULT_PREFILL_BUCKETS) == 2048
    assert eager_pad_length(2048, unit, DEFAULT_PREFILL_BUCKETS) == 2048
    assert eager_pad_length(2049, unit, DEFAULT_PREFILL_BUCKETS) == 4096
    assert eager_pad_length(4097, unit, DEFAULT_PREFILL_BUCKETS) == 16384  # no 8192 bucket (1024-row slab L1 clash)
    assert eager_pad_length(16362, unit, DEFAULT_PREFILL_BUCKETS) == 16384
    assert eager_pad_length(16385, unit, DEFAULT_PREFILL_BUCKETS) == 16640  # above the largest bucket: the ceiling
    assert eager_pad_length(300, unit, ()) == 512  # no buckets: the row-unit ceiling
    assert prefill_buckets({"GLM53_PREFILL_BUCKETS": "0"}) == ()
    assert prefill_buckets({"GLM53_PREFILL_BUCKETS": "4096,2048"}) == (2048, 4096)
    assert prefill_buckets({}) == DEFAULT_PREFILL_BUCKETS
    raised = False
    try:
        eager_pad_length(100, unit, (1000,))
    except ValueError:
        raised = True
    assert raised
