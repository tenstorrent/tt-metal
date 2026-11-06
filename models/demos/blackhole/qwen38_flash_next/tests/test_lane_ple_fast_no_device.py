# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QWEN38_LANE_PLE_FAST: vectorized gather, persistent host tensor and the flag-off path (no device)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module

ROW_ELEMS = 160
ROWS_PER_SHARD = 64
PARTS = 7


def _lookup_stub(seed=0):
    rng = np.random.default_rng(seed)
    tables = [rng.integers(0, 65535, size=(ROWS_PER_SHARD, ROW_ELEMS), dtype=np.uint16) for _ in range(PARTS)]
    proved: list = []

    class _Reader:
        def __init__(self, shard):
            self.shard = shard

        def prove_identity(self, _when):
            proved.append(self.shard)

    stub = SimpleNamespace(
        _mmap_tables=lambda: tables,
        host_embedding=SimpleNamespace(rows_per_shard=ROWS_PER_SHARD),
        _row_bytes=ROW_ELEMS * 2,
        readers=[_Reader(i) for i in range(PARTS)],
        _proven={},
    )
    return stub, tables, proved


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("count", [1, 16, 512])
def test_vectorized_gather_is_byte_equal_to_the_loop(seed, count):
    stub, _tables, _ = _lookup_stub(seed)
    ids = np.random.default_rng(100 + seed).integers(0, PARTS * ROWS_PER_SHARD, size=count, dtype=np.int64)
    gather = ple_module.Qwen38ResidentPLELookup._gather_mmap
    loop = gather(stub, ids)
    fast = gather(stub, ids, vectorized=True)
    assert bytes(fast) == bytes(loop)
    assert len(fast) == count * ROW_ELEMS * 2


def test_vectorized_gather_proofs_match_and_defer():
    stub, _tables, proved = _lookup_stub()
    ids = np.array([3, 70, 71, 200, 3, 441], dtype=np.int64)
    gather = ple_module.Qwen38ResidentPLELookup._gather_mmap
    gather(stub, ids)
    inline = sorted(set(proved))
    proved.clear()
    proofs: list = []
    gather(stub, ids, proofs, vectorized=True)
    assert proved == []  # deferred: nothing proven yet
    assert sorted(r.shard for r in proofs) == inline
    ple_module.Qwen38ResidentPLELookup.prove_deferred(proofs)
    assert sorted(proved) == inline and proofs == []


class _FakeTtnn:
    """from_torch borrows the torch buffer (as the chain's mapper does); copy snapshots what the device would get."""

    bfloat16 = "bf16"
    ROW_MAJOR_LAYOUT = "rm"

    def __init__(self):
        self.uploads: list[bytes] = []
        self.from_torch_calls = 0
        self.mapper_dims: list = []
        self.from_torch_mappers: list = []
        self.concat_dims: list = []

    def ShardTensor2dMesh(self, *a, **k):
        self.mapper_dims.append(k.get("dims"))
        return ("mapper", k.get("dims"))

    def from_torch(self, tensor, **_kw):
        self.from_torch_calls += 1
        self.from_torch_mappers.append(_kw.get("mesh_mapper"))
        host = SimpleNamespace(source=tensor, topology=("topology", _kw.get("mesh_mapper")))
        host.tensor_topology = lambda: host.topology
        host.update_tensor_topology = lambda topology: setattr(host, "topology", topology)
        return host

    def to_torch(self, host, **_kw):
        return host.source

    def ConcatMeshToTensor(self, *a, **k):
        self.concat_dims.append(k.get("dim"))
        return "composer"

    def copy_host_to_device_tensor(self, host, _device):
        self.uploads.append(bytes(host.source.contiguous().view(torch.uint8).flatten().tolist()))


def _model_stub(monkeypatch):
    fake = _FakeTtnn()
    monkeypatch.setattr(model_module, "ttnn", fake)
    # QWEN38_LANE_PLE_DMAJOR now defaults on; these tests cover the lane-major buffer (the d-major ones use _dmajor_stub)
    monkeypatch.setattr(model_module, "LANE_PLE_DMAJOR", False)
    stub = SimpleNamespace(mesh_device="mesh", _lane_ple_hosts={})
    stub._fresh_lane_ple_host = model_module.Qwen38TTNNTextModel._fresh_lane_ple_host
    stub._lane_ple_borrow_proven = lambda b, h, dim=3: model_module.Qwen38TTNNTextModel._lane_ple_borrow_proven(
        stub, b, h, dim
    )
    return fake, stub


def _method(name):
    return getattr(model_module.Qwen38TTNNTextModel, name)


def test_persistent_host_buffer_bytes_equal_fresh_from_torch(monkeypatch):
    fake, stub = _model_stub(monkeypatch)
    state = object()
    keep: list = []
    last = None
    for step in range(4):
        rows = torch.randn(1, 1, 32, 2560).to(torch.bfloat16)
        host = _method("_lane_ple_host_tensor")(stub, state, rows, keep)
        fresh = fake.from_torch(rows.contiguous())
        assert bytes(host.source.view(torch.uint8).flatten().tolist()) == bytes(
            fresh.source.view(torch.uint8).flatten().tolist()
        )
        if last is not None:
            assert host is last  # one persistent tensor, no per-step allocation
        last = host
    assert keep == []


def test_pending_upload_falls_back_to_a_fresh_tensor_until_fenced(monkeypatch):
    fake, stub = _model_stub(monkeypatch)
    state = object()
    rows = torch.ones(1, 1, 32, 2560, dtype=torch.bfloat16)
    first = _method("_lane_ple_host_tensor")(stub, state, rows, None)
    stub._lane_ple_hosts[id(state)][3] = True  # an upload from the buffer may be in flight
    keep: list = []
    second = _method("_lane_ple_host_tensor")(stub, state, rows * 2, keep)
    assert second is not first and keep == [second]
    assert torch.equal(first.source, rows)  # the buffer an in-flight copy reads is untouched
    model_module.Qwen38TTNNTextModel.mark_lane_ple_fenced(stub)
    third = _method("_lane_ple_host_tensor")(stub, state, rows * 3, None)
    assert third is first and torch.equal(first.source, rows * 3)


def test_unproven_borrow_uses_fresh_tensors(monkeypatch):
    fake, stub = _model_stub(monkeypatch)
    fake.to_torch = lambda host, **_kw: torch.zeros(1)  # the readback does not show the in-place write
    state = object()
    rows = torch.ones(1, 1, 32, 2560, dtype=torch.bfloat16)
    keep: list = []
    host = _method("_lane_ple_host_tensor")(stub, state, rows, keep)
    assert keep == [host] and stub._lane_ple_hosts[id(state)][2] is None


def _write_stub(monkeypatch):
    fake = _FakeTtnn()
    monkeypatch.setattr(model_module, "ttnn", fake)
    calls: list = []
    rows = torch.zeros(1, 1, 2, 2560, dtype=torch.bfloat16)

    def host_lanes(tokens, contexts, *args, **kwargs):
        calls.append((args, kwargs))
        return rows, tuple((1, 1) for _ in tokens)

    ple = SimpleNamespace(host_lanes=host_lanes)
    layers = [None] * (model_module.PLE_CHECKPOINT_LAYER + 1)
    layers[model_module.PLE_CHECKPOINT_LAYER] = SimpleNamespace(ple=ple)
    state = SimpleNamespace(ple_rows=SimpleNamespace(embedding_rows="dev"))
    stub = SimpleNamespace(
        _require_healthy=lambda: None,
        _ple_validated_lanes={id(state): (state, 2)},
        layers=layers,
        mesh_device="mesh",
        _lane_ple_hosts={},
        ple_split={},
    )
    return fake, stub, state, calls


def test_flag_off_calls_exactly_todays_functions(monkeypatch):
    monkeypatch.setattr(model_module, "LANE_PLE_FAST", False)
    monkeypatch.setattr(model_module, "LANE_PLE_SPLIT", False)
    fake, stub, state, calls = _write_stub(monkeypatch)
    keep: list = []
    out = model_module.Qwen38TTNNTextModel.write_lane_ple_rows(stub, state, [5, 6], [None, None], keep=keep)
    assert out == ((1, 1), (1, 1))
    assert calls == [((), {})]  # host_lanes(tokens, contexts): no proofs, no vectorized
    assert fake.from_torch_calls == 1 and len(fake.uploads) == 1 and len(keep) == 1
    assert stub._lane_ple_hosts == {} and stub.ple_split == {}


def test_flag_on_defers_proofs_and_reuses_host(monkeypatch):
    monkeypatch.setattr(model_module, "LANE_PLE_FAST", True)
    monkeypatch.setattr(model_module, "LANE_PLE_SPLIT", True)
    fake, stub, state, calls = _write_stub(monkeypatch)
    stub._lane_ple_host_tensor = lambda s, r, k: model_module.Qwen38TTNNTextModel._lane_ple_host_tensor(stub, s, r, k)
    stub._fresh_lane_ple_host = model_module.Qwen38TTNNTextModel._fresh_lane_ple_host
    stub._lane_ple_borrow_proven = lambda b, h, dim=3: model_module.Qwen38TTNNTextModel._lane_ple_borrow_proven(
        stub, b, h, dim
    )
    stub.prove_lane_ple_reads = lambda p: None
    proofs: list = []
    model_module.Qwen38TTNNTextModel.write_lane_ple_rows(stub, state, [5, 6], [None, None], proofs=proofs)
    assert calls[0][1] == {"proofs": proofs, "vectorized": True}
    assert set(stub.ple_split) == {"ple_lookup", "ple_host", "ple_h2d"}
    assert stub._lane_ple_hosts[id(state)][3] is True  # pending until a fence


# ---- QWEN38_LANE_PLE_DMAJOR -------------------------------------------------------------------------------------

D = 8
W = 2560 // D


class _DmajorTtnn(_FakeTtnn):
    """to_torch concatenates the borrowed buffer's per-device chunks along the composer's dim (like the mesh does)."""

    def ConcatMeshToTensor(self, *a, **k):
        super().ConcatMeshToTensor(*a, **k)
        return k.get("dim")

    def to_torch(self, host, mesh_composer=None, **_kw):
        return host.source


def _dmajor_stub(monkeypatch, flag=True):
    fake = _DmajorTtnn()
    monkeypatch.setattr(model_module, "ttnn", fake)
    monkeypatch.setattr(model_module, "LANE_PLE_DMAJOR", flag)
    monkeypatch.setattr(model_module, "MESH_SHAPE", (1, D))  # the TP8 mesh, whatever the default tree's TP is
    stub = SimpleNamespace(mesh_device="mesh", _lane_ple_hosts={})
    stub._fresh_lane_ple_host = model_module.Qwen38TTNNTextModel._fresh_lane_ple_host
    stub._lane_ple_borrow_proven = lambda b, h, dim=3: model_module.Qwen38TTNNTextModel._lane_ple_borrow_proven(
        stub, b, h, dim
    )
    return fake, stub


@pytest.mark.parametrize("lanes", [1, 8, 32])
def test_dmajor_buffer_shape_and_per_device_chunks(monkeypatch, lanes):
    fake, stub = _dmajor_stub(monkeypatch)
    state = object()
    for step in range(3):
        rows = torch.randn(1, 1, lanes, 2560).to(torch.bfloat16)
        host = _method("_lane_ple_host_tensor")(stub, state, rows, None)
        buffer = stub._lane_ple_hosts[id(state)][1]
        assert tuple(buffer.shape) == (1, D, lanes, W) and buffer.is_contiguous() and buffer.dtype == torch.bfloat16
        assert host.source is buffer
        for d in range(D):
            assert torch.equal(buffer[:, d], rows[..., d * W : (d + 1) * W].reshape(1, lanes, W))
            assert buffer[:, d].is_contiguous()
    assert fake.mapper_dims[0] == (None, 3) and (None, 1) in fake.mapper_dims
    assert fake.from_torch_mappers[0] == ("mapper", (None, 1))


def test_dmajor_host_topology_is_relabelled_lane_major(monkeypatch):
    """copy_host_to_device_tensor copies the host topology onto the device tensor: the d-major host tensor must carry
    the lane-major dims=(None, 3) topology (the contract checks shard dim 3)."""

    fake, stub = _dmajor_stub(monkeypatch)
    state = object()
    for _ in range(3):
        host = _method("_lane_ple_host_tensor")(stub, state, torch.randn(1, 1, 8, 2560).to(torch.bfloat16), None)
        assert host.source is stub._lane_ple_hosts[id(state)][1]
        assert host.tensor_topology() == ("topology", ("mapper", (None, 3)))


def test_dmajor_probe_uses_dim_1_and_logs_once(monkeypatch):
    fake, stub = _dmajor_stub(monkeypatch)
    messages: list = []
    monkeypatch.setattr(model_module.logger, "info", lambda m: messages.append(m))
    state = object()
    for _ in range(3):
        _method("_lane_ple_host_tensor")(stub, state, torch.randn(1, 1, 8, 2560).to(torch.bfloat16), None)
    assert fake.concat_dims == [1]
    assert messages == ["lane PLE host buffer: layout=dmajor lanes=8 borrowed=True"]


def test_dmajor_failed_probe_falls_back_to_fresh_lane_major(monkeypatch):
    fake, stub = _dmajor_stub(monkeypatch)
    fake.to_torch = lambda host, **_kw: torch.zeros(1)
    messages: list = []
    monkeypatch.setattr(model_module.logger, "info", lambda m: messages.append(m))
    state = object()
    rows = torch.randn(1, 1, 8, 2560).to(torch.bfloat16)
    keep: list = []
    host = _method("_lane_ple_host_tensor")(stub, state, rows, keep)
    assert keep == [host] and torch.equal(host.source, rows)  # fresh from rows, lane-major
    assert fake.from_torch_mappers[-1] == ("mapper", (None, 3))
    assert stub._lane_ple_hosts[id(state)][2] is None
    assert messages == ["lane PLE host buffer: layout=dmajor lanes=8 borrowed=False"]
    keep2: list = []
    _method("_lane_ple_host_tensor")(stub, state, rows, keep2)
    assert len(keep2) == 1 and fake.concat_dims == [1]  # no second probe


def test_dmajor_pending_falls_back_then_reuses(monkeypatch):
    fake, stub = _dmajor_stub(monkeypatch)
    state = object()
    rows = torch.randn(1, 1, 8, 2560).to(torch.bfloat16)
    first = _method("_lane_ple_host_tensor")(stub, state, rows, None)
    stub._lane_ple_hosts[id(state)][3] = True
    keep: list = []
    second = _method("_lane_ple_host_tensor")(stub, state, rows * 2, keep)
    assert second is not first and keep == [second] and torch.equal(second.source, rows * 2)
    model_module.Qwen38TTNNTextModel.mark_lane_ple_fenced(stub)
    third = _method("_lane_ple_host_tensor")(stub, state, rows * 3, None)
    assert third is first


def test_dmajor_flag_off_is_lane_major_with_dim_3(monkeypatch):
    fake, stub = _dmajor_stub(monkeypatch, flag=False)
    state = object()
    rows = torch.randn(1, 1, 8, 2560).to(torch.bfloat16)
    host = _method("_lane_ple_host_tensor")(stub, state, rows, None)
    buffer = stub._lane_ple_hosts[id(state)][1]
    assert buffer.shape == rows.shape and torch.equal(buffer, rows) and host.source is buffer
    assert fake.mapper_dims == [(None, 3)] and fake.concat_dims == [3]
    assert fake.from_torch_mappers == [("mapper", (None, 3))]


def test_dmajor_shape_change_rebuilds_buffer(monkeypatch):
    fake, stub = _dmajor_stub(monkeypatch)
    state = object()
    _method("_lane_ple_host_tensor")(stub, state, torch.randn(1, 1, 8, 2560).to(torch.bfloat16), None)
    rows = torch.randn(1, 1, 32, 2560).to(torch.bfloat16)
    host = _method("_lane_ple_host_tensor")(stub, state, rows, None)
    assert tuple(stub._lane_ple_hosts[id(state)][1].shape) == (1, D, 32, W) and host.source.shape[2] == 32
