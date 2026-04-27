# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.deepseek_v3_b1.weights import upload


def _crs(x0: int, y0: int, x1: int, y1: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _crs_to_tuples(crs: ttnn.CoreRangeSet) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return sorted(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in crs.ranges())


@dataclass
class _FakeTensor:
    name: str


@dataclass
class _TensorWithStableId(_FakeTensor):
    tensor_id: int


@dataclass
class _FakeOverlappedTensor:
    fused_tensor: _FakeTensor
    tensor_shape: tuple[int, int]
    shard_shape: tuple[int, int]
    core_range_set: ttnn.CoreRangeSet
    dtype: object
    tile_shape: tuple[int, int]
    byte_offset: int
    total_size: int


@dataclass
class _InnerFixture:
    fused: object
    tensors: list[object]


@dataclass
class _OuterFixture:
    direct: object
    inner: _InnerFixture
    extra: tuple[object, ...]
    note: str = "fixture"


def _patch_upload_tensor_types(monkeypatch):
    monkeypatch.setattr(upload.ttnn, "Tensor", _FakeTensor, raising=False)
    monkeypatch.setattr(upload, "OverlappedTensor", _FakeOverlappedTensor)


class _FakeDevice:
    def __init__(self, x: int, y: int):
        self._grid = SimpleNamespace(x=x, y=y)

    def compute_with_storage_grid_size(self):
        return self._grid


class _FakeHostTensor:
    def __init__(
        self, name: str, *, sharded: bool, grid: ttnn.CoreRangeSet | None = None, shard_spec_none: bool = False
    ):
        self.name = name
        self.spec = f"spec-{name}"
        self._sharded = sharded
        if shard_spec_none:
            self._mem_config = SimpleNamespace(shard_spec=None)
        else:
            self._mem_config = SimpleNamespace(shard_spec=SimpleNamespace(grid=grid))

    def is_sharded(self) -> bool:
        return self._sharded

    def memory_config(self):
        return self._mem_config


def test_get_fd_grid_uses_compute_grid_minus_one_x_column():
    fd_grid = upload.get_fd_grid(_FakeDevice(13, 10))
    assert _crs_to_tuples(fd_grid) == [((0, 0), (11, 9))]


def test_get_fd_grid_returns_empty_when_grid_too_small():
    fd_grid = upload.get_fd_grid(_FakeDevice(1, 10))
    assert fd_grid.empty()


def test_split_core_ranges_partitions_tensor_grid_into_fd_and_sd():
    tensor_grid = _crs(0, 0, 12, 9)
    fd_grid = _crs(0, 0, 11, 9)
    fd_filter, sd_filter = upload.split_core_ranges(tensor_grid, fd_grid)

    assert fd_filter.num_cores() == 120
    assert sd_filter.num_cores() == 10
    assert fd_filter.contains(ttnn.CoreCoord(11, 9))
    assert not fd_filter.contains(ttnn.CoreCoord(12, 0))
    assert sd_filter.contains(ttnn.CoreCoord(12, 9))
    assert not sd_filter.contains(ttnn.CoreCoord(11, 9))

    union = fd_filter.merge(sd_filter)
    assert union == tensor_grid


def test_uploadable_mixin_backing_tensors_deduplicates_fused_and_direct_tensors(monkeypatch):
    _patch_upload_tensor_types(monkeypatch)
    fused = _FakeTensor("fused")
    direct = _FakeTensor("direct")
    extra = _FakeTensor("extra")

    ot_a = upload.OverlappedTensor(
        fused_tensor=fused,
        tensor_shape=(32, 32),
        shard_shape=(32, 32),
        core_range_set=_crs(0, 0, 0, 0),
        dtype=object(),
        tile_shape=(32, 32),
        byte_offset=0,
        total_size=2048,
    )
    ot_b = upload.OverlappedTensor(
        fused_tensor=fused,
        tensor_shape=(32, 32),
        shard_shape=(32, 32),
        core_range_set=_crs(0, 0, 0, 0),
        dtype=object(),
        tile_shape=(32, 32),
        byte_offset=128,
        total_size=2048,
    )

    @dataclass
    class _UploadFixture(upload.UploadableMixin):
        direct: upload.ttnn.Tensor
        fused_a: upload.OverlappedTensor
        fused_b: upload.OverlappedTensor
        tensor_list: list[upload.ttnn.Tensor]
        maybe_tensor: upload.ttnn.Tensor | None
        maybe_overlapped: upload.OverlappedTensor | None

    fixture = _UploadFixture(
        direct=direct,
        fused_a=ot_a,
        fused_b=ot_b,
        tensor_list=[direct, extra],
        maybe_tensor=None,
        maybe_overlapped=None,
    )
    got = fixture.backing_tensors()
    assert got == [direct, fused, extra]


def test_uploadable_mixin_raises_on_unknown_field_type(monkeypatch):
    _patch_upload_tensor_types(monkeypatch)
    direct = _FakeTensor("direct")

    @dataclass
    class _BadFixture(upload.UploadableMixin):
        direct: upload.ttnn.Tensor
        unsupported: str

    fixture = _BadFixture(direct=direct, unsupported="bad")
    with pytest.raises(TypeError):
        fixture.backing_tensors()


def test_uploadable_mixin_backing_tensors_deduplicates_by_tensor_id(monkeypatch):
    _patch_upload_tensor_types(monkeypatch)
    a = _TensorWithStableId("a", tensor_id=123)
    b = _TensorWithStableId("b", tensor_id=123)

    @dataclass
    class _IdFixture(upload.UploadableMixin):
        direct: upload.ttnn.Tensor
        fused: upload.OverlappedTensor

    fixture = _IdFixture(
        direct=a,
        fused=_FakeOverlappedTensor(
            fused_tensor=b,
            tensor_shape=(32, 32),
            shard_shape=(32, 32),
            core_range_set=_crs(0, 0, 0, 0),
            dtype=object(),
            tile_shape=(32, 32),
            byte_offset=0,
            total_size=2048,
        ),
    )
    got = fixture.backing_tensors()
    assert got == [a]


def test_uploadable_mixin_with_device_tensors_replaces_nested_tensors_and_overlapped_views(monkeypatch):
    _patch_upload_tensor_types(monkeypatch)
    fused_h = _FakeTensor("fused_h")
    direct_h = _FakeTensor("direct_h")
    list_h = _FakeTensor("list_h")
    fused_d = _FakeTensor("fused_d")
    direct_d = _FakeTensor("direct_d")
    list_d = _FakeTensor("list_d")

    overlapped = upload.OverlappedTensor(
        fused_tensor=fused_h,
        tensor_shape=(32, 32),
        shard_shape=(32, 32),
        core_range_set=_crs(0, 0, 0, 0),
        dtype=object(),
        tile_shape=(32, 32),
        byte_offset=64,
        total_size=2048,
    )

    @dataclass
    class _RebuildFixture(upload.UploadableMixin):
        direct: upload.ttnn.Tensor
        fused: upload.OverlappedTensor
        tensor_list: list[upload.ttnn.Tensor]
        maybe_tensor: upload.ttnn.Tensor | None

    fixture = _RebuildFixture(
        direct=direct_h,
        fused=overlapped,
        tensor_list=[list_h],
        maybe_tensor=None,
    )
    mapping = {
        upload.tensor_identity_key(fused_h): fused_d,
        upload.tensor_identity_key(direct_h): direct_d,
        upload.tensor_identity_key(list_h): list_d,
    }

    rebuilt = fixture.with_device_tensors(mapping)
    assert rebuilt.direct is direct_d
    assert rebuilt.tensor_list == [list_d]
    assert rebuilt.fused.fused_tensor is fused_d
    assert rebuilt.fused.tensor_shape == overlapped.tensor_shape
    assert rebuilt.fused.shard_shape == overlapped.shard_shape
    assert rebuilt.fused.dtype == overlapped.dtype
    assert rebuilt.fused.tile_shape == overlapped.tile_shape
    assert rebuilt.fused.byte_offset == overlapped.byte_offset
    assert rebuilt.fused.total_size == overlapped.total_size


def test_uploadable_mixin_with_device_tensors_raises_on_missing_mapping(monkeypatch):
    _patch_upload_tensor_types(monkeypatch)
    host = _FakeTensor("host")

    @dataclass
    class _MissingMapFixture(upload.UploadableMixin):
        direct: upload.ttnn.Tensor
        fused: upload.OverlappedTensor

    fixture = _MissingMapFixture(
        direct=host,
        fused=upload.OverlappedTensor(
            fused_tensor=host,
            tensor_shape=(32, 32),
            shard_shape=(32, 32),
            core_range_set=_crs(0, 0, 0, 0),
            dtype=object(),
            tile_shape=(32, 32),
            byte_offset=0,
            total_size=2048,
        ),
    )
    with pytest.raises(KeyError):
        fixture.with_device_tensors({})


def test_two_phase_upload_non_sharded_goes_full_fd_copy(monkeypatch):
    calls: list[tuple] = []

    def _alloc(spec, _device):
        return f"dev-{spec}"

    def _full(host_tensor, device_tensor):
        calls.append(("full", host_tensor.name, device_tensor))

    def _partial(host_tensor, device_tensor, core_filter):
        calls.append(("partial", host_tensor.name, device_tensor, core_filter.num_cores()))

    class _FDContext:
        def __enter__(self):
            calls.append(("fd-enter",))

        def __exit__(self, exc_type, exc_val, exc_tb):
            calls.append(("fd-exit",))

    monkeypatch.setattr(upload.ttnn, "allocate_tensor_on_device", _alloc)
    monkeypatch.setattr(upload.ttnn, "copy_host_to_device_tensor", _full)
    monkeypatch.setattr(upload.ttnn, "copy_host_to_device_tensor_partial", _partial)
    monkeypatch.setattr(upload.ttnn.device, "setup_fast_dispatch", lambda _device: _FDContext())
    monkeypatch.setattr(upload, "get_fd_grid", lambda _device: _crs(0, 0, 11, 9))

    class _Uploadable:
        def __init__(self, tensors):
            self._tensors = tensors
            self.received_map = None

        def backing_tensors(self):
            return self._tensors

        def with_device_tensors(self, tensor_map):
            self.received_map = tensor_map
            return self

    host = _FakeHostTensor("a", sharded=False)
    weights = _Uploadable([host])
    uploaded = upload.two_phase_upload(object(), weights)

    assert uploaded is weights
    assert weights.received_map == {upload.tensor_identity_key(host): "dev-spec-a"}
    assert calls == [("fd-enter",), ("full", "a", "dev-spec-a"), ("fd-exit",)]


def test_two_phase_upload_sharded_with_both_filters_calls_fd_then_sd_partial(monkeypatch):
    calls: list[tuple] = []

    monkeypatch.setattr(upload.ttnn, "allocate_tensor_on_device", lambda spec, _device: f"dev-{spec}")
    monkeypatch.setattr(
        upload.ttnn,
        "copy_host_to_device_tensor",
        lambda host_tensor, device_tensor: calls.append(("full", host_tensor.name)),
    )
    monkeypatch.setattr(
        upload.ttnn,
        "copy_host_to_device_tensor_partial",
        lambda host_tensor, device_tensor, core_filter: calls.append(
            ("partial", host_tensor.name, core_filter.num_cores())
        ),
    )

    class _FDContext:
        def __enter__(self):
            calls.append(("fd-enter",))

        def __exit__(self, exc_type, exc_val, exc_tb):
            calls.append(("fd-exit",))

    monkeypatch.setattr(upload.ttnn.device, "setup_fast_dispatch", lambda _device: _FDContext())
    monkeypatch.setattr(upload, "get_fd_grid", lambda _device: _crs(0, 0, 11, 9))

    class _Uploadable:
        def __init__(self, tensors):
            self._tensors = tensors

        def backing_tensors(self):
            return self._tensors

        def with_device_tensors(self, tensor_map):
            return tensor_map

    host = _FakeHostTensor("b", sharded=True, grid=_crs(0, 0, 12, 9))
    result = upload.two_phase_upload(object(), _Uploadable([host]))
    assert result == {upload.tensor_identity_key(host): "dev-spec-b"}

    assert calls == [
        ("fd-enter",),
        ("partial", "b", 120),
        ("fd-exit",),
        ("partial", "b", 10),
    ]


def test_two_phase_upload_sharded_fd_only_uses_full_fd_copy(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(upload.ttnn, "allocate_tensor_on_device", lambda spec, _device: f"dev-{spec}")
    monkeypatch.setattr(
        upload.ttnn,
        "copy_host_to_device_tensor",
        lambda host_tensor, device_tensor: calls.append(("full", host_tensor.name)),
    )
    monkeypatch.setattr(
        upload.ttnn,
        "copy_host_to_device_tensor_partial",
        lambda host_tensor, device_tensor, core_filter: calls.append(
            ("partial", host_tensor.name, core_filter.num_cores())
        ),
    )

    class _FDContext:
        def __enter__(self):
            calls.append(("fd-enter",))

        def __exit__(self, exc_type, exc_val, exc_tb):
            calls.append(("fd-exit",))

    monkeypatch.setattr(upload.ttnn.device, "setup_fast_dispatch", lambda _device: _FDContext())
    monkeypatch.setattr(upload, "get_fd_grid", lambda _device: _crs(0, 0, 11, 9))

    class _Uploadable:
        def __init__(self, tensors):
            self._tensors = tensors

        def backing_tensors(self):
            return self._tensors

        def with_device_tensors(self, tensor_map):
            return tensor_map

    host = _FakeHostTensor("c", sharded=True, grid=_crs(0, 0, 11, 9))
    upload.two_phase_upload(object(), _Uploadable([host]))

    assert calls == [("fd-enter",), ("full", "c"), ("fd-exit",)]


def test_two_phase_upload_sharded_with_none_shard_spec_falls_back_to_full_fd(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(upload.ttnn, "allocate_tensor_on_device", lambda spec, _device: f"dev-{spec}")
    monkeypatch.setattr(
        upload.ttnn,
        "copy_host_to_device_tensor",
        lambda host_tensor, device_tensor: calls.append(("full", host_tensor.name)),
    )
    monkeypatch.setattr(
        upload.ttnn,
        "copy_host_to_device_tensor_partial",
        lambda host_tensor, device_tensor, core_filter: calls.append(
            ("partial", host_tensor.name, core_filter.num_cores())
        ),
    )

    class _FDContext:
        def __enter__(self):
            calls.append(("fd-enter",))

        def __exit__(self, exc_type, exc_val, exc_tb):
            calls.append(("fd-exit",))

    monkeypatch.setattr(upload.ttnn.device, "setup_fast_dispatch", lambda _device: _FDContext())
    monkeypatch.setattr(upload, "get_fd_grid", lambda _device: _crs(0, 0, 11, 9))

    class _Uploadable:
        def __init__(self, tensors):
            self._tensors = tensors

        def backing_tensors(self):
            return self._tensors

        def with_device_tensors(self, tensor_map):
            return tensor_map

    host = _FakeHostTensor("d", sharded=True, grid=None, shard_spec_none=True)
    upload.two_phase_upload(object(), _Uploadable([host]))

    assert calls == [("fd-enter",), ("full", "d"), ("fd-exit",)]
