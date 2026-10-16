# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device scalar and topology gates for the exact Qwen3.8 1x4 mesh."""

from __future__ import annotations

import inspect
import re

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    BLOCK_START_LANE_MASK,
    POSITION_INDEX_ROW_SHAPE,
    POSITION_SCALAR_SHAPE,
    Qwen38MeshContract,
    Qwen38TTNNDevicePosition,
    TensorPlacement,
)

PHYSICAL_IDS = (0, 1, 2, 3)
MESH_COORDS = ((0, 0), (0, 1), (0, 2), (0, 3))


class _BoundInteger:
    """Stand-in for an integer returned through a native Python binding."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


class PlacementReplicate:
    pass


class PlacementShard:
    def __init__(self, dim) -> None:
        self.dim = dim


class _Mesh:
    def __init__(self, *, shape=(1, 4), count=4, physical_ids=PHYSICAL_IDS) -> None:
        self.shape = shape
        self.count = count
        self.physical_ids = physical_ids

    def get_num_devices(self):
        return self.count

    def get_device_ids(self):
        return self.physical_ids


class _Topology:
    def __init__(self, *, distribution=(1, 4), coords=MESH_COORDS, shard_dim=3) -> None:
        self.distribution = distribution
        self.coords = coords
        self.shard_dim = shard_dim

    def distribution_shape(self):
        return self.distribution

    def mesh_coords(self):
        return self.coords

    def placements(self):
        return (PlacementReplicate(), PlacementShard(self.shard_dim))


class _Tensor:
    def __init__(self, *, mesh=None, topology=None) -> None:
        self.mesh = mesh if mesh is not None else _Mesh()
        self.topology = topology if topology is not None else _Topology()

    def device(self):
        return self.mesh

    def tensor_topology(self):
        return self.topology


@pytest.mark.parametrize("alias", (True, "1", 1.0))
def test_contract_mesh_shape_rejects_bool_string_and_float_aliases(alias, expect_error) -> None:
    with expect_error(ValueError, "mesh_shape.*must be an exact integer"):
        Qwen38MeshContract(PHYSICAL_IDS, mesh_shape=(alias, 4))


@pytest.mark.parametrize("alias", (False, "0", 0.0))
def test_contract_physical_order_rejects_bool_string_and_float_aliases(alias, expect_error) -> None:
    with expect_error(ValueError, "four distinct physical IDs"):
        Qwen38MeshContract((alias, 1, 2, 3))


@pytest.mark.parametrize("alias", (True, "1", 1.0))
def test_live_mesh_shape_rejects_bool_string_and_float_aliases(alias, expect_error) -> None:
    with expect_error(RuntimeError, "opened mesh shape.*must be an exact integer"):
        Qwen38MeshContract(PHYSICAL_IDS).validate_mesh(_Mesh(shape=(alias, 4)))


@pytest.mark.parametrize("alias", (True, "4", 4.0))
def test_live_mesh_count_rejects_bool_string_and_float_aliases(alias, expect_error) -> None:
    with expect_error(RuntimeError, "opened mesh device count must be an exact integer"):
        Qwen38MeshContract(PHYSICAL_IDS).validate_mesh(_Mesh(count=alias))


@pytest.mark.parametrize("alias", (False, "0", 0.0))
def test_live_mesh_order_rejects_bool_string_and_float_aliases(alias, expect_error) -> None:
    with expect_error(RuntimeError, "opened mesh physical IDs.*must be an exact integer"):
        Qwen38MeshContract(PHYSICAL_IDS).validate_mesh(_Mesh(physical_ids=(alias, 1, 2, 3)))


@pytest.mark.parametrize("alias", (True, "1", 1.0))
def test_tensor_distribution_rejects_bool_string_and_float_aliases(alias, expect_error) -> None:
    tensor = _Tensor(topology=_Topology(distribution=(alias, 4)))
    with expect_error(RuntimeError, "tensor distribution shape.*must be an exact integer"):
        Qwen38MeshContract(PHYSICAL_IDS).validate_tensor(
            tensor,
            placement=TensorPlacement.HIDDEN_SHARDED,
            shard_dim=3,
        )


@pytest.mark.parametrize("alias", (False, "0", 0.0))
def test_tensor_coordinates_reject_bool_string_and_float_aliases(alias, expect_error) -> None:
    coords = ((alias, 0), *MESH_COORDS[1:])
    tensor = _Tensor(topology=_Topology(coords=coords))
    with expect_error(RuntimeError, "tensor mesh coordinate.*must be an exact integer"):
        Qwen38MeshContract(PHYSICAL_IDS).validate_tensor(
            tensor,
            placement=TensorPlacement.HIDDEN_SHARDED,
            shard_dim=3,
        )


@pytest.mark.parametrize(
    ("alias", "expected_dim"),
    (
        (True, 1),
        ("3", 3),
        (3.0, 3),
    ),
)
def test_tensor_shard_dim_rejects_bool_string_and_float_aliases(alias, expected_dim, expect_error) -> None:
    tensor = _Tensor(topology=_Topology(shard_dim=alias))
    with expect_error(RuntimeError, "PlacementShard.dim must be an exact integer"):
        Qwen38MeshContract(PHYSICAL_IDS).validate_tensor(
            tensor,
            placement=TensorPlacement.HIDDEN_SHARDED,
            shard_dim=expected_dim,
        )


@pytest.mark.parametrize("alias", (True, "3", 3.0))
def test_requested_shard_dim_rejects_bool_string_and_float_aliases(alias, expect_error) -> None:
    with expect_error(ValueError, "shard_dim must be an exact integer"):
        Qwen38MeshContract(PHYSICAL_IDS).validate_tensor(
            _Tensor(),
            placement=TensorPlacement.HIDDEN_SHARDED,
            shard_dim=alias,
        )


def test_native_integer_protocol_values_remain_accepted() -> None:
    bound = _BoundInteger
    mesh = _Mesh(
        shape=(bound(1), bound(4)),
        count=bound(4),
        physical_ids=tuple(bound(value) for value in PHYSICAL_IDS),
    )
    topology = _Topology(
        distribution=(bound(1), bound(4)),
        coords=tuple(tuple(bound(value) for value in coord) for coord in MESH_COORDS),
        shard_dim=bound(3),
    )
    contract = Qwen38MeshContract(
        tuple(bound(value) for value in PHYSICAL_IDS),
        mesh_shape=(bound(1), bound(4)),
    )
    assert contract.physical_ids == PHYSICAL_IDS
    assert contract.mesh_shape == (1, 4)
    contract.validate_tensor(
        _Tensor(mesh=mesh, topology=topology),
        placement=TensorPlacement.HIDDEN_SHARDED,
        shard_dim=bound(3),
    )


# --- device position counter (position-generic decode body) ---------------------


class _FakeDeviceTensor:
    _next_id = 1

    def __init__(self, shape, dtype, layout, *, host=None) -> None:
        self.shape = tuple(shape)
        # Every tensor the position counter touches is ROW_MAJOR: no tile padding.
        self.padded_shape = tuple(shape)
        self.dtype = dtype
        self.layout = layout
        self.host = host
        self._id = _FakeDeviceTensor._next_id
        _FakeDeviceTensor._next_id += 1

    def tensor_id(self) -> int:
        return self._id


def _binary_ng_output(left, right) -> _FakeDeviceTensor:
    """Output metadata of a binary_ng op: the broadcast shape in the left operand's dtype and layout.

    ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:258-259
    (output dtype defaults to the input dtype) and :489 (compute_broadcasted_output);
    a Python scalar operand leaves the tensor's shape unchanged.
    """

    shape = left.shape
    if isinstance(right, _FakeDeviceTensor):
        assert len(left.shape) == len(right.shape)
        assert all(a == b or 1 in (a, b) for a, b in zip(left.shape, right.shape))
        shape = tuple(max(a, b) for a, b in zip(left.shape, right.shape))
    return _FakeDeviceTensor(shape, left.dtype, left.layout)


class _RecordingContract:
    def __init__(self) -> None:
        self.validated = []

    def validate_mesh(self, mesh_device) -> None:
        del mesh_device

    def validate_tensor(self, tensor, *, placement, **kwargs) -> None:
        del kwargs
        self.validated.append((tensor, placement))


def _patch_uploads(monkeypatch, uploads: list):
    def from_torch(host, **kwargs):
        uploads.append((host, kwargs))
        return _FakeDeviceTensor(tuple(host.shape), kwargs["dtype"], kwargs["layout"], host=host)

    monkeypatch.setattr(contracts_module.ttnn, "from_torch", from_torch)
    monkeypatch.setattr(
        contracts_module, "replicate_tensor_2d_mesh_mapper", lambda mesh_device: ("replicate", mesh_device)
    )


def test_device_position_allocates_three_replicated_uint32_row_major_constants(monkeypatch) -> None:
    uploads = []
    _patch_uploads(monkeypatch, uploads)
    contract = _RecordingContract()

    position = Qwen38TTNNDevicePosition.allocate("mesh", contract, position=7)

    # the position, the ones row, the block-start mask row and the rotary shift (0: a text sequence)
    assert [tuple(host.shape) for host, _ in uploads] == [(1, 1, 1, 1), (1, 1, 1, 32), (1, 1, 1, 32), (1, 1, 1, 1)]
    assert all(host.dtype == torch.uint32 for host, _ in uploads)
    assert [int(host.reshape(-1)[0].to(torch.int64)) for host, _ in uploads] == [7, 1, BLOCK_START_LANE_MASK, 0]
    assert all(bool(torch.all(host == host.reshape(-1)[0])) for host, _ in uploads)
    for _, kwargs in uploads:
        assert kwargs["dtype"] == ttnn.uint32
        assert kwargs["layout"] == ttnn.ROW_MAJOR_LAYOUT
        assert kwargs["device"] == "mesh"
        assert kwargs["memory_config"] == ttnn.DRAM_MEMORY_CONFIG
        assert kwargs["mesh_mapper"] == ("replicate", "mesh")
    assert (position.scalar.shape, position.ones_row.shape, position.block_start_mask_row.shape) == (
        POSITION_SCALAR_SHAPE,
        POSITION_INDEX_ROW_SHAPE,
        POSITION_INDEX_ROW_SHAPE,
    )
    assert position.shift.shape == POSITION_SCALAR_SHAPE and position.rope_shift == 0
    assert [placement for _, placement in contract.validated] == [TensorPlacement.REPLICATED] * 4
    assert BLOCK_START_LANE_MASK == 0xFFFFFFFC and BLOCK_START_LANE_MASK >= 2**31


@pytest.mark.parametrize("position", (-1, 2**32, True, 1.0, "3"))
def test_device_position_rejects_out_of_range_and_alias_positions(monkeypatch, position, expect_error) -> None:
    uploads = []
    _patch_uploads(monkeypatch, uploads)
    with expect_error(ValueError, "device position"):
        Qwen38TTNNDevicePosition.allocate("mesh", _RecordingContract(), position=position)
    assert uploads == []


def test_device_position_reset_is_one_host_write_into_the_resident_scalar(monkeypatch) -> None:
    uploads = []
    _patch_uploads(monkeypatch, uploads)
    writes = []
    monkeypatch.setattr(
        contracts_module.ttnn, "copy_host_to_device_tensor", lambda host, device: writes.append((host, device))
    )
    position = Qwen38TTNNDevicePosition.allocate("mesh", _RecordingContract())
    uploads.clear()

    position.reset(33)

    assert len(uploads) == 1 and len(writes) == 1
    host, kwargs = uploads[0]
    assert tuple(host.shape) == POSITION_SCALAR_SHAPE and host.dtype == torch.uint32
    assert int(host.reshape(-1)[0].to(torch.int64)) == 33
    assert "device" not in kwargs
    assert kwargs["dtype"] == ttnn.uint32 and kwargs["layout"] == ttnn.ROW_MAJOR_LAYOUT
    assert writes[0][0].host is host
    assert writes[0][1] is position.scalar
    source = inspect.getsource(Qwen38TTNNDevicePosition.reset)
    assert "ttnn.copy_host_to_device_tensor(host, self.scalar)" in source
    assert "ttnn.copy(" not in source and "ttnn.add(" not in source


def test_device_position_reset_writes_the_rotary_shift_when_it_changes(expect_error, monkeypatch) -> None:
    """An image prompt's hand-off: finish_prefill -> reset(P, S) writes the shift tensor and its mirror on the frozen
    dataclass (the escaped bug: a plain attribute assignment raised FrozenInstanceError inside the hand-off); a rewind
    inside the request keeps the shift with one write; a new text sequence writes it back to 0; the block-start rule
    refuses S > P & ~3 before any write; a stand-in without a shift tensor refuses a nonzero shift before any write."""

    uploads = []
    _patch_uploads(monkeypatch, uploads)
    writes = []
    monkeypatch.setattr(
        contracts_module.ttnn, "copy_host_to_device_tensor", lambda host, device: writes.append((host, device))
    )
    position = Qwen38TTNNDevicePosition.allocate("mesh", _RecordingContract())
    assert position.rope_shift == 0 and int(uploads[-1][0].reshape(-1)[0].to(torch.int64)) == 0
    uploads.clear()

    position.reset(48, 12)  # the hand-off of an image prompt: 48 tokens, shift 12
    assert [device for _, device in writes] == [position.scalar, position.shift]
    assert [int(host.host.reshape(-1)[0].to(torch.int64)) for host, _ in writes] == [48, 12]  # the uploads' hosts
    assert position.rope_shift == 12
    writes.clear()

    position.reset(50)  # a rewind inside the request: the shift is kept, one write
    assert [device for _, device in writes] == [position.scalar] and position.rope_shift == 12
    writes.clear()

    position.reset(0, 0)  # a new text sequence
    assert [device for _, device in writes] == [position.scalar, position.shift] and position.rope_shift == 0
    writes.clear()

    for bad_position, bad_shift in ((1, 1), (7, 5), (48, 49), (0, 1)):
        with expect_error(ValueError, match="rotary shift must be in"):
            position.reset(bad_position, bad_shift)
    assert writes == []
    assert position.reset(7, 4) is None and position.rope_shift == 4  # S == P & ~3: the limit is admitted
    writes.clear()

    stand_in = Qwen38TTNNDevicePosition(
        position.scalar, position.ones_row, position.block_start_mask_row, "mesh", _RecordingContract()
    )
    assert stand_in.shift is None and stand_in.rope_shift == 0
    with expect_error(RuntimeError, match="no rotary shift tensor"):
        stand_in.reset(8, 4)
    assert writes == []  # refused before the position write
    stand_in.reset(8)  # a text stand-in resets its position alone
    assert [device for _, device in writes] == [position.scalar]
    source = inspect.getsource(Qwen38TTNNDevicePosition.reset)
    assert 'object.__setattr__(self, "rope_shift", rope_shift)' in source  # the frozen dataclass's mirror


def test_device_position_advance_adds_one_then_copies_in_place_as_its_last_op(monkeypatch) -> None:
    uploads = []
    _patch_uploads(monkeypatch, uploads)
    position = Qwen38TTNNDevicePosition.allocate("mesh", _RecordingContract())
    calls = []

    def add(tensor, other, **kwargs):
        calls.append(("add", tensor, other, kwargs))
        return _binary_ng_output(tensor, other)

    def copy(source, target):
        calls.append(("copy", source, target))
        # ttnn/cpp/ttnn/operations/data_movement/copy/copy_nanobind.cpp:65: copies
        # input_a into input_b when shapes and memory layouts match and returns input_b.
        assert source.shape == target.shape and source.layout == target.layout
        return target

    monkeypatch.setattr(contracts_module.ttnn, "add", add)
    monkeypatch.setattr(contracts_module.ttnn, "copy", copy)
    monkeypatch.setattr(contracts_module.ttnn, "deallocate", lambda tensor: calls.append(("deallocate", tensor)))

    position.advance()

    assert [call[0] for call in calls] == ["add", "copy", "deallocate"]
    assert calls[0][1] is position.scalar and calls[0][2] == 1
    assert calls[0][3] == {"memory_config": ttnn.DRAM_MEMORY_CONFIG}
    advanced = calls[1][1]
    assert calls[1][2] is position.scalar and advanced is not position.scalar
    assert calls[2][1] is advanced
    source = inspect.getsource(Qwen38TTNNDevicePosition.advance)
    ttnn_calls = re.findall(r"ttnn\.(\w+)\(", source)
    assert ttnn_calls == ["add", "copy", "deallocate"]
    assert "ttnn.copy(advanced, self.scalar)" in source
    assert source.rstrip().endswith("ttnn.deallocate(advanced)")


def test_device_position_index_rows_are_exact_uint32_broadcasts_of_tensor_constants(monkeypatch) -> None:
    uploads = []
    _patch_uploads(monkeypatch, uploads)
    contract = _RecordingContract()
    position = Qwen38TTNNDevicePosition.allocate("mesh", contract)
    calls = []

    def binary(name):
        def operation(left, right, **kwargs):
            calls.append((name, left, right, kwargs))
            return _binary_ng_output(left, right)

        return operation

    monkeypatch.setattr(contracts_module.ttnn, "multiply", binary("multiply"))
    monkeypatch.setattr(contracts_module.ttnn, "bitwise_and", binary("bitwise_and"))

    index_row = position.index_row()
    block_start_row = position.block_start_index_row(index_row)

    assert [call[0] for call in calls] == ["multiply", "bitwise_and"]
    assert calls[0][1] is position.ones_row and calls[0][2] is position.scalar
    assert calls[1][1] is index_row and calls[1][2] is position.block_start_mask_row
    assert all(call[3] == {"memory_config": ttnn.DRAM_MEMORY_CONFIG} for call in calls)
    assert index_row.shape == block_start_row.shape == POSITION_INDEX_ROW_SHAPE
    assert index_row.dtype == block_start_row.dtype == ttnn.uint32
    assert contract.validated[-2:] == [
        (index_row, TensorPlacement.REPLICATED),
        (block_start_row, TensorPlacement.REPLICATED),
    ]
    # The mask is a tensor operand: the only literal >= 2**31 in the module is
    # the constant it is uploaded from, never a scalar operand of a device op.
    source = inspect.getsource(Qwen38TTNNDevicePosition)
    assert "0xFFFFFFFC" not in source and "BLOCK_START_LANE_MASK" in source
    assert "bitwise_and(index_row, self.block_start_mask_row" in source
    assert "multiply(self.ones_row, self.scalar" in source
    assert not re.search(r"ttnn\.\w+\([^)]*\b(?:4294967292|0xFFFFFFFC)", inspect.getsource(contracts_module))


def test_device_position_read_takes_coordinate_zero_only(monkeypatch) -> None:
    uploads = []
    _patch_uploads(monkeypatch, uploads)
    position = Qwen38TTNNDevicePosition.allocate("mesh", _RecordingContract())
    monkeypatch.setattr(contracts_module.ttnn, "get_device_tensors", lambda tensor: ("local0", "local1", "local2"))
    monkeypatch.setattr(
        contracts_module.ttnn,
        "to_torch",
        lambda local: torch.full((1, 1, 1, 1), 41 if local == "local0" else -1, dtype=torch.int64),
    )
    assert position.read() == 41
    released = []
    monkeypatch.setattr(contracts_module.ttnn, "deallocate", released.append)
    position.deallocate()
    assert released == [position.scalar, position.ones_row, position.block_start_mask_row, position.shift]
