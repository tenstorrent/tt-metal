# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the ttnn.reshape issues the new implementation fixes, and for the policy switch."""

import pytest
import torch

import ttnn
from ttnn._ttnn.operations.data_movement import reshape_legacy, reshape_new, set_reshape_execution_policy


@pytest.fixture
def compare_policy():
    set_reshape_execution_policy("compare")
    yield
    set_reshape_execution_policy(None)


def exact_values(shape):
    """Distinct values that bfloat16 represents exactly, so any misplaced element is caught."""
    return ((torch.arange(torch.Size(shape).numel()) % 251 + 1) / 8).reshape(shape).bfloat16()


def one_shard_on_core_0(shard_shape):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def shard_dims(tensor):
    return [
        placement.dim if isinstance(placement, ttnn.PlacementShard) else None
        for placement in tensor.tensor_topology().placements()
    ]


# #19272: a row-major height shard wider than the rows pads each row; viewing it with a new shard shape read the
# padding as data.
@pytest.mark.parametrize("width", [8, 16])
def test_19272_padded_height_shard_keeps_rows(device, width):
    torch_input = exact_values((1, 1, 8, width))
    tensor = ttnn.from_torch(
        torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=one_shard_on_core_0([8, 16])
    )
    output = ttnn.reshape(tensor, (1, 2, 4, width))
    assert torch.equal(ttnn.to_torch(output), torch_input.reshape(1, 2, 4, width))


# #27843: uint8 inputs were rejected.
def test_27843_uint8(device):
    torch_input = torch.arange(64, dtype=torch.uint8).reshape(4, 16)
    tensor = ttnn.from_torch(torch_input, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.reshape(tensor, (1, 2, 2, 16))
    assert output.dtype == ttnn.uint8
    assert torch.equal(ttnn.to_torch(output).to(torch.uint8), torch_input.reshape(1, 2, 2, 16))


# #55237: two calls differing only in the padded shape shared a program and the second used the first's row stride.
def test_55237_padded_shape_is_part_of_the_program(device):
    torch_input = exact_values((1, 1, 30, 32))
    tensor = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    expected = torch_input.reshape(1, 1, 32, 30)
    for padded_width in (32, 64, 32):
        output = ttnn.reshape(tensor, ttnn.Shape([1, 1, 32, 30]), ttnn.Shape([1, 1, 32, padded_width]))
        assert list(output.padded_shape) == [1, 1, 32, padded_width]
        assert torch.equal(ttnn.to_torch(output), expected)


# #61727: a tiled reshape first compiled after a trace capture kept a mapping tensor inside the trace's region,
# which execute_trace then overwrote. The new implementation owns no device memory across calls.
@pytest.mark.parametrize("device_params", [{"trace_region_size": 50_000_000}], indirect=True)
@pytest.mark.parametrize("order", ["trace_first", "reshape_first"])
def test_61727_reshape_after_execute_trace(device, order):
    torch_input = torch.randn(64, 64).bfloat16()

    def reshape():
        x = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
        output = ttnn.reshape(x, (32, 128))
        assert torch.equal(ttnn.to_torch(output), torch_input.reshape(32, 128))

    if order == "reshape_first":
        reshape()
    x = ttnn.from_torch(torch.randn(1024, 1024).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.add(x, x)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    ttnn.add(x, x)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    if order == "trace_first":
        reshape()
    ttnn.execute_trace(device, trace_id, cq_id=0)
    reshape()
    ttnn.release_trace(device, trace_id)


# #18088: splitting the last dim into pairs (Flux rotary embedding) on a mesh.
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_18088_pairs_on_mesh(mesh_device, layout):
    torch_input = torch.randn(128, 64).bfloat16()
    tensor = ttnn.from_torch(
        torch_input, layout=layout, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    output = ttnn.reshape(tensor, (128, 32, 2))
    for device_output in ttnn.get_device_tensors(output):
        assert torch.equal(ttnn.to_torch(device_output), torch_input.reshape(128, 32, 2))


# #64322 / #52331: the output topology must name the dims of the new shape.
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_64322_topology_follows_the_sharded_dim(mesh_device, layout):
    torch_input = torch.randn(1, 1, 32, 64).bfloat16()
    tensor = ttnn.from_torch(
        torch_input, layout=layout, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3)
    )
    assert 3 in shard_dims(tensor)

    output = ttnn.reshape(tensor, (1, 32, 32))
    assert shard_dims(output) == [None if dim is None else 2 for dim in shard_dims(tensor)]
    composed = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
    assert torch.equal(composed, torch_input.reshape(1, 32, 64))


# Every behavior above must also hold, bit for bit, against legacy where legacy is right.
@pytest.mark.parametrize(
    "shape, new_shape, layout",
    [
        ((1, 1, 32, 3072), (1, 32, 16, 192), ttnn.TILE_LAYOUT),
        ((2, 3, 40, 50), (3, 2, 50, 40), ttnn.TILE_LAYOUT),
        ((1, 128, 32, 128), (1, 1, 128, 4096), ttnn.TILE_LAYOUT),
        ((3, 5, 7), (7, 15), ttnn.ROW_MAJOR_LAYOUT),
        ((128, 64), (128, 32, 2), ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_compare_policy_accepts_matching_results(device, compare_policy, shape, new_shape, layout):
    torch_input = exact_values(shape)
    tensor = ttnn.from_torch(torch_input, layout=layout, device=device)
    output = ttnn.reshape(tensor, new_shape)
    assert torch.equal(ttnn.to_torch(output), torch_input.reshape(new_shape))


def test_pinned_entry_points_agree(device):
    torch_input = exact_values((2, 3, 40, 50))
    tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    shape = ttnn.Shape([6, 40, 50])
    assert torch.equal(ttnn.to_torch(reshape_new(tensor, shape)), ttnn.to_torch(reshape_legacy(tensor, shape)))


def test_policy_rejects_unknown_values(expect_error):
    with expect_error(RuntimeError, "TT_METAL_EXECUTION_POLICY_RESHAPE"):
        set_reshape_execution_policy("fastest")
