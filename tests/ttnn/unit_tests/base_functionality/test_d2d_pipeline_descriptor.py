# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Python side of D2DStreamService.export_pipeline_descriptor on one process: the receivers grouped
# by boundary, the file it writes, and the file handle that removes it.

import os

import pytest

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole

_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
_METADATA_SIZE_BYTES = 16
_NUM_GATES = 4


def _service_id(name: str) -> str:
    return f"pytest_{name}_{os.getpid()}"


def test_export_without_receivers():
    descriptor = ttnn.D2DStreamService.export_pipeline_descriptor(_service_id("empty"), [], root=0)
    assert descriptor is not None
    path = descriptor.path
    assert path == ttnn.d2d_pipeline_descriptor_path(_service_id("empty"))

    desc = ttnn.D2DPipelineDescriptor.wait_and_read(path)
    assert len(desc.ranks) == 1
    assert desc.ranks[0].rank == 0
    assert desc.ranks[0].host_name
    assert desc.ranks[0].boundaries == []

    del descriptor
    assert not os.path.exists(path)


def test_export_rejects_bad_arguments(expect_error):
    with expect_error(RuntimeError, "null receiver"):
        ttnn.D2DStreamService.export_pipeline_descriptor(_service_id("none"), [[None]], root=0)
    with expect_error(RuntimeError, "has no receivers"):
        ttnn.D2DStreamService.export_pipeline_descriptor(_service_id("empty_boundary"), [[]], root=0)
    with expect_error(RuntimeError, "outside world"):
        ttnn.D2DStreamService.export_pipeline_descriptor(_service_id("root"), [], root=1)


def test_wait_and_read_times_out(expect_error):
    with expect_error(RuntimeError, "Timeout waiting for descriptor file"):
        ttnn.D2DPipelineDescriptor.wait_and_read(ttnn.d2d_pipeline_descriptor_path(_service_id("missing")), 10)


def _create_pair(sender_mesh, receiver_mesh, *, with_extras: bool):
    global_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, 32, 512]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    mapper = ttnn.create_mesh_mapper(
        sender_mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()])
    )
    return ttnn.D2DStreamService.create_pair(
        sender_mesh=sender_mesh,
        receiver_mesh=receiver_mesh,
        global_spec=global_spec,
        mapper=mapper,
        fifo_size_bytes=4096,
        sender_worker_cores=_WORKER_CORES,
        receiver_worker_cores=_WORKER_CORES,
        metadata_size_bytes=_METADATA_SIZE_BYTES if with_extras else 0,
        # Lease mode, like the prefill runner: two pairs on one chip pair can't both own its link.
        share_fabric_links=True,
        stage_gate=ttnn.D2DStageGateConfig(_NUM_GATES, 0, 4) if with_extras else None,
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_export_two_stages(mesh_device):
    if not is_blackhole():
        pytest.skip("D2DStreamService needs Blackhole (or UBB Galaxy) service cores")

    def submesh(row, col):
        return mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(row, col))

    # Two pipeline stages on this rank, one per mesh row. The first boundary has two channels: one
    # with metadata and gates, and a plain one (like the prefill runner's indexer indices). The
    # second boundary has one plain channel.
    first_sender_mesh, first_receiver_mesh = submesh(0, 0), submesh(0, 1)
    sender, receiver = _create_pair(first_sender_mesh, first_receiver_mesh, with_extras=True)
    side_sender, side_receiver = _create_pair(first_sender_mesh, first_receiver_mesh, with_extras=False)
    second_sender, second_receiver = _create_pair(submesh(1, 0), submesh(1, 1), with_extras=False)

    rd = receiver.describe()
    assert rd.sender_rank == 0
    assert rd.metadata_size_bytes == _METADATA_SIZE_BYTES
    assert rd.metadata_addr == receiver.get_metadata_addr()
    assert rd.stage_gate.num_gates == _NUM_GATES
    assert len(rd.coords) == 1
    coord = rd.coords[0]
    gate = receiver.get_stage_gate_descriptor(coord.service_core.core.device_coord)
    assert coord.stage_gate_base_addr == gate.gate_base_addr
    assert coord.stage_gate_base_addr + rd.stage_gate_stride_bytes == gate.gate_addr(1)
    assert coord.metadata_core is not None
    assert coord.metadata_core.core.core_coord == _WORKER_CORES.start

    side = side_receiver.describe()
    second = second_receiver.describe()
    for plain in (side, second):
        assert plain.metadata_size_bytes == 0
        assert plain.metadata_addr == 0
        assert plain.stage_gate.num_gates == 0
        assert plain.stage_gate_stride_bytes == 0
        assert plain.coords[0].metadata_core is None
        assert plain.coords[0].stage_gate_base_addr == 0
    assert side.coords[0].service_core != coord.service_core

    descriptor = ttnn.D2DStreamService.export_pipeline_descriptor(
        _service_id("two_stages"), [[receiver, side_receiver], [second_receiver]], root=0
    )
    path = descriptor.path
    desc = ttnn.D2DPipelineDescriptor.wait_and_read(path)
    assert len(desc.ranks) == 1
    boundaries = desc.ranks[0].boundaries
    assert [b.receivers for b in boundaries] == [[rd, side], [second]]

    del descriptor
    assert not os.path.exists(path)
