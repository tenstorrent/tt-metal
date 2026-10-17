# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import gc

import torch

import ttnn


def test_mesh_coordinate_range_value_equality():
    a = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(1, 1))
    b = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(1, 1))
    c = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1))
    assert a is not b
    assert a == b and not (a != b)
    assert a != c
    assert hash(a) == hash(b)
    assert {a: 1}[b] == 1


def test_mesh_coordinate_range_set_value_equality():
    r = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(1, 1))
    a = ttnn.MeshCoordinateRangeSet(r)
    b = ttnn.MeshCoordinateRangeSet(r)
    assert a == b
    assert a != ttnn.MeshCoordinateRangeSet()


def _program_with_cb(face_geometry):
    fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)
    fmt.face_geometry = face_geometry
    cb = ttnn.CBDescriptor(
        total_size=2048,
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        format_descriptors=[fmt],
    )
    return ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[cb])


def test_generic_op_hash_includes_cb_face_geometry():
    default = ttnn.compute_program_descriptor_hash(_program_with_cb(None))
    assert default == ttnn.compute_program_descriptor_hash(_program_with_cb(None))
    assert default != ttnn.compute_program_descriptor_hash(_program_with_cb(ttnn.FaceGeometry(1, 4)))


def _device_tensor_owning_only_device_ref(data):
    device = ttnn.open_device(device_id=0)
    return ttnn.Tensor(
        data.flatten().tolist(), list(data.shape), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )


def test_tensor_ctor_with_memory_config_keeps_device_alive():
    data = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)
    tensor = _device_tensor_owning_only_device_ref(data)
    gc.collect()
    assert torch.equal(ttnn.to_torch(tensor), data)
    device = tensor.device()
    del tensor
    ttnn.close_device(device)
