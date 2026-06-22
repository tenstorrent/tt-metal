# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Deterministic debug harness for point_to_point. DO NOT DELETE — documents what could be
# validated on a single-device machine. The full sender->receiver transfer needs >= 2 mesh
# devices and is covered by the immutable acceptance test test_point_to_point.py.
#
# NOTE on environment: ccl_packet_dims requires an initialized fabric context, so the
# packet-framing / intermediate-spec paths can only be exercised on a mesh device opened
# WITH a fabric_config. ccl_dm_route + setup_fabric_connection + the transfer itself need a
# 2-device mesh and cannot run on a 1-device machine.

import pytest
import torch

import ttnn

from ttnn.operations.point_to_point.point_to_point import _intermediate_spec, _supported


_MATRIX = [
    (ttnn.bfloat16, ttnn.TILE_LAYOUT),
    (ttnn.float32, ttnn.TILE_LAYOUT),
    (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
]
_SHAPES = [(1, 1, 32, 32), (1, 1, 64, 128), (1, 1, 96, 32), (2, 3, 32, 64)]


def test_supported_axes_build():
    """SUPPORTED must build without circular-import / enum errors and cover the golden axes."""
    sup = _supported()
    assert ttnn.bfloat16 in sup["dtype"] and ttnn.bfloat8_b in sup["dtype"]
    assert ttnn.TILE_LAYOUT in sup["layout"] and ttnn.ROW_MAJOR_LAYOUT in sup["layout"]
    assert ttnn.Topology.Linear in sup["topology"] and ttnn.Topology.Ring in sup["topology"]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("ttnn_dtype, layout", _MATRIX)
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_packet_dims_and_intermediate_spec(mesh_device, ttnn_dtype, layout, shape):
    """ccl_packet_dims + intermediate TensorSpec construction + allocation (needs fabric ctx).

    Verifies the spec math the entry point relies on (esp. the bfloat8_b path where
    tt::datum_size throws and we derive packet_page_dim from the tile), and that the landing
    buffer actually allocates.
    """
    torch_dtype = torch.bfloat16 if ttnn_dtype in (ttnn.bfloat16, ttnn.bfloat8_b) else torch.float32
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    page = input_tensor.buffer_page_size()
    npages = input_tensor.buffer_num_pages()
    assert page > 0 and npages >= 1
    assert page % ttnn.get_l1_alignment() == 0 or page == ttnn.get_l1_alignment()

    pd = ttnn._ttnn.fabric.ccl_packet_dims(input_tensor.dtype, page, npages, ttnn.get_l1_alignment())
    assert pd.packet_size_bytes > 0 and pd.total_packets >= 1 and pd.pages_per_packet >= 1

    spec = _intermediate_spec(input_tensor, pd)
    intermediate = ttnn.allocate_tensor_on_device(spec, mesh_device)
    assert intermediate is not None

    # The landing buffer must hold total_packets packets of packet_size_bytes each.
    inter_bytes = intermediate.buffer_num_pages() * intermediate.buffer_page_size()
    assert inter_bytes >= pd.total_packets * pd.packet_size_bytes, (
        inter_bytes,
        pd.total_packets,
        pd.packet_size_bytes,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_single_device_python_paths(mesh_device):
    """Exercise the remaining device-touching Python APIs that do NOT need 2 devices."""
    shape = (1, 1, 64, 128)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.allocate_tensor_on_device(input_tensor.spec, mesh_device)
    assert tuple(out.shape) == tuple(input_tensor.shape)

    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
    sem = ttnn.create_global_semaphore(mesh_device, cores, 0)
    ttnn.synchronize_device(mesh_device)
    assert ttnn.get_global_semaphore_address(sem) > 0

    fid = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, 0))
    assert fid is not None
