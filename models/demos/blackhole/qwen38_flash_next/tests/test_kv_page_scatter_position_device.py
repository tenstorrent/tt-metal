# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""kv_page_scatter with a device position: the chunk lands at the lane's logical rows [P, P + rows) with P read from a
UINT32 [1, 1, 1, 1] device scalar; the whole pool equals the torch reference bitwise (rows outside the range unchanged)."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import qsa_paged_reference as ref
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.kv_page_scatter import kv_page_scatter

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held one-by-eight mesh"
)
MESH = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG


def up(mesh, tensor, dtype):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def host(tensor):
    replicas = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]
    assert all(torch.equal(replicas[0], other) for other in replicas[1:])
    return replicas[0]


@MESH
@PARAMS
@pytest.mark.parametrize("page_size", [128, 2048])
@pytest.mark.parametrize("rows", [32, 128, 2048])
@pytest.mark.parametrize("start", [0, 128, 96])  # 96: P % bs != 0 and P + rows crosses a page for bs = 128
@pytest.mark.parametrize("src_row0", [0, 64])
def test_kv_page_scatter_position(mesh_device, page_size, rows, start, src_row0):
    generator = torch.Generator().manual_seed(7000 + page_size + rows + start + src_row0)
    lane, lanes = 3, 4
    blocks = -(-(start + rows) // page_size // 8) * 8 or 8
    pages = 1 + lanes * blocks
    ids = 1 + torch.randperm(lanes * blocks, generator=generator)
    table = torch.zeros(32, blocks, dtype=torch.int32)
    table[:lanes] = ids.reshape(lanes, blocks).to(torch.int32)
    src_host = torch.randn(1, 1, src_row0 + rows, 512, generator=generator).to(torch.bfloat16)
    pool_host = torch.randn(1, 1, pages * page_size, 512, generator=generator).to(torch.bfloat16)
    expected = ref.pool_after_scatter_at(pool_host[0, 0], src_host[0, 0], table, lane, start, rows, page_size, src_row0)
    src = up(mesh_device, src_host, ttnn.bfloat16)
    pool = up(mesh_device, pool_host, ttnn.bfloat16)
    d_pt = up(mesh_device, table.reshape(1, 1, 32, blocks), ttnn.int32)
    position = up(mesh_device, torch.full((1, 1, 1, 1), start, dtype=torch.int64).to(torch.uint32), ttnn.uint32)
    kv_page_scatter(src, pool, d_pt, lane=lane, rows=rows, page_size=page_size, src_row0=src_row0, position=position)
    got = host(pool)[0, 0]
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))
