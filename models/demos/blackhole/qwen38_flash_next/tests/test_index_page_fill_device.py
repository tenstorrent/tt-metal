# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""index_page_fill on the held one-by-eight mesh: the whole index pool after the fill equals the torch reference bitwise
(touched tiles at the page-table tile-rows, every other tile unchanged), for block sizes 128 and 2048, one and sixteen
tile-rows, and start positions inside a block."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import qsa_paged_reference as ref
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.index_page_fill import index_page_fill

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held one-by-eight mesh"
)
MESH = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG


def up(mesh, tensor, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    if dtype == ttnn.uint32:
        tensor = tensor.to(torch.int64).to(torch.uint32)
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
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
@pytest.mark.parametrize(
    "block_size, tile_rows, start",
    [
        (128, 1, 0),
        (128, 1, 128),
        (128, 16, 0),
        (128, 16, 256),
        (2048, 1, 0),
        (2048, 1, 128),
        (2048, 16, 0),
        (2048, 16, 128),
        (2048, 16, 1920),
        (2048, 1, 2048 + 640),
        (2048, 16, 2048 + 640),
    ],
)
def test_index_page_fill(mesh_device, block_size, tile_rows, start):
    generator = torch.Generator().manual_seed(7000 + block_size + 31 * tile_rows + start)
    block_rows = block_size // 4
    lane = 3
    tpb = block_rows // 32
    max_blocks = -(-(-(-((start >> 7) + tile_rows) // tpb)) // 8) * 8  # blocks the fill reaches, up to a multiple of 8
    pages = 1 + torch.randperm(2 * max_blocks, generator=generator)
    table = torch.zeros(8, max_blocks, dtype=torch.int32)
    table[lane] = pages[:max_blocks].to(torch.int32)
    table[lane + 1] = pages[max_blocks:].to(torch.int32)
    num_blocks = 1 + 2 * max_blocks
    pool_host = torch.randn(num_blocks, 1, block_rows, 128, generator=generator).to(torch.bfloat16)
    src_host = torch.randn(1, 1, tile_rows * 32, 128, generator=generator).to(torch.bfloat16)
    expected = ref.index_pool_after_fill(pool_host, src_host[0, 0], table, lane, start, block_rows)

    pool = up(mesh_device, pool_host, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    src = up(mesh_device, src_host, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    d_pt = up(mesh_device, table.reshape(1, 1, 8, max_blocks), ttnn.int32)
    position = up(mesh_device, torch.full((1, 1, 1, 1), start, dtype=torch.int32), ttnn.uint32)
    index_page_fill(src, pool, d_pt, position, lane=lane, block_rows=block_rows)
    assert torch.equal(host(pool).view(torch.int16), expected.view(torch.int16))
