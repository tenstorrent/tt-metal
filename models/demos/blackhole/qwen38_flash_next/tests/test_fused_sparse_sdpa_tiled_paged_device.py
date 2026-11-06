# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Paged K/V gather of ``sparse_sdpa_tiled`` on the held one-by-eight mesh: the op on a pool + page table (one row of a
multi-row table) is bitwise equal to the unpaged op on the flat cache the page table describes."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import sparse_sdpa_tiled as sst
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.sparse_sdpa_tiled import synthetic

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held one-by-eight mesh"
)
MESH = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG


def host(tensor):
    copies = [ttnn.to_torch(local).float() for local in ttnn.get_device_tensors(tensor)]
    assert all(torch.equal(copies[0], other) for other in copies[1:])
    return copies[0]


def up(mesh, tensor, dtype):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


@MESH
@PARAMS
@pytest.mark.parametrize(
    "page_size,pages,position",
    [(128, 24, 1024), (128, 24, 2044), (2048, 8, 2048), (2048, 8, 14336), (128, 2056, 262144)],  # last: 256k context
)
@pytest.mark.parametrize("page_table_row", [0, 2])
def test_paged_gather_is_bitwise_equal_to_the_flat_gather(mesh_device, page_size, pages, position, page_table_row):
    rows, ids, users, spare = 64, 64, 3, 5
    T = pages * page_size
    inputs = synthetic.make_inputs(S=rows, T=T, IDS=ids, H=6, P=position, pattern="random", seed=page_size + position)
    generator = torch.Generator().manual_seed(position)
    num_blocks = pages + spare
    table = torch.zeros(users, pages, dtype=torch.int64)
    for u in range(users):
        table[u] = 1 + torch.randperm(num_blocks - 1, generator=generator)[:pages]  # never block 0
    pool = torch.zeros(1, 1, num_blocks * page_size, inputs.kv.shape[-1], dtype=torch.bfloat16)
    flat = inputs.kv.to(torch.bfloat16)
    for b in range(pages):
        start = int(table[page_table_row, b]) * page_size
        pool[0, 0, start : start + page_size] = flat[0, 0, b * page_size : (b + 1) * page_size]

    q, kv, block_ids, positions = synthetic.upload(inputs, mesh_device)
    pool_tt = up(mesh_device, pool, ttnn.bfloat16)
    table_tt = up(mesh_device, table.to(torch.int32).reshape(1, 1, users, pages), ttnn.int32)
    flat_out = paged_out = None
    try:
        flat_out = sst.sparse_sdpa_tiled(q, kv, block_ids, positions, config=sst.HIFI2)
        paged_out = sst.sparse_sdpa_tiled(
            q,
            pool_tt,
            block_ids,
            positions,
            config=sst.HIFI2,
            page_table=table_tt,
            page_size=page_size,
            page_table_row=page_table_row,
        )
        expected = host(flat_out)
        assert torch.isfinite(expected).all()
        assert torch.equal(host(paged_out), expected)
    finally:
        for tensor in (q, kv, block_ids, positions, pool_tt, table_tt, flat_out, paged_out):
            if tensor is not None:
                ttnn.deallocate(tensor)
