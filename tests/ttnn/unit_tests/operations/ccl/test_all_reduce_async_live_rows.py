# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""all_reduce_async (persistent-buffer overload) live_rows_packing contract.

With live_rows_packing=True and an eligible input (bf16 32x32 tiles, one 32-row tile row per shard, <= 16 logical
rows, even tiles per shard, every link's tile count a multiple of 4) the op sends only rows 0..15 of each tile, four
half-tiles per fabric packet. Rows 0..R-1 of the result must be byte-identical to the default path and match the torch
sum. Ineligible inputs (32 rows, odd tiles per shard, a link tile count that is not a multiple of 4) must fall back to
the default path: the whole result byte-identical.

Run: pytest tests/ttnn/unit_tests/operations/ccl/test_all_reduce_async_live_rows.py -v
"""

import pytest
import torch
from loguru import logger

import ttnn


def _width_mem(grid_x, grid_y, shard_w):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _run(mesh_device, full, mem, buf_mem, sem, num_links, topology, live_rows_packing):
    tp = full.shape[1]
    dim = full.shape[-1]
    inp = ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=mem,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )
    buf = ttnn.from_torch(
        torch.zeros([1, 1, ttnn.TILE_SIZE, dim * tp]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=buf_mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    extra = dict(live_rows_packing=True) if live_rows_packing else {}
    out = ttnn.experimental.all_reduce_async(
        inp,
        buf,
        cluster_axis=1,
        mesh_device=mesh_device,
        multi_device_global_semaphore=sem,
        topology=topology,
        num_links=num_links,
        memory_config=mem,
        **extra,
    )
    ttnn.synchronize_device(mesh_device)
    res = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(out)]
    for t in (out, buf, inp):
        ttnn.deallocate(t)
    return res


@pytest.mark.parametrize(
    "dim, grid, num_links, eligible",
    [
        (5120, (10, 4), 2, True),  # Qwen3.8 decode RES40: 4 tiles/core, 80 tiles per link
        (5120, (10, 4), 1, True),
        (960, (10, 1), 1, False),  # 3 tiles/core (odd) and 30 tiles on the link: full-tile fallback
    ],
    ids=["res40-2link", "res40-1link", "odd-shard-fallback"],
)
@pytest.mark.parametrize("rows", [1, 8, 16, 32])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 8), id="1x8")], indirect=True)
def test_all_reduce_async_live_rows(mesh_device, dim, grid, num_links, eligible, rows):
    tp = list(mesh_device.shape)[1]
    ncores = grid[0] * grid[1]
    mem = _width_mem(grid[0], grid[1], dim // ncores)
    buf_mem = _width_mem(grid[0], grid[1], dim * tp // ncores)
    sg = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(sg.x - 1, sg.y - 1))})
    sem = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    torch.manual_seed(rows)
    full = torch.randn(1, tp, rows, dim).to(torch.bfloat16)
    ref = full.float().sum(dim=1, keepdim=True)

    off = _run(mesh_device, full, mem, buf_mem, sem, num_links, ttnn.Topology.Linear, False)
    on = _run(mesh_device, full, mem, buf_mem, sem, num_links, ttnn.Topology.Linear, True)
    for dev, (o_off, o_on) in enumerate(zip(off, on)):
        assert torch.equal(o_off, o_on), (
            f"device {dev}: live_rows_packing changed rows 0..{rows - 1} "
            f"(max|d|={(o_off.float() - o_on.float()).abs().max().item():.3e})"
        )
        err = (o_on.float() - ref).abs().max().item()
        assert err <= 0.05 * ref.abs().max().item(), f"device {dev}: |all_reduce - torch sum| = {err:.3e}"
    logger.info(f"dim={dim} grid={grid} links={num_links} rows={rows}: packing {'on' if eligible else 'fallback'} OK")
