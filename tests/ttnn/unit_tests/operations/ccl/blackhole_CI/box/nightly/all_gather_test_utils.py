# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Shared helpers for the BH box all_gather device tests (test_all_gather_matrix.py, test_all_gather_stress.py).
#
# The tests reach each algorithm only through public levers (LEVERS below): the fabric config and the sub_core_grids
# argument. Which algorithm auto picks on the natural worker grid is pinned per case by the host gtest
# AllGatherPlan.DeviceTestAlgorithms (tests/ttnn/unit_tests/gtests/ccl/test_all_gather_plan.cpp), so a retune of the
# choice model cannot silently move a case off the algorithm it is meant to cover.

from dataclasses import dataclass, replace
from typing import Optional

import torch
import ttnn
from loguru import logger

from ttnn import ConcatMeshToTensor, ShardTensorToMesh

# BH's tuned fabric payload; 15232 B is the slower 4-slot router tier.
PAYLOAD = 15168

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
TILE = ttnn.TILE_LAYOUT
RM = ttnn.ROW_MAJOR_LAYOUT
BF16 = ttnn.bfloat16
BFP8 = ttnn.bfloat8_b
FP32 = ttnn.float32

LINE = ttnn.FabricConfig.FABRIC_1D
RING = ttnn.FabricConfig.FABRIC_1D_RING
NE = ttnn.FabricConfig.FABRIC_1D_NEIGHBOR_EXCHANGE
F2D = ttnn.FabricConfig.FABRIC_2D


def fabric_params(config, trace_region_size=None):
    router = ttnn._ttnn.fabric.FabricRouterConfig()
    router.max_packet_payload_size_bytes = PAYLOAD
    params = {"fabric_config": config, "fabric_router_config": router}
    if trace_region_size is not None:
        params["trace_region_size"] = trace_region_size
    return params


def grid(x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x1, y1))})


# sub_core_grids levers. Unicast places a mux and its workers on one core row per link-direction; multicast needs one
# core per link.
COLUMN = grid(0, 7)  # one core per row: unicast cannot place, multicast runs on 2 links at any size
ONE_CORE = grid(0, 0)  # multicast on 1 link
ROWS_OF_2 = grid(1, 3)  # 4 rows of mux + 1 worker: unicast with 1 worker per link-direction on 2 links
ROWS_OF_3_X2 = grid(2, 1)  # 2 rows of mux + 2 workers: unicast with 2 workers on 1 link

# lever id -> (fabric config, sub_core_grids). A case's own grid applies only where the lever sets none.
#   line, ring    auto on the natural worker grid (the host gtest pins which algorithm that is, per case)
#   *_mc          a one-column grid rules out unicast: multicast at any size
#   *_mc_1link    a single core: multicast on one link
#   ne            FABRIC_1D_NEIGHBOR_EXCHANGE routes one hop only, which rules out multicast: unicast at any size
#   ne_n1         as ne, with one worker per link-direction
#   ne_1link      as ne, with two workers on one link
LEVERS = {
    "line": (LINE, None),
    "ring": (RING, None),
    "line_mc": (LINE, COLUMN),
    "ring_mc": (RING, COLUMN),
    "line_mc_1link": (LINE, ONE_CORE),
    "ring_mc_1link": (RING, ONE_CORE),
    "ne": (NE, None),
    "ne_n1": (NE, ROWS_OF_2),
    "ne_1link": (NE, ROWS_OF_3_X2),
}


def with_lever(case, lever):
    """The case under a lever's sub_core_grids (its own grid where the lever sets none)."""
    lever_grid = LEVERS[lever][1]
    return case if lever_grid is None else replace(case, grid=lever_grid)


def _sharded(layout, buffer_type, shard_shape, num_cores):
    cores = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(8, 8), row_wise=True)
    return ttnn.MemoryConfig(layout, buffer_type, ttnn.ShardSpec(cores, shard_shape, ttnn.ShardOrientation.ROW_MAJOR))


def l1_height_sharded(shard_height, shard_width, num_cores):
    return _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, (shard_height, shard_width), num_cores)


def l1_width_sharded(shard_height, shard_width, num_cores):
    return _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, (shard_height, shard_width), num_cores)


def dram_width_sharded(shard_height, shard_width, num_cores):
    return _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, (shard_height, shard_width), num_cores)


def nd_sharded(buffer_type, shard_shape, num_cores=8):
    # A DRAM bank id is the core's x alone, so a DRAM grid is one row.
    rows = 8 if buffer_type == ttnn.BufferType.L1 else 1
    cores = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(8, rows), row_wise=True)
    spec = ttnn.NdShardSpec(
        ttnn.Shape(shard_shape), cores, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
    )
    return ttnn.MemoryConfig(buffer_type, spec)


@dataclass(frozen=True)
class Case:
    shape: list  # OUTPUT (gathered) shape
    dim: int
    layout: object = TILE
    dtype: object = BF16
    mem_in: object = DRAM
    mem_out: object = DRAM
    mesh: tuple = (1, 8)
    cluster_axis: Optional[int] = None
    grid: object = None  # sub_core_grids


def submesh(mesh_device, mesh):
    if tuple(mesh_device.shape) == tuple(mesh):
        return mesh_device
    return mesh_device.create_submesh(ttnn.MeshShape(*mesh))


def _mapper(mesh, c):
    if c.cluster_axis is None:
        return ShardTensorToMesh(mesh, dim=c.dim)
    dims = (None, c.dim) if c.cluster_axis == 1 else (c.dim, None)
    return ttnn.ShardTensor2dMesh(mesh, dims=dims, mesh_shape=tuple(mesh.shape))


def host_input(mesh, c, seed):
    """Host tensor distributed over the mesh, and the full torch tensor the gather must reproduce (bf16/fp32)."""
    torch.manual_seed(seed)
    full = torch.randn(c.shape, dtype=torch.float32 if c.dtype == FP32 else torch.bfloat16)
    return ttnn.from_torch(full, layout=c.layout, dtype=c.dtype, mesh_mapper=_mapper(mesh, c)), full


def device_input(mesh, c, seed):
    host, full = host_input(mesh, c, seed)
    tt_in = ttnn.to_device(host, mesh, memory_config=c.mem_in)
    if c.dtype != BFP8:
        return tt_in, full
    # Lossy dtype: the gather must reproduce exactly what the devices hold.
    assert c.cluster_axis is None
    return tt_in, ttnn.to_torch(tt_in, mesh_composer=ConcatMeshToTensor(mesh, dim=c.dim))


def sentinel_host(mesh, c):
    # -3 everywhere, so a region the gather never writes cannot pass.
    full = torch.full(c.shape, -3.0, dtype=torch.float32 if c.dtype == FP32 else torch.bfloat16)
    return ttnn.from_torch(full, layout=c.layout, dtype=c.dtype, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))


def sentinel_output(mesh, c):
    return ttnn.to_device(sentinel_host(mesh, c), mesh, memory_config=c.mem_out)


def gather(tt_in, c, out=None):
    return ttnn.all_gather(
        tt_in,
        dim=c.dim,
        cluster_axis=c.cluster_axis,
        memory_config=c.mem_out,
        output_tensor=out,
        sub_core_grids=c.grid,
    )


def check(tt_out, expected, label):
    for d, dev_out in enumerate(ttnn.get_device_tensors(tt_out)):
        got = ttnn.to_torch(dev_out)
        ok = got.shape == expected.shape and torch.equal(got, expected)
        if not ok and got.shape == expected.shape:
            bad = (got != expected).nonzero()
            logger.error(f"{label}: device {d}: {bad.shape[0]} mismatches, first at {bad[0].tolist()}")
        assert ok, f"{label}: device {d} mismatch (shape {tuple(got.shape)} vs {tuple(expected.shape)})"


def run_once(mesh_device, c, label, persistent=True, seed=0):
    """One gather on the case's (sub)mesh, bit-exact on every device; a persistent output is checked in place too."""
    mesh = submesh(mesh_device, c.mesh)
    tt_in, expected = device_input(mesh, c, seed)
    out = sentinel_output(mesh, c) if persistent else None
    tt_out = gather(tt_in, c, out)
    ttnn.synchronize_device(mesh)
    check(tt_out, expected, label)
    if out is not None:
        check(out, expected, f"{label} (persistent output)")
