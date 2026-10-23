# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Shared helpers for the BH box all_gather device tests (test_all_gather_matrix.py, test_all_gather_stress.py).
#
# Algorithms are forced through the TTNN_ALL_GATHER_FORCE override (grammar in all_gather_plan.hpp, parse_force):
# "<auto|unicast|multicast|composite>[:key=value(,key=value)*]", keys n (unicast), links, k (packets per slot),
# rdelay, wdelay. Unset = auto.

from dataclasses import dataclass
from typing import Optional

import torch
import ttnn
from loguru import logger

from ttnn import ConcatMeshToTensor, ShardTensorToMesh

FORCE_ENV = "TTNN_ALL_GATHER_FORCE"

# Column id -> override value (None = unset, the production path).
FORCE = {
    "auto": None,
    "uc_n1": "unicast:n=1",
    "uc_n2": "unicast:n=2",
    "uc_n3": "unicast:n=3",
    "mc": "multicast",
    "composite": "composite",
}

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


def force_value(algo, rdelay=0, wdelay=0):
    value = FORCE[algo]
    opts = [f"{k}={v}" for k, v in (("rdelay", rdelay), ("wdelay", wdelay)) if v]
    if opts:
        assert value is not None and value != "composite", f"delays need a forced kernel algorithm, not {algo}"
        value += ("," if ":" in value else ":") + ",".join(opts)
    return value


def set_force(monkeypatch, algo, rdelay=0, wdelay=0):
    value = force_value(algo, rdelay, wdelay)
    if value is None:
        monkeypatch.delenv(FORCE_ENV, raising=False)
    else:
        monkeypatch.setenv(FORCE_ENV, value)


def grid(x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x1, y1))})


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
    num_links: Optional[int] = None


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
        num_links=c.num_links,
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
