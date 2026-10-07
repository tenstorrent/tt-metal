# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the MiMo attention mesh tests."""
import json
import os

import torch

import ttnn

OUT = os.environ.get("ATTN_OUT", "/tmp/attn_mesh.jsonl")


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("ATTN_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


def open_mesh(trace_mb=256):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(ttnn.MeshShape(4, 8), trace_region_size=trace_mb * 1024 * 1024)


def open_mesh_full(trace_mb=512):
    """mesh for the full model tests (same params as the smoke fixtures: FABRIC_1D_RING, l1_small 16384)"""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    return ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=trace_mb * 1024 * 1024)


def close_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def rows_mapper(mesh):
    return ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, None))


def host_rows(mesh, t, dtype=ttnn.bfloat16):
    """host tensor (no device) sharded over mesh rows, tile layout"""
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=rows_mapper(mesh))


def dev_rows(mesh, t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rows_mapper(mesh),
    )


def gather_all(mesh, t):
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape))
    ).float()  # [rows, cols, ...]


def gather_col(mesh, t, col=0):
    """read only mesh column `col` of every row: [rows, ...]"""
    rows, cols = tuple(mesh.shape)
    dts = ttnn.get_device_tensors(t)
    return torch.cat([ttnn.to_torch(dts[r * cols + col]).float() for r in range(rows)], 0)


def dram(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    mb = lambda b: round(b / 2**20, 1)
    return dict(
        alloc_mb_per_bank=mb(v.total_bytes_allocated_per_bank),
        free_mb_per_bank=mb(v.total_bytes_free_per_bank),
        banks=v.num_banks,
    )


class Traced:
    """capture fn() once (after an eager warmup) and replay; `out` is the persistent output tensor"""

    def __init__(self, mesh, fn):
        self.mesh = mesh
        fn().deallocate(True)  # compile
        ttnn.synchronize_device(mesh)
        self.tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.out = fn()
        ttnn.end_trace_capture(mesh, self.tid, cq_id=0)
        ttnn.synchronize_device(mesh)

    def replay(self, block=True):
        ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=block)
        return self.out

    def release(self):
        ttnn.release_trace(self.mesh, self.tid)
