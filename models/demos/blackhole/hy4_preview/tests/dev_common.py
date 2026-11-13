# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the standalone device scripts (python -m ...tests.dev_*): mesh open/close, PCC, real-activation loader."""


import ttnn

ACTS = "/mnt/tt-data/ssinghal/hy4-logs/program/layer-moe/acts/"


def p(*a):
    print("LM", *a, flush=True)


def open_mesh(trace=200_000_000):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    return ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=trace)


def close_mesh(md):
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def rep(md, t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])
