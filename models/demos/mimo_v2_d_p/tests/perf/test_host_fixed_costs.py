# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed host costs around a prefill chunk call: the token H2D (from_torch onto the mesh), a synchronize on an idle
mesh, and a tiny op's enqueue -> sync round trip (median of 20)."""

import time

import torch

import ttnn
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id


def _med(fn, n=20):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[n // 2]


@MESH_PARAMS
def test_host_fixed_costs(mesh_device, device_params):
    sp, tp = tuple(mesh_device.shape)
    ids = torch.randint(0, 1000, (sp, 1, 2048), dtype=torch.int32)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, None))
    tok = lambda: ttnn.from_torch(
        ids, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=mapper
    )
    h2d = _med(lambda: tok().deallocate(True))
    h2d_sync = _med(lambda: (tok().deallocate(True), ttnn.synchronize_device(mesh_device)))
    sync = _med(lambda: ttnn.synchronize_device(mesh_device))
    a = ttnn.from_torch(torch.ones(1, 1, 32, 32), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    rt = _med(lambda: (ttnn.add(a, a).deallocate(True), ttnn.synchronize_device(mesh_device)))
    print(
        f"HOST_FIXED {mesh_id(mesh_device)}: token H2D {h2d:.2f} ms (+sync {h2d_sync:.2f}), idle sync {sync:.2f} ms, "
        f"tiny op round trip {rt:.2f} ms"
    )
