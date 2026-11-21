# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MB.4: host cost of one ttnn.bringup.ckv_bfp4_rows call on the mesh (a measurement): the whole call, its geometry
key, and run_prepared alone, on a served-slab-like ND-sharded source [4, 1, 8160, 512] and a record buffer
[1, 1, 8160, 320]; plus the old path's slice + to_layout + ckv_bfp4 for the same rows. Mesh only (0 model loads).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/perf/test_ckv_rows_host.py"
"""

import sys
import time

import pytest

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
N = 200


def _t(fn, mesh, n=N):
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    host = (time.perf_counter() - t0) / n * 1e3
    ttnn.synchronize_device(mesh)
    wall = (time.perf_counter() - t0) / n * 1e3
    return round(host, 4), round(wall, 4)


@mesh_parametrize
def test_ckv_rows_host(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid
    from models.demos.deepseek_v41_flash_d_p.tt.served import alloc_served

    mesh = mesh_device
    R = 8160
    slab = alloc_served(mesh, 512, R, 32, 4, dedup=True)
    loc = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R, 320]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    mg = max_grid(mesh)
    mod = sys.modules.get("ttnn.bringup.ckv_bfp4_ttnn.ckv_bfp4")
    ttnn.bringup.ckv_bfp4_rows(slab, loc, R, 64, 0, max_grid=mg)
    mod = sys.modules["ttnn.bringup.ckv_bfp4_ttnn.ckv_bfp4"]
    out = {}
    for rows in (64, 1024, 8000 // 32 * 32):
        out[f"call_{rows}"] = _t(lambda: ttnn.bringup.ckv_bfp4_rows(slab, loc, 2 * R, rows, 0, max_grid=mg), mesh)
        out[f"direct_{rows}"] = _t(lambda: mod.ckv_bfp4_rows(slab, loc, 2 * R, rows, 0, max_grid=mg), mesh)
    out["geo_pair"] = _t(lambda: (mod._tensor_geo(slab), mod._tensor_geo(loc)), mesh)
    pp = next(iter(mod._ROWS_PREPARED.values()))[1]
    out["run_prepared"] = _t(lambda: ttnn.bringup.run_prepared(pp, [slab, loc], [2 * R, 32, 32, 0, 32]), mesh)
    # the old path for the same rows (a 2 R batch's rows: slice of a flat interleaved copy stands in for slice_rows)
    flat = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R, 512]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )

    def old(rows=1024):
        s = ttnn.slice(flat, [0, 0, 0, 0], [1, 1, rows, 512], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tl = ttnn.to_layout(s, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s)
        ttnn.bringup.ckv_bfp4(tl, loc, 0, prepared=True, compact=True, max_grid=mg)
        ttnn.deallocate(tl)

    out["old_1024"] = _t(old, mesh)
    print(f"CKV_ROWS_HOST ms a call (host enqueue, synced wall): {out}", flush=True)
