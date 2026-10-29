# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.7 (CKV_BFP4_PREPARED, DS41-135): ttnn.bringup.ckv_bfp4's host time a call, prepared=False (a fresh descriptor
through generic_op every call, the path before P.7) vs prepared=True (one prepared program, row0 a value slot), on the
run's mesh, as tt/compressor.AttnTable.append_tiles calls it (rounded=True, DRAM, a new row0 each call). Not a gate:
numbers for BREADCRUMBS.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \
        models/demos/deepseek_v41_flash_d_p/tt/perf_ckv_bfp4_host.py -s

Per mode and n: N warm calls (after WARM), each at another row0; host time = the Python call (dispatch is
asynchronous), device-synchronised before and after the timed loop. The tables of the two modes are compared byte
for byte at the end.
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import spec

WARM, N = 3, 20


@pytest.mark.parametrize("device_params", [harness.device_params(spec())], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
@pytest.mark.parametrize("n", [1280, 64])
def test_ckv_bfp4_host(mesh_device, n):
    hist = n * (WARM + N + 1)
    g = torch.Generator().manual_seed(n)
    x = torch.randn(1, 1, n, 512, generator=g).bfloat16()
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    ckv = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    res = {}
    tables = {}
    for prepared in (False, True):
        rows = ttnn.zeros([1, 1, hist, 320], dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
        tables[prepared] = rows
        ts = []
        for i in range(WARM + N):
            if i == WARM:
                ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            out = ttnn.bringup.ckv_bfp4(
                ckv, rows, i * n, rounded=True, memory_config=ttnn.DRAM_MEMORY_CONFIG, prepared=prepared
            )
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)
        res[prepared] = ts[WARM:]
    a, b = (ttnn.to_torch(ttnn.get_device_tensors(tables[p])[0]) for p in (False, True))
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), "prepared table != generic_op table"
    for p in (False, True):
        ms = [t * 1e3 for t in res[p]]
        print(
            f"CKV_BFP4_HOST n={n} prepared={p}: median {statistics.median(ms):.3f} ms, min {min(ms):.3f}, "
            f"max {max(ms):.3f} ({N} warm calls)"
        )
