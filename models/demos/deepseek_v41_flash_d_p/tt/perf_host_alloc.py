# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.15 (HOST_ALLOC): host time a call of an op with its output allocated by the op (+ the caller's deallocate) vs
written into a preallocated output, on the run's mesh. Not a gate: numbers for BREADCRUMBS.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \
        models/demos/deepseek_v41_flash_d_p/tt/perf_host_alloc.py -s

Each line: the median host time of N warm calls (the device synchronised before and after each timed loop).
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import spec

WARM, N = 5, 40
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _time(mesh, fn):
    for _ in range(WARM):
        fn()
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.synchronize_device(mesh)
    return statistics.median(ts)


@pytest.mark.parametrize("device_params", [harness.device_params(spec())], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_host_alloc(mesh_device):
    mesh = mesh_device
    rep = ttnn.ReplicateTensorToMesh(mesh)
    g = torch.Generator().manual_seed(0)

    def up(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            torch.randn(*shape, generator=g),
            dtype=dtype,
            layout=layout,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=rep,
        )

    dealloc = ttnn._ttnn.operations.core.deallocate
    x = up([1, 1, 640, 1024])
    w = up([1, 1, 1024, 1024], ttnn.bfloat8_b)
    x32 = up([1, 1, 640, 1024], ttnn.float32)
    rows = {}

    def alloc_free():
        dealloc(
            ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 640, 1024]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, DRAM)
        )

    rows["allocate + deallocate [640, 1024] bf16 DRAM"] = _time(mesh, alloc_free)
    l1 = ttnn.L1_MEMORY_CONFIG

    def alloc_free_l1():
        dealloc(ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 1024]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, l1))

    rows["allocate + deallocate [32, 1024] bf16 L1 interleaved"] = _time(mesh, alloc_free_l1)
    out_mm = ttnn.linear(x, w, dtype=ttnn.bfloat16, memory_config=DRAM)

    rows["linear (op allocates) + deallocate"] = _time(
        mesh, lambda: dealloc(ttnn.linear(x, w, dtype=ttnn.bfloat16, memory_config=DRAM))
    )
    rows["linear into optional_output_tensor"] = _time(
        mesh, lambda: ttnn.linear(x, w, dtype=ttnn.bfloat16, memory_config=DRAM, optional_output_tensor=out_mm)
    )
    out_tc = ttnn.typecast(x32, ttnn.bfloat16, memory_config=DRAM)
    rows["typecast (op allocates) + deallocate"] = _time(
        mesh, lambda: dealloc(ttnn.typecast(x32, ttnn.bfloat16, memory_config=DRAM))
    )
    rows["typecast into output_tensor"] = _time(mesh, lambda: ttnn.typecast(x32, ttnn.bfloat16, output_tensor=out_tc))
    out_mul = ttnn.multiply(x32, x32, dtype=ttnn.float32, memory_config=DRAM)
    rows["multiply (op allocates) + deallocate"] = _time(
        mesh, lambda: dealloc(ttnn.multiply(x32, x32, dtype=ttnn.float32, memory_config=DRAM))
    )
    rows["multiply into output_tensor"] = _time(
        mesh, lambda: ttnn.multiply(x32, x32, dtype=ttnn.float32, memory_config=DRAM, output_tensor=out_mul)
    )
    rows["to_layout ROW_MAJOR + deallocate"] = _time(
        mesh, lambda: dealloc(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM))
    )
    rows["slice + deallocate"] = _time(
        mesh, lambda: dealloc(ttnn.slice(x, [0, 0, 0, 0], [1, 1, 640, 512], memory_config=DRAM))
    )
    out_sl = ttnn.slice(x, [0, 0, 0, 0], [1, 1, 640, 512], memory_config=DRAM)
    rows["slice into output_tensor"] = _time(
        mesh, lambda: ttnn.slice(x, [0, 0, 0, 0], [1, 1, 640, 512], memory_config=DRAM, output_tensor=out_sl)
    )
    for k, v in rows.items():
        print(f"P15HOST {k:55s} {v:7.1f} us", flush=True)
