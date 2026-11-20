# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host cost of the basic per-call steps on K3's 8x4 Galaxy mesh (device opened as the K3 model opens it): tensor
allocation / deallocation (an L1 arena over the whole worker grid as the MoE block takes one a call, a small sharded L1
words tensor, a DRAM activation) and a cached op launch (ttnn.add, DRAM and L1) - host time a call, N back to back,
synced only around the batch. HOST_MICRO_N (default 200)."""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_moe_layer_bench import PLACEMENTS

N = int(os.environ.get("HOST_MICRO_N", "200"))


def _time(fn, n=N):
    """us a call of fn, n back to back (host only: synced before and after the batch, not inside), median of 5 batches."""
    per = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn()
        per.append((time.perf_counter_ns() - t0) / n / 1e3)
    return statistics.median(per)


def _grid_mc(mesh, rows, bt=ttnn.BufferType.L1):
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    shape = ttnn.Shape([grid.x * grid.y * rows, 16])
    return shape, ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, bt, ttnn.ShardSpec(crs, [rows, 16], ttnn.ShardOrientation.ROW_MAJOR)
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_host_micro(mesh_device, device_params):
    mesh = mesh_device
    res = {}

    def alloc_free(spec, held_n):
        shape, mc = spec

        def alloc():
            return ttnn.allocate_tensor_on_device(shape, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, mc)

        held = []
        t0 = time.perf_counter_ns()  # (one batch of held_n: they stack below each other, freed after)
        for _ in range(held_n):
            held.append(alloc())
        res_a = (time.perf_counter_ns() - t0) / held_n / 1e3
        t0 = time.perf_counter_ns()
        n_free = len(held)
        while held:
            ttnn.deallocate(held.pop())
        res_f = (time.perf_counter_ns() - t0) / n_free / 1e3
        # (allocation stacks held_n below each other: a lone alloc + free a call is what the block does)
        res_af = _time(lambda: ttnn.deallocate(alloc()))
        return res_a, res_f, res_af

    arena_rows = (1300 * 1024) // 64
    for name, spec, held_n in (
        ("L1 arena 1.3 MB a core (120 cores)", _grid_mc(mesh, arena_rows), 0),
        ("L1 words 64 B a core (120 cores)", _grid_mc(mesh, 1), 50),
        ("L1_SMALL words 64 B a core", _grid_mc(mesh, 1, ttnn.BufferType.L1_SMALL), 8),
    ):
        if name.startswith("L1 arena"):  # (one at a time: 50 of them do not fit)
            shape, mc = spec
            a = _time(
                lambda: ttnn.deallocate(
                    ttnn.allocate_tensor_on_device(shape, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, mc)
                )
            )
            res[name] = f"alloc + free {a:.1f} us"
            logger.info(f"HOST_MICRO {name}: {res[name]}")
            continue
        a, f, af = alloc_free(spec, held_n)
        res[name] = f"alloc {a:.1f} us, free {f:.1f} us, alloc + free {af:.1f} us"
        logger.info(f"HOST_MICRO {name}: {res[name]}")

    dram_shape = ttnn.Shape([1, 1, 640, 1792])
    a = _time(
        lambda: ttnn.deallocate(
            ttnn.allocate_tensor_on_device(dram_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        )
    )
    res["DRAM [640, 1792] bf16 tile"] = f"alloc + free {a:.1f} us"

    for mc_name, mc in (("DRAM", ttnn.DRAM_MEMORY_CONFIG), ("L1", ttnn.L1_MEMORY_CONFIG)):
        x = ttnn.from_torch(
            torch.randn(1, 1, 640, 1792),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=mc,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.deallocate(ttnn.add(x, x, memory_config=mc))  # (compile + cache)
        ttnn.synchronize_device(mesh)
        a = _time(lambda: ttnn.deallocate(ttnn.add(x, x, memory_config=mc)))
        ttnn.synchronize_device(mesh)
        res[f"ttnn.add [640, 1792] {mc_name} (cached)"] = f"{a:.1f} us a call (launch + output alloc / free)"
        out = ttnn.add(x, x, memory_config=mc)
        a = _time(lambda: ttnn.add(x, x, memory_config=mc, output_tensor=out))
        ttnn.synchronize_device(mesh)
        res[f"ttnn.add [640, 1792] {mc_name} into a preallocated output"] = f"{a:.1f} us a call (launch only)"
        ttnn.deallocate(out)
        ttnn.deallocate(x)

    for k, v in res.items():
        logger.info(f"HOST_MICRO {k}: {v}")

    spy = os.environ.get("HOST_MICRO_SPY")  # (a py-spy --native profile of the cached launch path: out file stem)
    if spy:
        import subprocess

        x = ttnn.from_torch(
            torch.randn(1, 1, 640, 1792),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        out = ttnn.add(x, x)
        ttnn.synchronize_device(mesh)
        prof = subprocess.Popen(
            [
                "py-spy",
                "record",
                "--native",
                "--rate",
                "200",
                "--format",
                "raw",
                "-o",
                f"{spy}_add.txt",
                "--pid",
                str(os.getpid()),
                "--duration",
                "25",
            ]
        )
        time.sleep(2)
        t_end = time.time() + 26
        n = 0
        while time.time() < t_end:
            ttnn.add(x, x, output_tensor=out)
            n += 1
            if n % 2000 == 0:
                ttnn.synchronize_device(mesh)
        ttnn.synchronize_device(mesh)
        prof.wait()
        logger.info(f"HOST_MICRO py-spy: {n} cached ttnn.add launches profiled -> {spy}_add.txt")
