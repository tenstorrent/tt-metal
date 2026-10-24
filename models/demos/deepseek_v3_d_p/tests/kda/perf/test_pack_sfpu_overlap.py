# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How much pack-thread FP32 SFPU work hides under math-thread matmul (a measurement tool, not a gate).

Runs kernels/pack_sfpu_overlap.cpp, a compute-only program, on every worker core of all 32 chips: a loop of
1 x 4 FP32-DEST HiFi2 matmul subblocks (kt = k_block) on the math thread, with sfpu_tiles FP32-accurate sigmoid tiles
per subblock on the pack thread. KDA_PSO_CASES is a JSON list of [k_block, sfpu_tiles, sfpu_mode]; sfpu_mode 1 is
sigmoid_tile_pack, 2 the same SFPU code without its wait for an idle math unit; k_block 0 runs the pack thread alone.
Each case runs KDA_PSO_SUBBLOCKS subblocks (default 4096 / k_block, 512 at k_block 0) and reports the slowest chip's
realtime-profiler duration (median of KDA_PSO_RUNS runs) as KDA_PSO= lines, with cycles per subblock at 1.35 GHz.
"""

from __future__ import annotations

import json
import os
import statistics

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _L1_SMALL_SIZE
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_WORKER_L1_SIZE
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

_CASES = json.loads(os.environ.get("KDA_PSO_CASES", "[]"))
_RUNS = int(os.environ.get("KDA_PSO_RUNS", "5"))
_SUBBLOCKS = os.environ.get("KDA_PSO_SUBBLOCKS")
_KERNEL = "models/demos/deepseek_v3_d_p/tests/kda/perf/kernels/pack_sfpu_overlap.cpp"
_BF16_TILE, _BF8_TILE, _SUB_W = 2048, 1088, 4


def _program(cores: ttnn.CoreRangeSet, k_block: int, subblocks: int, sfpu_tiles: int, mode: int):
    def cb(index, fmt, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    kt = max(k_block, 1)
    return ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL,
                core_ranges=cores,
                compile_time_args=[k_block, subblocks, sfpu_tiles, mode],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True),
            )
        ],
        semaphores=[],
        cbs=[
            cb(0, ttnn.bfloat16, _BF16_TILE, kt),
            cb(1, ttnn.bfloat8_b, _BF8_TILE, kt * _SUB_W),
            cb(16, ttnn.bfloat16, _BF16_TILE, _SUB_W),
        ],
    )


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=_L1_SMALL_SIZE, worker_l1_size=KDA_WORKER_L1_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
    ],
    indirect=True,
)
def test_pack_sfpu_overlap(mesh_device: ttnn.MeshDevice, device_params: dict) -> None:
    """Time each KDA_PSO_CASES matmul / pack-thread sigmoid mix."""
    if not _CASES:
        pytest.skip("KDA_PSO_CASES is empty")
    require_realtime_profiler("the pack-thread SFPU overlap microbenchmark")
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    # generic_op takes an input and an output tensor; the kernel touches neither.
    io = [
        ttnn.from_torch(
            torch.zeros(1, 1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(2)
    ]
    for k_block, sfpu_tiles, mode in _CASES:
        subblocks = int(_SUBBLOCKS) if _SUBBLOCKS else (4096 // k_block if k_block else 512)
        program = _program(cores, k_block, subblocks, sfpu_tiles, mode)

        def run():
            return ttnn.generic_op(io, program)

        run()  # compile
        durations = []
        for _ in range(_RUNS):
            _, programs = profile_realtime_program_merged(mesh_device, run)
            durations.append(max(p["duration_ns"] for p in programs.values()) / 1e3)
        us = statistics.median(durations)
        entry = {
            "k_block": k_block,
            "sfpu_tiles": sfpu_tiles,
            "mode": mode,
            "subblocks": subblocks,
            "us": us,
            "cycles_per_subblock": us * 1350 / subblocks,
            "runs_us": durations,
        }
        print("KDA_PSO=" + json.dumps(entry))
