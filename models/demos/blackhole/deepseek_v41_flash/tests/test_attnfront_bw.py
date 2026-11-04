# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bandwidth microbenchmark for the T=12 weight delivery: each bank core streams its wqkv shard (1.22 MB) and forwards every block to a partner core (modes: 0 none, 1 unicast, 2 multicast,
3 half by NCRISC/NOC0 + half by BRISC/NOC1). Kernel duration per op from the profiler digest (sumarm.py on FD_OUT)."""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attn_front import KDIR, TILE_BYTES_B8, blaze_weight, core_layout
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _hash

OUT = os.environ.get("FD_OUT", "/mnt/tt-data/ssinghal/dsv4-logs/attnfront/run_bw")
MODES = [int(m) for m in os.environ.get("BW_MODES", "0,1,2,3").split(",")]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 100_000_000}], indirect=True)
@pytest.mark.timeout(1200)
@torch.no_grad()
def test_attnfront_bw(mesh_device):
    md = mesh_device
    os.makedirs(OUT, exist_ok=True)
    w = blaze_weight(md, (torch.randn(1, 1, 5120, 1792) * 0.03), ttnn.bfloat8_b, None)
    bank_cores, allc = core_layout(md, 2)
    phys = [md.worker_core_from_logical_core(ttnn.CoreCoord(c.x, c.y)) for c in allc]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in allc])
    arms = {}
    for mode in MODES:
        ring = ttnn.CBDescriptor(
            total_size=3 * 80 * TILE_BYTES_B8,
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=0,
                    data_format=ttnn.bfloat8_b,
                    page_size=TILE_BYTES_B8,
                    tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
                )
            ],
        )
        kernels = []
        for b, c in enumerate(bank_cores):
            cs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y))])
            rt = ttnn.RuntimeArgs()
            rt[c.x][c.y] = [phys[2 * b + 1].x, phys[2 * b + 1].y]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/bw_reader.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cs,
                    compile_time_args=[0, b, b & 3, w.buffer_address(), mode, 0],
                    runtime_args=rt,
                    common_runtime_args=[],
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )
            rt2 = ttnn.RuntimeArgs()
            rt2[c.x][c.y] = [phys[2 * b + 1].x, phys[2 * b + 1].y]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/bw_brisc.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cs,
                    compile_time_args=[0, mode, 0],
                    runtime_args=rt2,
                    common_runtime_args=[],
                    config=ttnn.WriterConfigDescriptor(),
                )
            )
        prog = ttnn.ProgramDescriptor(
            kernels=kernels,
            semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=core_set, initial_value=0)],
            cbs=[ring],
        )
        prog.custom_program_hash = _hash(0xB77, mode, w.buffer_address())

        def run():
            ttnn.generic_op([w, w], prog)

        run()
        ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(4):
            run()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(md)
        arms[f"mode{mode}"] = dict(trace_id=int(tid))
        json.dump(dict(arms=arms, layer=-1), open(os.path.join(OUT, "layer_arms.json"), "w"))
    print("BW_DONE", flush=True)
