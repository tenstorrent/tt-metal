# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""custom_mm throughput on resident data: arms 'name:ct:calls:dtype' (default ct 1 / 7 and bfp8 / bfp4, 1120 tile products each); sumarm.py on FD_OUT gives the kernel time of the op."""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attn_front import KDIR
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _hash

OUT = os.environ.get("FD_OUT", "/mnt/tt-data/ssinghal/dsv4-logs/attnfront/run_mmrate")
ARMS = os.environ.get("MR_ARMS", "ct1_b8:1:14:bfp8,ct7_b8:7:2:bfp8,ct1_b4:1:14:bfp4,ct7_b4:7:2:bfp4,ct2_b8:2:7:bfp8")
TB = {"bfp8": 1088, "bfp4": 576}
DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 100_000_000}], indirect=True)
@pytest.mark.timeout(1200)
@torch.no_grad()
def test_mmrate(mesh_device):
    md = mesh_device
    os.makedirs(OUT, exist_ok=True)
    cores = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(md, ttnn.NOC.RISCV_0_default)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
    dummy = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
    arms = {}
    for spec in ARMS.split(","):
        name, ct, calls, dt = spec.split(":")
        ct, calls = int(ct), int(calls)
        tile = ttnn.Tile([8, 32])
        cbs = [
            ttnn.CBDescriptor(
                total_size=80 * 512,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=0, data_format=ttnn.bfloat16, page_size=512, tile=ttnn.TileDescriptor(tile)
                    )
                ],
            ),
            ttnn.CBDescriptor(
                total_size=80 * ct * TB[dt],
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=1,
                        data_format=DT[dt],
                        page_size=TB[dt],
                        tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
                    )
                ],
            ),
            ttnn.CBDescriptor(
                total_size=ct * 512,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=2, data_format=ttnn.bfloat16, page_size=512, tile=ttnn.TileDescriptor(tile)
                    )
                ],
            ),
        ]
        k = lambda src, ct_, cfg: ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/{src}",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=ct_,
            runtime_args=ttnn.RuntimeArgs(),
            common_runtime_args=[],
            config=cfg,
        )
        prog = ttnn.ProgramDescriptor(
            kernels=[
                k("bw_push.cpp", [0, 1, ct], ttnn.ReaderConfigDescriptor()),
                k(
                    "bw_compute.cpp",
                    [0, 1, 2, ct, calls, 1],
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=True
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        prog.custom_program_hash = _hash(0xB78, ct, calls, TB[dt])

        def run():
            ttnn.generic_op([dummy, dummy], prog)

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
        arms[name] = dict(trace_id=int(tid))
        json.dump(dict(arms=arms, layer=-1), open(os.path.join(OUT, "layer_arms.json"), "w"))
    print("MR_DONE", flush=True)
