# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compute-only throughput of custom_mm (tt/skinny_kernels/sk_bench_compute.cpp): ns per tile product for MT (row tiles) x CT (ct_dim) x mode, one core of one chip (8 identical cores).
Env: SB_CFGS ';'-separated 'name:KT:CT:MT:mode:fp32' (default grid), SB_R (repetitions, 8), SB_NG (groups, 8), FD_OUT.
The zone B_LOOP duration is read from the raw device log by dsv4-logs/skinnymm/zbench.py."""

import json
import os

import pytest
import torch

import ttnn

OUT = os.environ.get("FD_OUT", "/mnt/tt-data/ssinghal/dsv4-logs/skinnymm/run_bench")
KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tt", "skinny_kernels")
R = int(os.environ.get("SB_R", "8"))
NG = int(os.environ.get("SB_NG", "8"))
DEFAULT = ";".join(
    f"m{m}c{c}k{k}f{f}x{x}:{k}:{c}:{m}:{x}:{f}"
    for f in (1,)
    for m in (1, 2)
    for c in (1, 2, 4)
    for k in (32, 128)
    for x in (0, 1)
    if m * c <= 4 and k * c <= 256
)
CFGS = os.environ.get("SB_CFGS") or DEFAULT


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 100_000_000}], indirect=True)
@pytest.mark.timeout(1200)
@torch.no_grad()
def test_skinny_bench(mesh_device):
    md = mesh_device
    os.makedirs(OUT, exist_ok=True)
    cores = [ttnn.CoreCoord(0, 0)]
    cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    RING = 256
    dummy = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    res = {}
    for spec in CFGS.split(";"):
        name, K, C, M, X, F = spec.split(":")
        K, C, M, X, F = int(K), int(C), int(M), int(X), int(F)
        tiny = lambda idx, pages: ttnn.CBDescriptor(
            total_size=pages * 512,
            core_ranges=cs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=idx,
                    data_format=ttnn.bfloat16,
                    page_size=512,
                    tile=ttnn.TileDescriptor(ttnn.Tile([8, 32])),
                )
            ],
        )
        cbs = [
            tiny(0, K * M),
            ttnn.CBDescriptor(
                total_size=RING * 1088,
                core_ranges=cs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=1,
                        data_format=ttnn.bfloat8_b,
                        page_size=1088,
                        tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
                    )
                ],
            ),
            tiny(2, 8),
        ]
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/sk_bench_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[0, 1, 2, K, C, NG, M, R, F, RING, X],
            runtime_args=ttnn.RuntimeArgs(),
            common_runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=bool(F), math_approx_mode=True
            ),
        )
        prog = ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)
        prog.custom_program_hash = (0x5B << 40) | (hash((K, C, NG, M, R, F, X)) & ((1 << 40) - 1))
        ttnn.generic_op([dummy, dummy], prog)
        ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        res[name] = dict(K=K, C=C, M=M, X=X, F=F, R=R, NG=NG, products=R * NG * C * K)
        print("BENCH " + json.dumps({name: res[name]}), flush=True)
    json.dump(res, open(os.path.join(OUT, "bench.json"), "w"))
    print("SB_DONE", flush=True)
