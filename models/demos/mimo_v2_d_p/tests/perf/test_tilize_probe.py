# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-core tilize throughput: row-major bf16 -> bfp8 / bf16 tiles, blocks of 32 rows x W tiles (zone TZB)."""

import os

import pytest
import torch

import ttnn
from models.demos.mimo_v2_d_p.tests.perf.test_stream_matmul import BF8_TILE, _crs

KDIR = "models/demos/mimo_v2_d_p/tests/perf/kernels/stream_mm"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("out_fmt", os.environ.get("MIMO_TZ_FMT", "bf8,bf16").split(","))
@pytest.mark.parametrize("w", [int(v) for v in os.environ.get("MIMO_TZ_W", "8,32").split(",")])
def test_tilize_probe(device, w, out_fmt):
    n = 256 * 8 // w  # 2048 tiles
    core = ttnn.CoreCoord(0, 0)
    crs = _crs([core])
    dt, page = (ttnn.bfloat8_b, BF8_TILE) if out_fmt == "bf8" else (ttnn.bfloat16, 2048)
    fmt = lambda i, d_, p_: [ttnn.CBFormatDescriptor(buffer_index=i, data_format=d_, page_size=p_)]
    cbs = [
        ttnn.CBDescriptor(total_size=2 * w * 2048, core_ranges=crs, format_descriptors=fmt(0, ttnn.bfloat16, 2048)),
        ttnn.CBDescriptor(total_size=2 * w * page, core_ranges=crs, format_descriptors=fmt(16, dt, page)),
    ]
    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
    FP = ttnn.KernelDescriptor.SourceType.FILE_PATH
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/tzb_dm.cpp",
            source_type=FP,
            core_ranges=crs,
            compile_time_args=[w, n, 1],
            runtime_args=[],
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/tzb_dm.cpp",
            source_type=FP,
            core_ranges=crs,
            compile_time_args=[w, n, 0],
            runtime_args=[],
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/tzb_compute.cpp",
            source_type=FP,
            core_ranges=crs,
            compile_time_args=[w, n],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
        ),
    ]
    a = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for _ in range(3):
        ttnn.generic_op([a, b], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    ttnn.synchronize_device(device)
