# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Repro (NOT yet run) for noc_async_read_barrier_with_trid returning before its read has landed: see README.md.

One core reads K DRAM pages per batch on one transaction id, barriers on the id, and checks that every destination's
last word arrived. Expected from the theory: failures with the plain trid barrier (on the batch's last read),
none with DRAIN_CMDBUF or GLOBAL_BARRIER. Untested skeleton: the generic_op plumbing follows
tests/ttnn/unit_tests/base_functionality/test_cb_address_offset.py and may need small fixes on first run."""

import os

import pytest
import torch
from loguru import logger

import ttnn

KERNEL = os.path.join(os.path.dirname(__file__), "trid_read_race.cpp")
NPAGES = 64
ITERS = int(os.environ.get("TRID_RACE_ITERS", "100000"))


def _run(device, k, read_bytes, defines):
    words = read_bytes // 4
    host_in = (torch.arange(NPAGES, dtype=torch.int64).unsqueeze(1) + 1).expand(NPAGES, words).to(torch.int32)
    tt_in = ttnn.from_torch(
        host_in, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.from_torch(
        torch.zeros(1, 16, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    cb = ttnn.CBDescriptor(
        total_size=k * read_bytes,
        core_ranges=core,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=read_bytes)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=[0, k, read_bytes, NPAGES, ITERS],
        defines=[(d, "1") for d in defines],
        runtime_args=[[[tt_in.buffer_address(), tt_out.buffer_address()]]],
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
        ),
    )
    pd = ttnn.ProgramDescriptor(cbs=[cb], kernels=[kernel])
    mesh_pd = ttnn.MeshProgramDescriptor()
    coord = ttnn.MeshCoordinate(0, 0)
    mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = pd
    ttnn.generic_op([tt_in, tt_out], mesh_pd)
    ttnn.synchronize_device(device)
    r = ttnn.to_torch(tt_out).flatten().tolist()
    return dict(fails=r[0], iters=r[1], first_iter=r[2], first_j=r[3], fails_last=r[4], fails_other=r[5])


@pytest.mark.parametrize("read_bytes", [64, 2048, 14336, 32768])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("barrier", ["trid", "trid_drain", "global"])
def test_trid_read_race(mesh_device, k, read_bytes, barrier):
    defines = {"trid": [], "trid_drain": ["DRAIN_CMDBUF"], "global": ["GLOBAL_BARRIER"]}[barrier]
    r = _run(mesh_device, k, read_bytes, defines)
    logger.info(f"K {k} READ_BYTES {read_bytes} barrier {barrier}: {r}")
    if barrier != "trid":
        assert r["fails"] == 0, f"{barrier} barrier lost reads: {r}"
    # the plain trid barrier is the one under suspicion: report, do not assert (the point is to see whether / where it
    # fails)
