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


def _run(device, k, read_bytes, defines, hammer_cores=0):
    words = read_bytes // 4
    host_in = (torch.arange(NPAGES, dtype=torch.int64).unsqueeze(1) + 1).expand(NPAGES, words).contiguous()
    tt_in = ttnn.from_torch(
        host_in.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = ttnn.from_torch(
        torch.zeros(1, 16, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cb = ttnn.CBDescriptor(
        total_size=k * read_bytes,
        core_ranges=core,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=read_bytes)],
    )
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [tt_in.buffer_address(), tt_out.buffer_address()]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=[0, k, read_bytes, NPAGES, ITERS],
        defines=[(d, "1") for d in defines],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    kernels, cbs = [kernel], [cb]
    if hammer_cores:
        # load: cores (1..hammer_cores, 0) .. read the same DRAM pages back to back for the whole run
        hcores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(hammer_cores, 0))])
        cbs.append(
            ttnn.CBDescriptor(
                total_size=8 * max(read_bytes, 64),
                core_ranges=hcores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=read_bytes)
                ],
            )
        )
        hrt = ttnn.RuntimeArgs()
        for x in range(1, hammer_cores + 1):
            hrt[x][0] = [tt_in.buffer_address(), tt_out.buffer_address(), 7 * x]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=hcores,
                compile_time_args=[0, 8, read_bytes, NPAGES, ITERS],
                defines=[("HAMMER", "1"), ("HAMMER_SCALE", "2")],
                runtime_args=hrt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        )
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([tt_in, tt_out], program)
    ttnn.synchronize_device(device)
    r = [int(v) & 0xFFFFFFFF for v in ttnn.to_torch(tt_out).flatten().tolist()]
    return dict(fails=r[0], iters=r[1], first_iter=r[2], first_j=r[3], fails_last=r[4], fails_other=r[5])


@pytest.mark.parametrize("read_bytes", [64, 2048, 14336, 32768])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("barrier", ["trid", "trid_drain", "global"])
def test_trid_read_race(device, k, read_bytes, barrier):
    defines = {"trid": [], "trid_drain": ["DRAIN_CMDBUF"], "global": ["GLOBAL_BARRIER"]}[barrier]
    r = _run(device, k, read_bytes, defines)
    logger.info(f"TRIDRACE K {k} READ_BYTES {read_bytes} barrier {barrier}: {r}")
    assert r["iters"] == ITERS, f"kernel did not report: {r}"
    if barrier != "trid":
        assert r["fails"] == 0, f"{barrier} barrier lost reads: {r}"
    # the plain trid barrier is the one under suspicion: report, do not assert


@pytest.mark.parametrize("read_bytes", [2048, 14336])
@pytest.mark.parametrize("k", [1, 4, 8])
@pytest.mark.parametrize("barrier", ["trid", "trid_drain", "global"])
def test_trid_read_race_loaded(device, k, read_bytes, barrier):
    """The same check with 10 other cores of row 0 reading the same DRAM pages back to back (NoC 0 back-pressure)."""
    defines = {"trid": [], "trid_drain": ["DRAIN_CMDBUF"], "global": ["GLOBAL_BARRIER"]}[barrier]
    r = _run(device, k, read_bytes, defines, hammer_cores=10)
    logger.info(f"TRIDRACE loaded K {k} READ_BYTES {read_bytes} barrier {barrier}: {r}")
    assert r["iters"] == ITERS, f"kernel did not report: {r}"
    if barrier != "trid":
        assert r["fails"] == 0, f"{barrier} barrier lost reads: {r}"


@pytest.mark.parametrize("hammer", [0, 10])
@pytest.mark.parametrize("k", [16, 32, 64])
@pytest.mark.parametrize("barrier", ["trid", "trid_drain", "global"])
def test_trid_read_race_meta(device, k, hammer, barrier):
    """combine's metadata prefetch shape: up to 64 back-to-back 64 B reads, then one barrier."""
    defines = {"trid": [], "trid_drain": ["DRAIN_CMDBUF"], "global": ["GLOBAL_BARRIER"]}[barrier]
    r = _run(device, k, 64, defines, hammer_cores=hammer)
    logger.info(f"TRIDRACE meta K {k} hammer {hammer} barrier {barrier}: {r}")
    assert r["iters"] == ITERS, f"kernel did not report: {r}"
    if barrier != "trid":
        assert r["fails"] == 0, f"{barrier} barrier lost reads: {r}"
