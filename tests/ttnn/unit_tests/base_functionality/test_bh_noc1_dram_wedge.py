# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Blackhole NOC1 wedge repro: small DRAM writes from many cores plus large DRAM reads, both on NOC1.

No ops and no fabric. Two generic kernels run on every chip of the mesh (8 independent trials), replayed as a trace:

  W  writer cores: all but the last column (110 on a 12x10 grid). Each launch: NOC1_W_ITERS bursts of 32 writes of
     NOC1_W_XFER bytes, the bank advancing by one per write, then a write barrier. Every core starts at bank 0
     (NOC1_LOCKSTEP=1) or core i at bank i % 8 (NOC1_LOCKSTEP=0); both hang.
  R  NOC1_R_CORES reader cores (last column). Each launch: NOC1_R_ITERS windows of NOC1_R_WINDOW reads of NOC1_R_XFER
     bytes, then a read barrier. Readers 0,1 read the even banks, 2,3 the odd banks.
  W writes the low 2 MiB of each bank, R reads the high 2 MiB.

On a LoudBox (8x p150b) this hangs within minutes: the hung chip's NOC1 DRAM endpoints hold up to 31 writes they
received but never acked, while NOC0 still reaches every DRAM channel. Each of these was clean for 600 s: no reads
(NOC1_R_ITERS=0), reads on NOC0 (NOC1_R_NOC=0), and 1088 B reads with the same bytes in flight
(NOC1_R_XFER=1088 NOC1_R_WINDOW=52).

Manual repro, skipped unless NOC1_SECS is set:
  NOC1_SECS=600 TT_METAL_OPERATION_TIMEOUT_SECONDS=5 \
      pytest tests/ttnn/unit_tests/base_functionality/test_bh_noc1_dram_wedge.py
"""

import os
import time

import pytest
from loguru import logger

import ttnn

BANK_BYTES = 4 * 1024 * 1024  # bytes of the test buffer in each DRAM bank; W uses the low half, R the high half

KERNEL = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    const uint32_t span = get_arg_val<uint32_t>(1);        // bytes usable per bank, from base
    const uint32_t xfer = get_arg_val<uint32_t>(2);
    const uint32_t n_iter = get_arg_val<uint32_t>(3);
    const uint32_t burst = get_arg_val<uint32_t>(4);       // transactions per barrier
    const uint32_t first_bank = get_arg_val<uint32_t>(5);
    const uint32_t bank_stride = get_arg_val<uint32_t>(6);
    const uint32_t start_off = get_arg_val<uint32_t>(7);
    const uint32_t l1 = get_write_ptr(0);
    const uint32_t step = (xfer + 63u) & ~63u;
    uint32_t bank = first_bank;
    uint32_t off = start_off;
    for (uint32_t i = 0; i < n_iter; ++i) {
        for (uint32_t j = 0; j < burst; ++j) {
            const uint64_t addr = get_noc_addr_from_bank_id<true>(bank, base + off, noc_index);
#if IS_WRITE
            noc_async_write(l1, addr, xfer);
#else
            noc_async_read(addr, l1 + j * step, xfer);
#endif
            bank += bank_stride;
            if (bank >= NUM_DRAM_BANKS) {
                bank -= NUM_DRAM_BANKS;
            }
            if (bank == first_bank) {  // visited every bank once: next offset
                off += step;
                if (off + xfer > span) {
                    off = 0;
                }
            }
        }
#if IS_WRITE
        noc_async_write_barrier();
#else
        noc_async_read_barrier();
#endif
    }
}
"""


def _env(name, default):
    return int(os.environ.get(name, default))


def _cores(coords):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords])


def _kernel(cores, runtime_args, is_write, noc):
    config = ttnn.DataMovementConfigDescriptor()
    config.processor = ttnn.DataMovementProcessor.RISCV_0
    config.noc = ttnn.NOC.NOC_1 if noc == 1 else ttnn.NOC.NOC_0
    return ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_cores(cores),
        defines=[("IS_WRITE", str(is_write))],
        runtime_args=runtime_args,
        config=config,
    )


def _cb(cores, size):
    return ttnn.CBDescriptor(
        total_size=size,
        core_ranges=_cores(cores),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)],
    )


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1024 * 1024}], indirect=True)
@pytest.mark.timeout(0)
def test_bh_noc1_dram_wedge(mesh_device):
    if "NOC1_SECS" not in os.environ:
        pytest.skip("manual hang repro: set NOC1_SECS to run")
    secs = _env("NOC1_SECS", 900)
    lockstep = _env("NOC1_LOCKSTEP", 1) == 1
    w_xfer, w_iters, w_noc = _env("NOC1_W_XFER", 320), _env("NOC1_W_ITERS", 36), _env("NOC1_W_NOC", 1)
    r_xfer, r_window, r_iters = _env("NOC1_R_XFER", 14144), _env("NOC1_R_WINDOW", 4), _env("NOC1_R_ITERS", 80)
    r_cores, r_noc = _env("NOC1_R_CORES", 4), _env("NOC1_R_NOC", 1)

    banks = mesh_device.dram_grid_size().x
    grid = mesh_device.compute_with_storage_grid_size()
    buf = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, banks * BANK_BYTES // 16384, 4096]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    base = buf.buffer_address()

    w_xy = [(x, y) for x in range(grid.x - 1) for y in range(grid.y)]
    r_xy = [(grid.x - 1, y) for y in range(r_cores)]
    w_args, r_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(w_xy):
        first_bank = 0 if lockstep else i % banks
        w_args[x][y] = [base, BANK_BYTES // 2, w_xfer, w_iters, 32, first_bank, 1, (i * 4096) % (BANK_BYTES // 4)]
    for i, (x, y) in enumerate(r_xy):
        # readers 0,1 read the even banks, 2,3 the odd banks
        r_args[x][y] = [base + BANK_BYTES // 2, BANK_BYTES // 2, r_xfer, r_iters, r_window, (i // 2) % 2, 2, 0]
    r_cb_bytes = r_window * ((r_xfer + 63) // 64 * 64)

    program = ttnn.ProgramDescriptor(
        kernels=[_kernel(w_xy, w_args, 1, w_noc), _kernel(r_xy, r_args, 0, r_noc)],
        cbs=[_cb(w_xy, 16384), _cb(r_xy, r_cb_bytes)],
    )
    logger.info(
        f"{len(w_xy)} writers: {w_iters} x 32 x {w_xfer} B on NOC{w_noc}, lockstep={lockstep}; "
        f"{len(r_xy)} readers: {r_iters} x {r_window} x {r_xfer} B on NOC{r_noc}; {secs} s"
    )

    ttnn.generic_op([buf, buf], program)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    ttnn.generic_op([buf, buf], program)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    t0 = last = time.monotonic()
    launches = 0
    try:
        while time.monotonic() - t0 < secs:
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            launches += 1
            if time.monotonic() - last >= 30:
                last = time.monotonic()
                logger.info(f"{last - t0:.0f} s: {launches} launches")
        logger.info(f"done: {launches} launches in {time.monotonic() - t0:.0f} s, no hang")
    finally:
        ttnn.release_trace(mesh_device, tid)
