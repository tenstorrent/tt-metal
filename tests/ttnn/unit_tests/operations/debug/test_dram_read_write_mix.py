# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blackhole: large DRAM reads beside non-posted DRAM writes on the same NoC.

Each of up to 110 worker cores runs kernels/dram_read_write_mix/dram_read_write_mix.cpp. It reads 16 KiB DRAM pages
on NoC 1, one request per page, and writes 1088 B pieces to DRAM on NoC 1, waiting for the write acks after each
task. The program is run back to back for TEST_SECONDS.

Without the Blackhole split of DRAM reads into 2 KiB requests in noc_async_read, this traffic hung the chip in all nine
runs we recorded on Blackhole P300 chips, after 35 to 526 program runs (0.18 to 0.25 s where timed). With the split,
runs of up to 10 minutes (3.5 million program runs) never hung.

A NoC stall cannot be recovered from Python: the host blocks waiting for the program to finish. The waits this test
observes are capped: after each program run the host waits in ttnn.synchronize_device, which releases the GIL, and a
watchdog thread ends the process with exit status 1 when no run finishes for STALL_SECONDS. That does not cap every
host wait, so run the test under an outer process timeout, which bounds execution:

    timeout --signal=TERM --kill-after=10s 600s pytest tests/ttnn/unit_tests/operations/debug/test_dram_read_write_mix.py

After a stall the chip is hung and must be reset (tt-smi -r) before it is used again.
"""

import os
import sys
import threading
import time

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole

KERNEL = "tests/ttnn/unit_tests/operations/debug/kernels/dram_read_write_mix/dram_read_write_mix.cpp"

TEST_SECONDS = float(os.getenv("TT_DRAM_READ_WRITE_MIX_SECONDS", "20"))
FIRST_RUN_SECONDS = 600  # includes compiling the kernel
STALL_SECONDS = 10  # a program run takes well under a millisecond

COLS, ROWS = 11, 10
TABLE_ROWS, ROW_WORDS = 32, 4096  # one 16 KiB DRAM page per table row
TASKS = 8192
PIECE_WORDS = 272  # 1088 B
SEED = 52270
NOC = 1

pytestmark = [
    pytest.mark.skipif(not is_blackhole(), reason="Blackhole-specific NoC stall"),
    pytest.mark.skipif(os.getenv("TT_METAL_SIMULATOR") is not None, reason="NoC stall is a hardware property"),
]


class Watchdog:
    """Caps the observed waits: ends the process if no program run finishes in time, since a stalled NoC never returns
    control to Python. It runs only while the main thread has released the GIL, as ttnn.synchronize_device does; an
    outer process timeout bounds everything else."""

    def __init__(self):
        self.runs = 0
        self.last = time.monotonic()
        self.limit = FIRST_RUN_SECONDS
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def progress(self):
        self.runs += 1
        self.last = time.monotonic()
        self.limit = STALL_SECONDS

    def watch(self):
        while not self.stop.wait(1.0):
            idle = time.monotonic() - self.last
            if idle > self.limit:
                sys.stderr.write(
                    f"\nSTALL: no program run finished for {idle:.0f} s after {self.runs} runs. The NoC is hung; "
                    "reset the chip (tt-smi -r) before using it again.\n"
                )
                sys.stderr.flush()
                os._exit(1)


def split_tasks(worker, workers):
    count, extra = divmod(TASKS, workers)
    return worker * count + min(worker, extra), count + int(worker < extra)


def payload_row():
    return torch.tensor([(SEED * 65537 + i * 37 + 11) & 0x7FFFFFFF for i in range(PIECE_WORDS)], dtype=torch.int32)


@pytest.mark.parametrize("read_bytes", [16384])
def test_dram_read_write_mix(device, read_bytes):
    grid = device.compute_with_storage_grid_size()
    if grid.x < COLS or grid.y < ROWS:
        pytest.skip(f"needs an {COLS}x{ROWS} worker grid, grid is {grid.x}x{grid.y}")
    cores = [ttnn.CoreCoord(w % COLS, w // COLS) for w in range(COLS * ROWS)]
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(COLS - 1, ROWS - 1))])

    # Row r of the table selects cache pages through its first 32 words; even rows pick pages 0..1023 and odd rows
    # pick pages 131072..132095, so exactly those two spans of the cache are written.
    table_host = (torch.arange(TABLE_ROWS * ROW_WORDS, dtype=torch.int32) % TASKS).reshape(TABLE_ROWS, ROW_WORDS)
    table = ttnn.from_torch(
        table_host,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cache = ttnn.empty(
        (TASKS * 32, PIECE_WORDS),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    runtime_args = ttnn.RuntimeArgs()
    for worker, core in enumerate(cores):
        first, count = split_tasks(worker, len(cores))
        runtime_args[core.x][core.y] = [table.buffer_address(), cache.buffer_address(), first, count, SEED]
    compile_time_args = (
        [NOC, read_bytes]
        + ttnn.TensorAccessorArgs(table).get_compile_time_args()
        + ttnn.TensorAccessorArgs(cache).get_compile_time_args()
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
    )
    cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=core_range,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=size)],
        )
        for index, size in ((0, ROW_WORDS * 4), (1, 4 * PIECE_WORDS * 4))
    ]
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)

    watchdog = Watchdog()
    try:
        ttnn.generic_op([table, cache], program)  # the first run compiles the kernel
        ttnn.synchronize_device(device)
        watchdog.progress()
        start = time.monotonic()
        while time.monotonic() - start < TEST_SECONDS:
            ttnn.generic_op([table, cache], program)
            ttnn.synchronize_device(device)
            watchdog.progress()
        seconds = time.monotonic() - start
    finally:
        watchdog.stop.set()
    print(f"{watchdog.runs} program runs, {seconds:.1f} s of traffic, no stall")

    expected = payload_row()
    for first_page in (0, 4096 * 32):
        written = ttnn.to_torch(ttnn.slice(cache, [first_page, 0], [first_page + 1024, PIECE_WORDS])).to(torch.int32)
        assert torch.equal(written, expected.expand_as(written)), f"cache pages {first_page}.. differ from the payload"
