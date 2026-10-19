# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmarks for x injectors on the gate/up cores (instead of the x relays), one chip, RT-profiler device time.

A piece is 32 gathered x rows x W tiles of columns (W 8: 512 B of each 8 KB row, the flat expert's K block): the
reader gathers its rows (random token ids in [0, T)) from the row-major bf16 x [T, H] in DRAM into the rm CB, the
compute tilizes it to bfp8 (fast_tilize, as the relay's se11_tz.cpp), the writer sends the W bfp8 tiles to another
core's L1 (the snake head's x ring). Cases:
  tilize   reader without DRAM reads (the CB data is whatever is there), tilize, writer pops: TRISC cycles a tile
  read     reader gathers, no compute, writer pops the rm CB: gather-read GB/s a core (D pieces in flight)
  inject   all three: an injector's GB/s of x delivered
on 1 and 8 cores, the reader on NOC0 or NOC1. GLM_XI_CASES (comma list of case:cores:noc:W:D) overrides the list."""

import os
import statistics
import time

import pytest
import torch

import ttnn

T, H = 5120, 4096
N_PIECES = int(os.environ.get("GLM_XI_PIECES", "2000"))
ITERS = 5

_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t W = get_compile_time_arg_val(0), N = get_compile_time_arg_val(1), D = get_compile_time_arg_val(2);
constexpr uint32_t READ = get_compile_time_arg_val(3), T = get_compile_time_arg_val(4), KB = get_compile_time_arg_val(5);
constexpr auto x_args = TensorAccessorArgs<6>();
constexpr uint32_t PB = W * 64;  // (a piece row: W tiles x 32 bf16 columns)
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0), seed = get_arg_val<uint32_t>(1);
    const auto x = TensorAccessor(x_args, x_addr, 8192);
    uint32_t s = seed * 2654435761u + 1;
    for (uint32_t i = 0; i < N; i += D) {
        const uint32_t d = N - i < D ? N - i : D;
        cb_reserve_back(0, d * W);
        uint32_t l1 = get_write_ptr(0);
        if constexpr (READ) {
            for (uint32_t j = 0; j < d; ++j) {
                const uint32_t col = ((i + j) % KB) * PB;
                for (uint32_t r = 0; r < 32; ++r) {
                    s = s * 1664525u + 1013904223u;
                    noc_async_read(x.get_noc_addr((s >> 8) % T) + col, l1 + j * 32 * PB + r * PB, PB);
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(0, d * W);
    }
}
"""

_TILIZE = r"""
#include <cstdint>
#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_hw_startup.h"
constexpr uint32_t W = get_compile_time_arg_val(0), N = get_compile_time_arg_val(1);
void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    fast_tilize_init(0, W, 16);
    for (uint32_t i = 0; i < N; ++i) {
        cb_wait_front(0, W);
        cb_reserve_back(16, W);
        fast_tilize_block(0, W, 16, 0, 0);
        cb_push_back(16, W);
        cb_pop_front(0, W);
    }
    fast_tilize_uninit(0, 16, W);
}
"""

_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t W = get_compile_time_arg_val(0), N = get_compile_time_arg_val(1), CB = get_compile_time_arg_val(2);
constexpr uint32_t SEND = get_compile_time_arg_val(3), PAGE = get_compile_time_arg_val(4);
void kernel_main() {
    const uint32_t dx = get_arg_val<uint32_t>(0), dy = get_arg_val<uint32_t>(1), dst = get_arg_val<uint32_t>(2);
    const uint64_t d = get_noc_addr(dx, dy, dst);
    for (uint32_t i = 0; i < N; ++i) {
        cb_wait_front(CB, W);
        if constexpr (SEND) {
            noc_async_write(get_read_ptr(CB), d, W * PAGE);
            noc_async_write_barrier();
        }
        cb_pop_front(CB, W);
    }
}
"""


def _program(device, x, case, cores, noc, W, D, sink):
    """One program: the case's kernels on `cores` (logical), the writers sending to `sink` (logical) L1."""
    read = case in ("read", "inject")
    tz = case in ("tilize", "inject")
    send = case == "inject"
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    rm_page, bf8_page = 2048, 1088
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * max(D, 1) * W * rm_page,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, rm_page)],
        )
    ]
    if tz:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * W * bf8_page,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(16, ttnn.bfloat8_b, bf8_page)],
            )
        )
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    rd_noc = ttnn.NOC.RISCV_0_default if noc == 0 else ttnn.NOC.RISCV_1_default
    wr_noc = ttnn.NOC.RISCV_1_default if noc == 0 else ttnn.NOC.RISCV_0_default
    sv = device.worker_core_from_logical_core(sink)
    rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rrt[c.x][c.y] = [int(x.buffer_address()), 17 + i]
        wrt[c.x][c.y] = [sv.x, sv.y, 0x100000]  # (a scratch L1 address on the sink, past the CBs)
    rct = [W, N_PIECES, D, int(read), T, H // (W * 32)] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=st,
            core_ranges=crs,
            compile_time_args=rct,
            runtime_args=rrt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=rd_noc),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=st,
            core_ranges=crs,
            compile_time_args=[W, N_PIECES, 16 if tz else 0, int(send), bf8_page if tz else rm_page],
            runtime_args=wrt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=wr_noc),
        ),
    ]
    if tz:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_TILIZE,
                source_type=st,
                core_ranges=crs,
                compile_time_args=[W, N_PIECES],
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.ComputeConfigDescriptor(),
            )
        )
    p = ttnn.ProgramDescriptor()
    p.cbs = cbs
    p.kernels = kernels
    return p


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_x_injector(device):
    x = ttnn.from_torch(
        torch.randn(T, H).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # a dummy output for generic_op (the programs write no tensor)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    recs = {}

    def cb(batch):
        for r in batch.records:
            recs[r.runtime_id] = (r.end_timestamp - r.start_timestamp) / (r.frequency * 1e3), r.frequency

    rt = ttnn.device.IsProgramRealtimeProfilerActive()
    hdl = ttnn.device.RegisterProgramRealtimeProfilerCallback(cb) if rt else None
    # gate/up-like cores in the middle of the grid; the sink a core beside them
    # GLM_XI_PLACE row: the cores side by side on rows 4, 5; spread: two a row over rows 1-8, different columns
    if os.environ.get("GLM_XI_PLACE", "row") == "spread":
        row = [ttnn.CoreCoord(1 + (3 * y + 4 * k) % 8, y) for y in range(1, 9) for k in range(2)]
        row = row[0::2] + row[1::2]  # (the first 8: one a row)
    else:
        row = [ttnn.CoreCoord(1 + i % 8, 4 + i // 8) for i in range(16)]
    sink = ttnn.CoreCoord(9, 6)
    default = [
        "tilize:1:0:8:1",
        "tilize:1:0:32:1",
        "read:1:1:8:1",
        "read:1:1:8:4",
        "read:1:0:8:4",
        "read:8:1:8:4",
        "read:8:0:8:4",
        "inject:1:1:8:4",
        "inject:8:1:8:4",
        "inject:8:0:8:4",
    ]
    cases = os.environ.get("GLM_XI_CASES", ",".join(default)).split(",")
    try:
        for spec in cases:
            case, n, noc, W, D = spec.split(":")
            n, noc, W, D = int(n), int(noc), int(W), int(D)
            desc = ttnn.MeshProgramDescriptor()
            desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))] = _program(
                device, x, case, row[:n], noc, W, D, sink
            )
            ttnn.generic_op([x, out], desc)  # compile
            ttnn.synchronize_device(device)
            ids = []
            for _ in range(ITERS):
                i0 = ttnn._ttnn.get_device_operation_id()
                ttnn.generic_op([x, out], desc)
                ids.append((i0, ttnn._ttnn.get_device_operation_id()))
            ttnn.synchronize_device(device)
            time.sleep(0.5)
            per = [sum(recs[i][0] for i in range(a, b) if i in recs) for a, b in ids]
            us = statistics.median(per) if per and all(per) else float("nan")
            ghz = next((recs[i][1] for a, b in ids for i in range(a, b) if i in recs), 1.35)
            tiles = N_PIECES * W
            rm_bytes = N_PIECES * 32 * W * 64  # (bf16 x delivered, per core)
            print(
                f"[xi] {case:6s} cores {n} noc {noc} W {W:2d} D {D}: {us:8.1f} us, per core {rm_bytes / (us * 1e3):6.2f} GB/s "
                f"bf16 x ({n * rm_bytes / (us * 1e3):6.1f} all), {us * 1e3 * ghz / tiles:6.1f} cycles a tile @ {ghz:.3f} GHz",
                flush=True,
            )
    finally:
        if hdl is not None:
            ttnn.device.UnregisterProgramRealtimeProfilerCallback(hdl)
