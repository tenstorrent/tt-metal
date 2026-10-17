# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: NoC multicast bandwidth on the 12 x 10 grid - the top core of every column multicasts TOTAL bytes in CHUNK
pieces down its column (loopback), no handshakes (the destination region is just overwritten); vs a unicast to each
column member in turn; and the row variant. Device time per variant (1 x 1 mesh)."""
import os

import pytest
from loguru import logger

import ttnn

_KEY = "DEVICE KERNEL DURATION [ns]"

_K = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t TOTAL = get_compile_time_arg_val(0), CHUNK = get_compile_time_arg_val(1), MODE = get_compile_time_arg_val(2);
constexpr uint32_t NDEST = get_compile_time_arg_val(3), NBUF = get_compile_time_arg_val(4);
void kernel_main() {
    const uint32_t sender = get_arg_val<uint32_t>(0);
    if (!sender) return;
    const uint32_t x0 = get_arg_val<uint32_t>(1), y0 = get_arg_val<uint32_t>(2), x1 = get_arg_val<uint32_t>(3), y1 = get_arg_val<uint32_t>(4);
    const uint32_t buf = get_write_ptr(0);
    for (uint32_t off = 0, i = 0; off < TOTAL; off += CHUNK, ++i) {
        const uint32_t dst = buf + (i % NBUF) * CHUNK;
        if constexpr (MODE == 0) {  // multicast, loopback
            noc_async_write_multicast_loopback_src(buf, get_noc_multicast_addr(x0, y0, x1, y1, dst), CHUNK, NDEST);
        } else if constexpr (MODE == 1) {  // multicast, no loopback
            noc_async_write_multicast(buf, get_noc_multicast_addr(x0, y0, x1, y1, dst), CHUNK, NDEST - 1);
        } else {  // unicast to each destination (column: y0 .. y1 at x0; row: x0 .. x1 at y0)
            for (uint32_t d = 0; d < NDEST; ++d) {
                const uint32_t x = (MODE == 2) ? x0 : get_arg_val<uint32_t>(5 + d), y = (MODE == 2) ? get_arg_val<uint32_t>(5 + d) : y0;
                noc_async_write(buf, get_noc_addr(x, y, dst), CHUNK);
            }
        }
        if ((i & 3) == 3) noc_async_writes_flushed();
    }
    noc_async_write_barrier();
}
"""


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mcast_bw(mesh_device):
    g = mesh_device.compute_with_storage_grid_size()
    vv = {(x, y): mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x in range(g.x) for y in range(g.y)}
    allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    total = int(os.environ.get("MB_TOTAL", str(2400 * 1024)))
    dummy = ttnn.allocate_tensor_on_device(ttnn.Shape([32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
    for axis in ("col", "row"):
        for mode, mname in ((0, "mcast loopback"), (1, "mcast"), (2 if axis == "col" else 3, "unicast each")):
            for chunk in (16384, 65536):
                for nsend in ("one", "all"):
                    pd = ttnn.ProgramDescriptor()
                    pd.cbs = [ttnn.CBDescriptor(total_size=2 * chunk, core_ranges=allc,
                                                format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, 2 * chunk)])]
                    rt = ttnn.RuntimeArgs()
                    ndest = g.y if axis == "col" else g.x
                    for x in range(g.x):
                        for y in range(g.y):
                            if axis == "col":
                                send = y == 0 and (nsend == "all" or x == 0)
                                args = [int(send), vv[(x, 0)].x, vv[(x, 0)].y, vv[(x, g.y - 1)].x, vv[(x, g.y - 1)].y] + [vv[(x, i)].y for i in range(g.y)]
                            else:
                                send = x == 0 and (nsend == "all" or y == 0)
                                args = [int(send), vv[(0, y)].x, vv[(0, y)].y, vv[(g.x - 1, y)].x, vv[(g.x - 1, y)].y] + [vv[(i, y)].x for i in range(g.x)]
                            rt[x][y] = args
                    pd.kernels = [ttnn.KernelDescriptor(
                        kernel_source=_K, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=allc,
                        compile_time_args=[total, chunk, mode, ndest, 2], runtime_args=rt,
                        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))]
                    md = ttnn.MeshProgramDescriptor()
                    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))] = pd
                    best = None
                    for _ in range(3):
                        ttnn.ReadDeviceProfiler(mesh_device)
                        ttnn.get_latest_programs_perf_data()
                        ttnn.generic_op([dummy], md)
                        ttnn.synchronize_device(mesh_device)
                        ttnn.ReadDeviceProfiler(mesh_device)
                        us = [float(p.program_analyses_results[_KEY].duration) / 1e3 for progs in (ttnn.get_latest_programs_perf_data() or {}).values()
                              for p in progs if _KEY in (p.program_analyses_results or {})]
                        t = us[-1] if us else float("nan")
                        best = t if best is None or t < best else best
                    logger.info(f"MCBW {axis} {mname:15s} chunk {chunk // 1024:3d} KB senders {nsend:3s}: {best:8.1f} us = "
                                f"{total / best / 1e3:6.1f} GB/s per sender")
