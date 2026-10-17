# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the K3 routed experts' DRAM read mix without compute (one chip). The flat op streams gate/up weights as
54 KB reads, each reader from its own bank (16 readers, 2 per bank), and the down weights as 9 KB blocks that rotate
over the banks (~38 down cores), at K3 volumes per call (gate/up 694 MB + down 347 MB). This replays those patterns as
plain NoC reads (a few in flight per core) and times variants: each stream alone, the mix, bigger down blocks, down
blocks pinned to one bank per core. Question: does the down pattern cost the mix its DRAM efficiency (87% vs 93%)?"""
import os

import pytest
from loguru import logger

import ttnn

_KEY = "DEVICE KERNEL DURATION [ns]"

_K = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t PAGE = get_compile_time_arg_val(0), DEPTH = get_compile_time_arg_val(1);
constexpr auto a_args = TensorAccessorArgs<2>();
void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0), first = get_arg_val<uint32_t>(1), n = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    const auto a = TensorAccessor(a_args, addr, PAGE);
    const uint32_t l1 = get_write_ptr(0);
    // a continuous queue of DEPTH reads, one transaction id each (as the flat op's readers): read i reuses read
    // i - DEPTH's slot once that one has landed
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t trid = 1 + i % DEPTH;
        if (i >= DEPTH) {
            while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
            }
        }
        noc_async_read_set_trid(trid);
        noc_async_read(a.get_noc_addr(first + i * stride), l1 + (i % DEPTH) * PAGE, PAGE);
    }
    noc_async_read_barrier();
}
"""


def _dur(mesh_device):
    ttnn.ReadDeviceProfiler(mesh_device)
    us = [float(p.program_analyses_results[_KEY].duration) / 1e3 for progs in (ttnn.get_latest_programs_perf_data() or {}).values()
          for p in progs if _KEY in (p.program_analyses_results or {})]
    return us[-1] if us else float("nan")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dram_mix(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    gu_mb, dn_mb = float(os.environ.get("DM_GU_MB", "694")), float(os.environ.get("DM_DN_MB", "347"))
    GU_PAGE, BANKS = 55296, 8
    # the K3 routed plan's own cores (readers next to the DRAM columns, 28 down cores of 4 output tile columns)
    plan = dict(ttnn._ttnn.operations.bringup.flat_routed_expert_plan(
        mesh_device, 3584, 3072, 56, 448, 8192, weights_bf8=False, pin=1, x_bf16=False, h_bf16=False))
    plan_rd = [ttnn.CoreCoord(x, y) for x, y in plan["readers"]]
    plan_dn = [ttnn.CoreCoord(x, y) for x, y in plan["down"]]
    use_plan = os.environ.get("DM_PLAN_CORES", "1") == "1"
    n_gu_rd, n_dn = 16, (len(plan_dn) if use_plan else 38)
    gu_pages = int(gu_mb * 1e6 / GU_PAGE) // n_gu_rd * n_gu_rd
    t_gu = ttnn.allocate_tensor_on_device(ttnn.Shape([gu_pages, GU_PAGE // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device,
                                          ttnn.DRAM_MEMORY_CONFIG)
    dn = {}
    for page in (9216, 36864):
        pages = int(dn_mb * 1e6 / page) // n_dn * n_dn
        dn[page] = (pages, ttnn.allocate_tensor_on_device(ttnn.Shape([pages, page // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT,
                                                         mesh_device, ttnn.DRAM_MEMORY_CONFIG))
    gu_cores, dn_cores = (plan_rd, plan_dn) if use_plan else (cores[:n_gu_rd], cores[n_gu_rd : n_gu_rd + n_dn])

    def group(t, page, depth, core_list, plan):
        rt = ttnn.RuntimeArgs()
        for i, c in enumerate(core_list):
            rt[c.x][c.y] = [int(t.buffer_address())] + plan(i)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_list])
        cb = ttnn.CBDescriptor(total_size=depth * page, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, depth * page)])
        k = ttnn.KernelDescriptor(kernel_source=_K, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=crs,
                                  compile_time_args=[page, depth] + list(ttnn.TensorAccessorArgs(t).get_compile_time_args()),
                                  runtime_args=rt, config=ttnn.DataMovementConfigDescriptor(
                                      processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
        return cb, k, page * sum(plan(i)[1] for i in range(len(core_list)))

    per_rd = gu_pages // n_gu_rd

    def gu_plan(i):  # reader i: bank i % 8, its half of that bank's pages (page b + 8 k), sequential in the bank
        b, half = i % BANKS, i // BANKS
        n = per_rd
        return [b + BANKS * half * n, n, BANKS]

    def dn_rot(page):
        pages = dn[page][0]
        per = pages // n_dn
        return lambda i: [i * per, per, 1]  # consecutive pages: rotate over the banks

    def dn_pin(page):
        pages = dn[page][0]
        k_per = (pages // BANKS) // -(-n_dn // BANKS)  # (at most ceil(n_dn / 8) cores per bank)
        # core i: bank i % 8, its slot i // 8 among that bank's cores: pages b + 8 k, k in [slot k_per, (slot + 1) k_per)
        return lambda i: [(i % BANKS) + BANKS * (i // BANKS) * k_per, k_per, BANKS]

    # cores next to the DRAM columns that the readers leave (logical columns 0, 1, 7, 8)
    near = [ttnn.CoreCoord(x, y) for x in (0, 1, 7, 8) for y in range(grid.y) if (x, y) not in {(c.x, c.y) for c in plan_rd}]
    only = os.environ.get("DM_ONLY")
    variants = {
        "gate/up alone (16 x 54 KB, own bank)": [("gu",)],
        "down alone (9 KB, rotating)": [("dn", 9216, "rot")],
        "mix: gate/up + down 9 KB rotating (as today)": [("gu",), ("dn", 9216, "rot")],
        "mix: gate/up + down 36 KB rotating": [("gu",), ("dn", 36864, "rot")],
        "mix: gate/up + down 9 KB pinned per bank": [("gu",), ("dn", 9216, "pin")],
        "down alone 36 KB rotating": [("dn", 36864, "rot")],
        "down alone 9 KB pinned per bank": [("dn", 9216, "pin")],
        "down alone 9 KB rotating, from 24 cores next to DRAM": [("dn", 9216, "rot", "near")],
        "mix: gate/up + down 9 KB rotating from cores next to DRAM": [("gu",), ("dn", 9216, "rot", "near")],
    }
    if only:
        variants = {k: v for k, v in variants.items() if any(o in k for o in only.split(","))}
    for name, parts in variants.items():
        pd = ttnn.ProgramDescriptor()
        cbs, ks, total = [], [], 0
        for p in parts:
            if p[0] == "gu":
                cb, k, b = group(t_gu, GU_PAGE, int(os.environ.get("DM_GU_DEPTH", "4")), gu_cores, gu_plan)
            else:
                page, mode = p[1], p[2]
                cl = near[:24] if len(p) > 3 else dn_cores
                if len(p) > 3:  # (the same bytes over the 24 near cores: rotating consecutive page ranges)
                    per_n = dn[page][0] // len(cl)
                    plan_f = lambda i, per_n=per_n: [i * per_n, per_n, 1]  # noqa: E731
                else:
                    plan_f = dn_rot(page) if mode == "rot" else dn_pin(page)
                cb, k, b = group(dn[page][1], page, int(os.environ.get("DM_DN_DEPTH", "2")), cl, plan_f)
            cbs.append(cb)
            ks.append(k)
            total += b
        pd.cbs, pd.kernels = cbs, ks
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))] = pd
        best = None
        for _ in range(3):
            _dur(mesh_device)
            ttnn.generic_op([t_gu], md)
            ttnn.synchronize_device(mesh_device)
            t = _dur(mesh_device)
            best = t if best is None or t < best else best
        logger.info(f"DMIX {name:48s} {total / 1e6:7.1f} MB in {best:8.1f} us = {total / best / 1e3:6.1f} GB/s ({total / best / 1e3 / 512 * 100:4.1f}%)")
