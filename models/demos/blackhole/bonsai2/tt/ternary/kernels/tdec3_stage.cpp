// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Ternary decode linear, weight staging: copies this core's weight pages (its output tiles' records, as the linear's
// core_span splits them) from DRAM into L1 at `stage`, PARITY 0 / 1 taking the even / odd pages. It runs inside an
// earlier op that leaves DRAM idle (gdn_step.py extra kernels); the linear then streams them locally (W_STAGED).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_span.hpp"

void kernel_main() {
    // common args: w, stage, Nt, ncores, grid_x, unit
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t stage = get_common_arg_val<uint32_t>(1);
    uint32_t first_nt, num_nt;
    core_span(
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3),
        get_common_arg_val<uint32_t>(4),
        get_common_arg_val<uint32_t>(5),
        first_nt,
        num_nt);
    constexpr uint32_t ppr = get_compile_time_arg_val(0);
    constexpr uint32_t PARITY = get_compile_time_arg_val(1);
    constexpr auto w_args = TensorAccessorArgs<3>();
    const auto w = TensorAccessor(w_args, w_addr, 4096);
    const uint32_t page0 = first_nt * ppr, pages = num_nt * ppr;
    // at most INFLIGHT reads in flight per RISC: the host op's own (latency-bound) DRAM reads must not queue behind
    // the whole stream at the banks
    constexpr uint32_t INFLIGHT = get_compile_time_arg_val(2);
    uint32_t trid = 0, issued = 0;
    for (uint32_t q = PARITY; q < pages; q += 2) {
        if (issued >= INFLIGHT) {
            noc_async_read_barrier_with_trid(((trid + INFLIGHT - INFLIGHT) % INFLIGHT) + 1);
        }
        noc_async_read_set_trid(trid + 1);
        noc_async_read(w.get_noc_addr(page0 + q), stage + q * 4096, 4096);
        trid = (trid + 1) % INFLIGHT;
        ++issued;
    }
    noc_async_read_barrier();
    noc_async_read_set_trid(0);
}
