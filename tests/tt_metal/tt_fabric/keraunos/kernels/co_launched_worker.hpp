// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

// Runs `stock_main`, a stock fabric test kernel's renamed kernel_main, on a hart of a Keraunos CCE whose hart 0 runs
// the fabric router; the worker is launched with the router, before the router is ready for traffic.
//
// Named compile-time arg WORKER_ARGS_INDEX: where this worker's runtime args start, after the stock kernel's. They are
// the router's EDM status address in the worker's view (uncached), the READY_FOR_TRAFFIC value, and the run's nonce.
// Blocks until the status word holds READY_FOR_TRAFFIC, then runs `stock_main`, then writes the nonce to the last
// word of the stock result block (`results`, `results_size_bytes`), which the stock kernels leave unused.
template <typename StockMain>
void run_co_launched_worker(StockMain stock_main, tt_l1_ptr uint32_t* results, uint32_t results_size_bytes) {
    constexpr uint32_t args_index = get_named_compile_time_arg_val("WORKER_ARGS_INDEX");
    const uint32_t router_status_address = get_arg_val<uint32_t>(args_index);
    const uint32_t router_ready = get_arg_val<uint32_t>(args_index + 1);
    const uint32_t nonce = get_arg_val<uint32_t>(args_index + 2);

    volatile tt_l1_ptr uint32_t* router_status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(router_status_address);
    while (*router_status != router_ready) {
        invalidate_l1_cache();
    }
    stock_main();
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results)[results_size_bytes / sizeof(uint32_t) - 1] = nonce;
}
