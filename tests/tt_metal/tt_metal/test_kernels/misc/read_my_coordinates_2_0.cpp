// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Gen2 variant of read_my_coordinates.cpp: same kernel, named compile-time args.
// Writes my_x, my_y, my_logical_x, my_logical_y, my_sub_device_x and my_sub_device_y
// to results_addr, in that order. Requires 24B.
//

#include "experimental/kernel_args.h"

#ifdef COMPILE_FOR_TRISC
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif

void kernel_main() {
    constexpr uint32_t results_addr = get_arg(args::results_addr);
    volatile tt_l1_ptr uint32_t* results = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results_addr);
#ifndef COMPILE_FOR_TRISC
    results[0] = my_x[noc_index];
    results[1] = my_y[noc_index];
#endif
    results[2] = get_absolute_logical_x();
    results[3] = get_absolute_logical_y();
    results[4] = get_relative_logical_x();
    results[5] = get_relative_logical_y();
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    // These stores land in L2; the host reads node memory, so flush them through.
    flush_l2_cache_range(results_addr, 6 * sizeof(uint32_t));
#endif
}
