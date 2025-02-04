// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
// clang-format on

void kernel_main() {
    uint32_t rt_args_idx = 0;
    uint32_t time_seed = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t num_tx_workers = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t tx_sync_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t host_sync_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t num_mcast_dests = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t mcast_encoding = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    // wait for sync from tx kernels
    while (*(volatile tt_l1_ptr uint32_t*)tx_sync_addr != num_tx_workers);

    // wait for signal from host
    // this is needed to know that all the routers are up and running on all the chips
    while (*(volatile tt_l1_ptr uint32_t*)host_sync_addr == 0);

    tt_l1_ptr uint32_t* sync_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(0x100000);
    *sync_addr = 1;

    /*
    // do a noc multicast to tx kernels
    uint64_t mcast_dest_addr = get_noc_addr_helper(mcast_encoding, tx_sync_addr);
    noc_async_write_multicast_one_packet((uint32_t)sync_addr, mcast_dest_addr, sizeof(uint32_t), num_mcast_dests);
    */

    uint64_t ucast_addr = NOC_XY_ADDR(NOC_X(18), NOC_Y(20), tx_sync_addr);
    noc_inline_dw_write(ucast_addr, 1);
}
