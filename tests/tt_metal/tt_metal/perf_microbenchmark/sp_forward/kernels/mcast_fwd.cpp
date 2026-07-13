// SPDX-License-Identifier: Apache-2.0
// EXP2 mcast consumer (NCRISC/NOC1): consume blocks from cb0 (filled by the BRISC reader) and multicast
// each block to K worker cores (contiguous valid worker rectangle) on NOC1. `md` mcasts in flight before
// a barrier. Workers are receive-only (no consume) -> dst is a fixed cb1 slot, overwritten each block.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_src = get_compile_time_arg_val(3);  // cb0, from reader
    constexpr uint32_t cb_dst = get_compile_time_arg_val(4);  // cb1, on workers (fixed dst)
    constexpr uint32_t md = get_compile_time_arg_val(5);      // mcast pipeline depth
    constexpr uint32_t use_cb = get_compile_time_arg_val(6);  // 0 = decoupled (mcast from fixed src, no wait)

    const uint32_t num_dests = get_arg_val<uint32_t>(0);
    const uint32_t x0 = get_arg_val<uint32_t>(1);
    const uint32_t y0 = get_arg_val<uint32_t>(2);
    const uint32_t x1 = get_arg_val<uint32_t>(3);
    const uint32_t y1 = get_arg_val<uint32_t>(4);

    const uint32_t dst = get_write_ptr(cb_dst);
    const uint64_t maddr = get_noc_multicast_addr(x0, y0, x1, y1, dst);
    const uint32_t fixed_src = get_write_ptr(cb_src);

    uint32_t inflight = 0;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        uint32_t l1_src;
        if constexpr (use_cb) {
            cb_wait_front(cb_src, block_tiles);
            l1_src = get_read_ptr(cb_src);
        } else {
            l1_src = fixed_src;  // decoupled: mcast a static buffer, no producer dependence
        }
        noc_async_write_multicast(l1_src, maddr, block_bytes, num_dests);
        inflight++;
        if (inflight >= md) {
            noc_async_write_barrier();
            if constexpr (use_cb) {
                cb_pop_front(cb_src, inflight * block_tiles);
            }
            inflight = 0;
        }
    }
    if (inflight > 0) {
        noc_async_write_barrier();
        if constexpr (use_cb) {
            cb_pop_front(cb_src, inflight * block_tiles);
        }
    }
}
