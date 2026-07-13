// SPDX-License-Identifier: Apache-2.0
// in0 loader MCAST stage (NCRISC, chosen NoC): consume in0 K-blocks from cb1 (filled by loader_read on the
// other RISC) and multicast each to the compute band. Overlaps the read (BRISC) so the loader finishes
// fast and contends with in1 only briefly. md mcasts in flight.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_kblocks = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_kb = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb1 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(4);
    constexpr uint32_t md = get_compile_time_arg_val(5);

    const uint32_t ndest = get_arg_val<uint32_t>(0);
    const uint32_t x0 = get_arg_val<uint32_t>(1);
    const uint32_t y0 = get_arg_val<uint32_t>(2);
    const uint32_t x1 = get_arg_val<uint32_t>(3);
    const uint32_t y1 = get_arg_val<uint32_t>(4);

    const uint32_t dst = get_write_ptr(cb_dst);
    const uint64_t maddr = get_noc_multicast_addr(x0, y0, x1, y1, dst);
    const uint32_t blk_bytes = tiles_per_kb * tile_bytes;

    uint32_t inflight = 0;
    for (uint32_t kb = 0; kb < num_kblocks; ++kb) {
        cb_wait_front(cb1, tiles_per_kb);
        uint32_t src = get_read_ptr(cb1);
        noc_async_write_multicast(src, maddr, blk_bytes, ndest);
        inflight++;
        if (inflight >= md) {
            noc_async_write_barrier();
            cb_pop_front(cb1, inflight * tiles_per_kb);
            inflight = 0;
        }
    }
    if (inflight > 0) {
        noc_async_write_barrier();
        cb_pop_front(cb1, inflight * tiles_per_kb);
    }
}
