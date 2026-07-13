// SPDX-License-Identifier: Apache-2.0
// Worker-pull consumer (NCRISC/NOC1): waits for its reader to signal "block b produced" (valid sem, set
// locally by the reader's multicast), then NoC-reads block b straight out of the reader's L1 ring into a
// local scratch, WD reads in flight. After a batch lands, increments the reader's free-credit semaphore so
// the reader can reuse those ring slots. This moves the L1 read-out off the reader onto the workers.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t D = get_compile_time_arg_val(3);
    constexpr uint32_t WD = get_compile_time_arg_val(4);
    constexpr uint32_t cb_shared = get_compile_time_arg_val(5);  // same id/layout as reader's ring
    constexpr uint32_t cb_local = get_compile_time_arg_val(6);
    constexpr uint32_t valid_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t free_sem_id = get_compile_time_arg_val(8);

    const uint32_t reader_x = get_arg_val<uint32_t>(0);
    const uint32_t reader_y = get_arg_val<uint32_t>(1);

    const uint32_t valid_addr = get_semaphore(valid_sem_id);
    volatile tt_l1_ptr uint32_t* valid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);
    const uint32_t free_addr = get_semaphore(free_sem_id);
    const uint64_t reader_free = get_noc_addr(reader_x, reader_y, free_addr);

    const uint32_t reader_ring_base = get_write_ptr(cb_shared);  // == reader's ring base (identical layout)
    const uint32_t local_base = get_write_ptr(cb_local);

    uint32_t b = 0;
    while (b < num_blocks) {
        uint32_t n = (num_blocks - b < WD) ? (num_blocks - b) : WD;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t blk = b + i;
            noc_semaphore_wait_min(valid_ptr, blk + 1);
            uint64_t src = get_noc_addr(reader_x, reader_y, reader_ring_base + (blk % D) * block_bytes);
            noc_async_read(src, local_base + (i % WD) * block_bytes, block_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            noc_semaphore_inc(reader_free, 1);
        }
        b += n;
    }
}
