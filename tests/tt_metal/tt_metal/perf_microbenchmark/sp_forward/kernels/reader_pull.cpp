// SPDX-License-Identifier: Apache-2.0
// Worker-pull reader (BRISC/NOC0): peak contiguous DRAM bank read into a D-deep ring in this core's L1.
// Does NOT send the data. Instead it (a) signals "block b produced" to its K workers via a multicast
// semaphore set (valid), and (b) waits on a local free-credit semaphore that workers increment after they
// have pulled a block, before reusing that ring slot. The L1 read-OUT is done by the workers (NOC1),
// keeping this core's NoC/RISC on the DRAM read only.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t D = get_compile_time_arg_val(5);
    constexpr uint32_t K = get_compile_time_arg_val(6);
    constexpr uint32_t valid_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t free_sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t produced_sem_id = get_compile_time_arg_val(9);

    uint32_t ai = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t bank_id = get_arg_val<uint32_t>(ai++);
    const uint32_t vc = get_arg_val<uint32_t>(ai++);
    const uint32_t wx0 = get_arg_val<uint32_t>(ai++);  // worker mcast rect (NOC0 normal order)
    const uint32_t wy0 = get_arg_val<uint32_t>(ai++);
    const uint32_t wx1 = get_arg_val<uint32_t>(ai++);
    const uint32_t wy1 = get_arg_val<uint32_t>(ai++);

    const uint32_t base_l1 = get_write_ptr(cb_id);
    constexpr uint32_t block_bytes = num_pages * page_bytes;

    const uint32_t free_addr = get_semaphore(free_sem_id);
    volatile tt_l1_ptr uint32_t* free_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(free_addr);
    const uint32_t valid_addr = get_semaphore(valid_sem_id);
    const uint32_t produced_addr = get_semaphore(produced_sem_id);
    volatile tt_l1_ptr uint32_t* produced_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(produced_addr);
    const uint64_t mcast_valid = get_noc_multicast_addr(wx0, wy0, wx1, wy1, valid_addr);

    uint64_t src_base = get_noc_addr_from_bank_id<true>(bank_id, input_addr);
    noc_async_read_one_packet_set_state<true>(src_base, page_bytes, vc);
    uint32_t l1_read_addr = 0;

    constexpr uint32_t total_num_blocks_in_buffer = 3;  // trid depth
    uint32_t num_free = total_num_blocks_in_buffer, curr_trid = 1, trid_wait = 1;

    for (uint32_t b = 0; b < num_blocks; ++b) {
        if (b >= D) {
            noc_semaphore_wait_min(free_ptr, (b - D + 1) * K);
        }
        uint32_t l1w = base_l1 + (b % D) * block_bytes;
        noc_async_read_set_trid(curr_trid);
        for (uint32_t h = 0; h < num_pages; ++h) {
            noc_async_read_one_packet_with_state_with_trid(src_base, l1_read_addr, l1w, curr_trid);
            l1_read_addr += page_bytes;
            l1w += page_bytes;
        }
        if (num_free == 2) {
            noc_async_read_barrier_with_trid(trid_wait);
            trid_wait = trid_wait == 3 ? 1 : (trid_wait + 1);
        } else {
            num_free -= 1;
        }
        // signal block b ready to all K workers
        *produced_ptr = b + 1;
        noc_semaphore_set_multicast(produced_addr, mcast_valid, K);
        curr_trid = curr_trid == 3 ? 1 : (curr_trid + 1);
    }
    noc_async_read_barrier_with_trid(trid_wait);
    *produced_ptr = num_blocks;
    noc_semaphore_set_multicast(produced_addr, mcast_valid, K);
    // wait until all workers have drained everything (avoids closing while workers still read)
    noc_semaphore_wait_min(free_ptr, num_blocks * K);
}
