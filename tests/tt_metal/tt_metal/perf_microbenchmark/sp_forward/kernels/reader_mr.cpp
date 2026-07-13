// SPDX-License-Identifier: Apache-2.0
// Multi-reader peak contiguous DRAM bank read (one_packet_with_state + triple-buffer TRID). Each reader
// reads a UNIQUE contiguous slice [base_off, base_off + num_blocks*block) of its bank. NOC is chosen by the
// kernel's DataMovementConfig (RISCV_0/NOC0 or RISCV_1/NOC1) at create time. Reader==consumer model:
// no forwarding. Reports kernel-time via profiler.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);   // packets per block
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);  // <=16384
    constexpr uint32_t cb_id = get_compile_time_arg_val(3);

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t vc = get_arg_val<uint32_t>(2);
    const uint32_t base_off = get_arg_val<uint32_t>(3);

    uint64_t src_base = get_noc_addr_from_bank_id<true>(bank_id, input_addr + base_off);
    noc_async_read_one_packet_set_state<true>(src_base, page_bytes, vc);
    const uint32_t l1_base = get_write_ptr(cb_id);
    uint32_t l1_read_addr = 0;

    constexpr uint32_t total_num_blocks_in_buffer = 3;
    uint32_t num_free = total_num_blocks_in_buffer, curr_trid = 1, trid_wait = 1;
    constexpr uint32_t block_bytes = num_pages * page_bytes;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        uint32_t l1w = l1_base + (block % total_num_blocks_in_buffer) * block_bytes;
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
        curr_trid = curr_trid == 3 ? 1 : (curr_trid + 1);
    }
    noc_async_read_barrier_with_trid(trid_wait);
}
