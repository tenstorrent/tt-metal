#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    // Constexpr
    constexpr uint32_t cb_id_out0                       = 16;

    const uint32_t dst_addr                   = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks                 = get_arg_val<uint32_t>(1);
    const uint32_t stick_size                 = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles_per_block        = get_arg_val<uint32_t>(3);
    const uint32_t block_width_size           = get_arg_val<uint32_t>(4);
    const uint32_t num_full_blocks_in_row     = get_arg_val<uint32_t>(5);
    // const uint32_t num_leftover_tiles_in_row  = get_arg_val<uint32_t>(6);
    // const uint32_t leftover_width_in_row      = get_arg_val<uint32_t>(7);


    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    uint32_t stick_id          = 0;

    #define stick_size_is_power_of_two get_compile_time_arg_val(0) == 1
    #if (stick_size_is_power_of_two)
    const uint32_t log_base_2_of_page_size = get_arg_val<uint32_t>(8);
    const InterleavedPow2AddrGen<true> s = {
        .bank_base_address = dst_addr,


        .log_base_2_of_page_size = log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen<true> s = {
        .bank_base_address = dst_addr,


        .page_size = stick_size
    };
    #endif

    uint64_t base_dst_noc_addr[32];

    auto write_tiles = [&] (const uint32_t& num_tiles, const uint32_t& width_size) {
        cb_wait_front(cb_id_out0, num_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        // for (uint32_t i = 0; i < num_tiles; i++) {
        //     for (uint32_t k = 0; k < 32; k++) {
        //         uint64_t dst_noc_addr = base_dst_noc_addr[k];

        //         noc_async_write(l1_read_addr, dst_noc_addr, 64);
        //         l1_read_addr += 64;
        //         base_dst_noc_addr[k] += 64;
        //     }
        // }
        for (uint32_t k = 0; k < 32; k++) {
            uint64_t dst_noc_addr = base_dst_noc_addr[k];

            noc_async_write(l1_read_addr, dst_noc_addr, width_size);
            l1_read_addr += width_size;
            base_dst_noc_addr[k] += width_size;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_tiles);
    };


    for (uint32_t i = 0; i < num_sticks / 32; i++) {
        // Get Base Addresses
        for (uint32_t j = 0; j < 32; j++) {
            base_dst_noc_addr[j] = get_noc_addr(stick_id, s);
            stick_id++;
        }

        for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
            write_tiles(num_tiles_per_block, block_width_size);
        }

        // if (num_leftover_tiles_in_row > 0) {
        //     write_tiles(num_leftover_tiles_in_row, leftover_width_in_row);
        // }
    }
}
