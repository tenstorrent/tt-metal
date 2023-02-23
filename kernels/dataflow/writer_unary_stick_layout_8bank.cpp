#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    // Constexpr
    constexpr uint32_t num_dram_channels               = 8;
    constexpr uint32_t log_base_2_of_num_dram_channels = 3;
    constexpr uint32_t cb_id_out0                      = 16;

    uint32_t dst_addr                 = get_arg_val<uint32_t>(0);
    uint32_t num_sticks               = get_arg_val<uint32_t>(1);
    uint32_t stick_size               = get_arg_val<uint32_t>(2);
    uint32_t log_base_2_of_stick_size = get_arg_val<uint32_t>(3);

    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    const uint32_t num_tiles_c = stick_size / 64; // Assuming 2 bytes per datum, there are 64 bytes per tile row
    uint32_t stick_id          = 0;

    for (uint32_t i = 0; i < num_sticks / 32; i++) {
        // We reserve back an entire tile row and issue a bunch of reads
        cb_wait_front(cb_id_out0, num_tiles_c);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        for (uint32_t j = 0; j < 32; j++) {
            uint64_t dst_noc_addr = get_noc_addr(
                stick_id, dst_addr, num_dram_channels,
                log_base_2_of_num_dram_channels, log_base_2_of_stick_size);

            uint32_t bank_id = stick_id & (num_dram_channels - 1);
            noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
            l1_read_addr += stick_size;
            stick_id++;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_tiles_c);
    }
}
