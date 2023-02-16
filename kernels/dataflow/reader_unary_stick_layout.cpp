#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    // Constexpr
    constexpr uint32_t num_dram_channels               = 8;
    constexpr uint32_t log_base_2_of_num_dram_channels = 3;
    constexpr uint32_t cb_id_in0                       = 0;

    uint32_t src_addr                 = get_arg_val<uint32_t>(0); 
    uint32_t num_sticks               = get_arg_val<uint32_t>(1);
    uint32_t stick_size               = get_arg_val<uint32_t>(2);
    uint32_t log_base_2_of_stick_size = get_arg_val<uint32_t>(3);

    // Assuming 2 bytes per datum, there are 64 bytes per tile row
    const uint32_t num_tiles_c = stick_size / 64;
    uint32_t stick_id          = 0;

    for (uint32_t i = 0; i < num_sticks / 32; i++) {
        // We reserve back an entire tile row and issue a bunch of reads 
        cb_reserve_back(cb_id_in0, num_tiles_c);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t j = 0; j < 32; j++) {
            uint64_t src_noc_addr = get_noc_addr(
                stick_id, src_addr, num_dram_channels, 
                log_base_2_of_num_dram_channels, log_base_2_of_stick_size);

            uint32_t bank_id = stick_id & (num_dram_channels - 1);
            noc_async_read(src_noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, num_tiles_c);
    }
}
    

