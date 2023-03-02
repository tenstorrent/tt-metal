#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    // Constexpr
    constexpr uint32_t num_dram_channels               = 8;
    constexpr uint32_t log_base_2_of_num_dram_channels = 3;
    constexpr uint32_t cb_id_in0                       = 0;

    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks               = get_arg_val<uint32_t>(1);
    const uint32_t stick_size               = get_arg_val<uint32_t>(2);
    uint32_t l1_buffer_addr                 = get_arg_val<uint32_t>(4);
    uint32_t start_id                       = get_arg_val<uint32_t>(5);

    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    const uint32_t num_tiles_c = stick_size / 64; // Assuming 2 bytes per datum, there are 64 bytes per tile row
    uint32_t stick_id          = 0;

    constexpr bool stick_size_is_power_of_two = (get_compile_time_arg_val(0) == 1);
    #if (stick_size_is_power_of_two)
    const uint32_t log_base_2_of_bank_unit_size = get_arg_val<uint32_t>(3);
    const InterleavedPow2AddrGen s = {
        .bank_base_address = src_addr,
        .num_used_banks = num_dram_channels,
        .log_base_2_of_num_used_banks = log_base_2_of_num_dram_channels,
        .log_base_2_of_bank_unit_size = log_base_2_of_bank_unit_size // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen s = {
        .bank_base_address = src_addr,
        .num_used_banks = num_dram_channels,
        .log_base_2_of_num_used_banks = log_base_2_of_num_dram_channels,
        .bank_unit_size = stick_size
    };
    #endif
    for (uint32_t stick_id = start_id; stick_id < start_id + num_sticks; stick_id++) {
        uint64_t src_noc_addr = get_noc_addr(stick_id, s);

        uint32_t bank_id = stick_id & (num_dram_channels - 1);
        noc_async_read(src_noc_addr, l1_buffer_addr, stick_size);
        noc_async_read_barrier();
        l1_buffer_addr += stick_size;
    }
}
