#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    // Constexpr
    constexpr uint32_t num_dram_channels               = 8;
    constexpr uint32_t log_base_2_of_num_dram_channels = 3;
    constexpr uint32_t cb_id_in0                       = 0;

    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t num_2d_faces             = get_arg_val<uint32_t>(1);
    const uint32_t num_rows                     = get_arg_val<uint32_t>(2);
    const uint32_t num_rows_padded              = get_arg_val<uint32_t>(3);
    const uint32_t row_size                 = get_arg_val<uint32_t>(4);
    const uint32_t zero_buffer_l1_addr      = get_arg_val<uint32_t>(5);

    volatile tt_l1_ptr std::uint32_t* zero_buffer = (volatile tt_l1_ptr uint32_t*)(zero_buffer_l1_addr);

    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    const uint32_t num_tiles_c = row_size / 64; // Assuming 2 bytes per datum, there are 64 bytes per tile row

    constexpr bool stick_size_is_power_of_two = (get_compile_time_arg_val(0) == 1);
    #if (stick_size_is_power_of_two)
    const uint32_t log_base_2_of_page_size = get_arg_val<uint32_t>(3);
    const InterleavedPow2AddrGen<true> s = {
        .bank_base_address = src_addr,


        .log_base_2_of_page_size = log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen<true> s = {
        .bank_base_address = src_addr,


        .page_size = row_size
    };
    #endif
    uint32_t row_id = 0;
    for(uint32_t i = 0; i < num_2d_faces; i++) {
        uint32_t row_id_in_face = 0;
        for (uint32_t j = 0; j < num_rows_padded / 32; j++) {
            // We reserve back an entire tile row and issue a bunch of reads
            cb_reserve_back(cb_id_in0, num_tiles_c);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            for (uint32_t j = 0; j < 32; j++) {
                if (row_id_in_face >= num_rows) {
                    // pad the tile by reading values from zero buffer in L1
                    volatile tt_l1_ptr std::uint32_t* dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr);
                    for(uint32_t z = 0; z < row_size / 4; z++) {
                        dst[z] = zero_buffer[z];
                    }
                }
                else {
                    uint64_t src_noc_addr = get_noc_addr(
                        row_id, s);

                    uint32_t bank_id = row_id & (num_dram_channels - 1);
                    noc_async_read(src_noc_addr, l1_write_addr, row_size);
                    row_id++;
                }
                l1_write_addr += row_size;
                row_id_in_face++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, num_tiles_c);
        }
    }
}
