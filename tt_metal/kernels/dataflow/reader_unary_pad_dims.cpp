#include <stdint.h>
#include "dataflow_api.h"

uint64_t round_down_32(uint64_t a){
    return (a >> 5) << 5;
}

void kernel_main() {

    // Constexpr
    constexpr uint32_t cb_id_in0                       = 0;

    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t num_unpadded_W           = get_arg_val<uint32_t>(1);
    const uint32_t num_total_W              = get_arg_val<uint32_t>(2);
    const uint32_t num_unpadded_Z           = get_arg_val<uint32_t>(3);
    const uint32_t num_total_Z              = get_arg_val<uint32_t>(4);
    const uint32_t num_unpadded_Y           = get_arg_val<uint32_t>(5);
    const uint32_t num_total_Y              = get_arg_val<uint32_t>(6);
    const uint32_t num_unpadded_X           = get_arg_val<uint32_t>(7);
    const uint32_t num_total_X              = get_arg_val<uint32_t>(8);
    const uint32_t unpadded_X_size          = get_arg_val<uint32_t>(9);
    const uint32_t padded_X_size            = get_arg_val<uint32_t>(10);
    const uint32_t pad_value                = get_arg_val<uint32_t>(11);
    const uint32_t temp_buffer_l1_addr      = get_arg_val<uint32_t>(12);

    tt_l1_ptr std::uint32_t* temp_buffer = (tt_l1_ptr uint32_t*)(temp_buffer_l1_addr);

    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    const uint32_t num_tiles_c = padded_X_size / 64; // Assuming 2 bytes per datum, there are 64 bytes per tile row

    #define stick_size_is_pow2 get_compile_time_arg_val(0) == 1
    #if (stick_size_is_pow2)
    const uint32_t log_base_2_of_page_size = get_arg_val<uint32_t>(13);
    const InterleavedPow2AddrGen<true> s = {
        .bank_base_address = src_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen<true> s = {
        .bank_base_address = src_addr,
        .page_size = unpadded_X_size
    };
    #endif

    uint32_t row_id = 0;
    uint32_t padded_Z_diff_tile_rows = (num_total_Z - num_unpadded_Z) * num_total_Y / 32;
    uint32_t padded_W_diff_tile_rows = (num_total_W - num_unpadded_W) * num_total_Z * num_total_Y / 32;
    for (uint32_t w = 0; w < num_unpadded_W; w++) {
        for (uint32_t z = 0; z < num_unpadded_Z; z++) {
            uint32_t row_id_in_face = 0;
            for (uint32_t y_t = 0; y_t < num_total_Y / 32; y_t++) {
                cb_reserve_back(cb_id_in0, num_tiles_c);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                for (uint32_t k = 0; k < 32; k++) {
                    if (row_id_in_face >= num_unpadded_Y) {
                        // pad the tile by reading values from zero buffer in L1
                        volatile tt_l1_ptr std::uint32_t* dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr);
                        for(uint32_t z = 0; z < padded_X_size / 4; z++) {
                            dst[z] = pad_value;
                        }
                    }
                    else {
                        uint64_t src_noc_addr = get_noc_addr(
                            row_id, s);

                        // Read from DRAM to tmp buffer
                        uint64_t round_down_addr = round_down_32(src_noc_addr);
                        uint64_t diff_addr = src_noc_addr - round_down_addr;
                        noc_async_read(round_down_addr, temp_buffer_l1_addr, unpadded_X_size + diff_addr);

                        volatile tt_l1_ptr std::uint32_t* dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr + unpadded_X_size);
                        volatile tt_l1_ptr std::uint32_t* temp = (volatile tt_l1_ptr uint32_t*)(temp_buffer_l1_addr + diff_addr);

                        // Pad Columns first
                        for(uint32_t z = 0; z < (padded_X_size - unpadded_X_size) / 4; z++) {
                            dst[z] = pad_value;
                        }

                        // Block before copying data from tmp to cb buffer
                        noc_async_read_barrier();
                        dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr);
                        for(uint32_t z = 0; z < (unpadded_X_size) / 4; z++) {
                            dst[z] = temp[z];
                        }

                        row_id++;
                    }
                    l1_write_addr += padded_X_size;
                    row_id_in_face++;
                }

                cb_push_back(cb_id_in0, num_tiles_c);
            }
        }
        for (uint32_t i = 0; i < padded_Z_diff_tile_rows; i++) {
            cb_reserve_back(cb_id_in0, num_tiles_c);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            // pad the tile by reading values from zero buffer in L1
            volatile tt_l1_ptr std::uint32_t* dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr);
            // * 32 / 4
            for(uint32_t z = 0; z < padded_X_size * 8; z++) {
                dst[z] = pad_value;
            }
            l1_write_addr += padded_X_size;
            cb_push_back(cb_id_in0, num_tiles_c);
        }
    }
    for (uint32_t i = 0; i < padded_W_diff_tile_rows; i++) {
        cb_reserve_back(cb_id_in0, num_tiles_c);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        // pad the tile by reading values from zero buffer in L1
        volatile tt_l1_ptr std::uint32_t* dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr);
        // * 32 / 4
        for(uint32_t z = 0; z < padded_X_size * 8; z++) {
            dst[z] = pad_value;
        }
        l1_write_addr += padded_X_size;
        cb_push_back(cb_id_in0, num_tiles_c);
    }
}
