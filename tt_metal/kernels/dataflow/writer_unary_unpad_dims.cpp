#include <stdint.h>
#include "dataflow_api.h"

uint64_t round_down_32(uint64_t a){
    return (a >> 5) << 5;
}

void kernel_main() {

    // Constexpr
    constexpr uint32_t cb_id_out0                      = 16;
    constexpr uint32_t alignment                       = 32;

    const uint32_t dst_addr                 = get_arg_val<uint32_t>(0);
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
    const uint32_t cache_buffer_l1_addr     = get_arg_val<uint32_t>(11);
    const uint32_t temp_buffer_l1_addr      = get_arg_val<uint32_t>(12);

    tt_l1_ptr std::uint32_t* cache_buffer = (tt_l1_ptr uint32_t*)(cache_buffer_l1_addr);
    tt_l1_ptr std::uint32_t* temp_buffer = (tt_l1_ptr uint32_t*)(temp_buffer_l1_addr);


    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    const uint32_t num_tiles_c = num_total_X / 32; // Assuming 2 bytes per datum, there are 64 bytes per tile row
    uint32_t stick_id          = 0;

    #define stick_size_is_pow2 get_compile_time_arg_val(0) == 1
    #if (stick_size_is_pow2)
    const uint32_t log_base_2_of_page_size = get_arg_val<uint32_t>(13);
    const InterleavedPow2AddrGen<true> s = {
        .bank_base_address = dst_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen<true> s = {
        .bank_base_address = dst_addr,
        .page_size = unpadded_X_size
    };
    #endif

    uint32_t l1_cache_addr = cache_buffer_l1_addr;
    uint32_t padded_Z_diff_tile_rows = (num_total_Z - num_unpadded_Z) * num_total_Y / 32;
    uint32_t padded_W_diff_tile_rows = (num_total_W - num_unpadded_W) * num_total_Z * num_total_Y / 32;
    for (uint32_t w = 0; w < num_unpadded_W; w++) {
        for (uint32_t z = 0; z < num_unpadded_Z; z++) {
            uint32_t row_id_in_face = 0;
            for (uint32_t y_t = 0; y_t < num_total_Y / 32; y_t++) {
                cb_wait_front(cb_id_out0, num_tiles_c);
                if (row_id_in_face < num_unpadded_Y) {
                    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                    for (uint32_t k = 0; k < 32; k++) {
                        if (row_id_in_face < num_unpadded_Y) {
                            uint64_t dst_noc_addr = get_noc_addr(stick_id, s);
                            uint64_t round_down_addr = round_down_32(dst_noc_addr);
                            uint64_t diff_addr = dst_noc_addr - round_down_addr;

                            // Copy from cache to tmp buffer
                            volatile tt_l1_ptr std::uint32_t* cache = (volatile tt_l1_ptr uint32_t*)(l1_cache_addr);
                            volatile tt_l1_ptr std::uint32_t* temp = (volatile tt_l1_ptr uint32_t*)(temp_buffer_l1_addr);
                            for(uint32_t z = 0; z < (diff_addr) / 4; z++) {
                                temp[z] = cache[z];
                            }
                            // Copy from CB to tmp buffer
                            volatile tt_l1_ptr std::uint32_t* src = (volatile tt_l1_ptr uint32_t*)(l1_read_addr);
                            temp = (volatile tt_l1_ptr uint32_t*)(temp_buffer_l1_addr + diff_addr);
                            for(uint32_t z = 0; z < (unpadded_X_size) / 4; z++) {
                                temp[z] = src[z];
                            }

                            // Write out tmp buffer
                            noc_async_write(temp_buffer_l1_addr, round_down_addr, unpadded_X_size + diff_addr);

                            // Copy from tmp to cache
                            uint64_t next_round_down_addr = round_down_32(dst_noc_addr + unpadded_X_size);
                            uint64_t cache_to_write = dst_noc_addr + unpadded_X_size - next_round_down_addr;
                            temp = (volatile tt_l1_ptr uint32_t*)(temp_buffer_l1_addr + diff_addr + unpadded_X_size - cache_to_write);
                            for(uint32_t z = 0; z < (cache_to_write) / 4; z++) {
                                cache[z] = temp[z];
                            }
                            l1_read_addr += padded_X_size;
                            row_id_in_face++;
                            stick_id++;
                            // Go back to start of cache buffer when we go back to bank 0
                            if (stick_id & 7) {
                                l1_cache_addr += alignment;
                            } else {
                                l1_cache_addr = cache_buffer_l1_addr;
                            }

                            // Block write
                            noc_async_write_barrier();
                        } else {
                            break;
                        }
                    }
                }
                cb_pop_front(cb_id_out0, num_tiles_c);
            }
        }
        for (uint32_t i = 0; i < padded_Z_diff_tile_rows; i++) {
            cb_wait_front(cb_id_out0, num_tiles_c);
            cb_pop_front(cb_id_out0, num_tiles_c);
        }
    }
    for (uint32_t i = 0; i < padded_W_diff_tile_rows; i++) {
        cb_wait_front(cb_id_out0, num_tiles_c);
        cb_pop_front(cb_id_out0, num_tiles_c);
    }
}
