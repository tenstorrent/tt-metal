#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    // Constexpr
    constexpr uint32_t cb_id_in0                       = 0;
    constexpr uint32_t tile_height = 32;

    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t num_unpadded_W           = get_arg_val<uint32_t>(1);
    const uint32_t padded_W_diff_blocks     = get_arg_val<uint32_t>(2);
    const uint32_t num_unpadded_Z           = get_arg_val<uint32_t>(3);
    const uint32_t padded_Z_diff_blocks     = get_arg_val<uint32_t>(4);
    const uint32_t num_unpadded_Y           = get_arg_val<uint32_t>(5);
    const uint32_t padded_Y_diff_blocks     = get_arg_val<uint32_t>(6);
    const uint32_t num_leftover_Y           = get_arg_val<uint32_t>(7);
    const uint32_t num_unpadded_X           = get_arg_val<uint32_t>(8);
    const uint32_t unpadded_X_size          = get_arg_val<uint32_t>(9);
    const uint32_t padded_X_size            = get_arg_val<uint32_t>(10);
    const uint32_t pad_value                = get_arg_val<uint32_t>(11);
    const uint32_t num_blocks_w_input       = get_arg_val<uint32_t>(12);
    const uint32_t num_blocks_w_output      = get_arg_val<uint32_t>(13);
    const uint32_t num_blocks_w_diff        = get_arg_val<uint32_t>(14);
    const uint32_t block_row_size           = get_arg_val<uint32_t>(15);
    const uint32_t block_row_leftover_size  = get_arg_val<uint32_t>(16);

    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    const uint32_t num_tiles_block_c = block_row_size / 64; // Assuming 2 bytes per datum, there are 64 bytes per tile row

    constexpr bool src0_is_dram          = get_compile_time_arg_val(0) == 1;
    #define stick_size_is_pow2 get_compile_time_arg_val(1) == 1
    #if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(2);
    const InterleavedPow2AddrGen<src0_is_dram> s = {
        .bank_base_address = src_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen<src0_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = unpadded_X_size
    };
    #endif

    uint32_t stick_id = 0;


    auto pad_blocks = [&](uint32_t num_blocks) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            cb_reserve_back(cb_id_in0, num_tiles_block_c);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            // pad the tile by reading values from zero buffer in L1
            volatile std::uint32_t* dst = (volatile uint32_t*)(l1_write_addr);
            // 8 = tile_height / 4
            for(uint32_t z = 0; z < block_row_size * 8; z++) {
                dst[z] = pad_value;
            }
            cb_push_back(cb_id_in0, num_tiles_block_c);
        }
    };

    auto read_block = [&](uint32_t base_stick_id, uint32_t num_rows, uint32_t offset, uint32_t block_size) {
        cb_reserve_back(cb_id_in0, num_tiles_block_c);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        uint32_t curr_stick_id = base_stick_id;
        for (uint32_t k = 0; k < num_rows; k++) {
            uint64_t src_noc_addr = get_noc_addr(curr_stick_id + k, s) + offset;

            // Read from DRAM to tmp buffer
            noc_async_read(src_noc_addr, l1_write_addr, block_size);

            if (block_row_size > block_size) {
                volatile std::uint32_t* dst = (volatile uint32_t*)(l1_write_addr + block_size);
                for(uint32_t z = 0; z < (block_row_size - block_size) / 4; z++) {
                    dst[z] = pad_value;
                }
            }

            // Block before copying data from tmp to cb buffer
            noc_async_read_barrier();
            l1_write_addr += block_row_size;
        }
        if (num_rows < tile_height) {
            volatile std::uint32_t* dst = (volatile uint32_t*)(l1_write_addr);

            for(uint32_t z = 0; z < (block_row_size) / 4 * (tile_height - num_rows); z++) {
                dst[z] = pad_value;
            }
        }
        cb_push_back(cb_id_in0, num_tiles_block_c);
    };

    auto read_block_rows = [&] (uint32_t base_stick_id, uint32_t num_rows_block) {
        uint32_t block_row_offset = 0;

        for(uint32_t block_w = 0; block_w < num_blocks_w_input; block_w++) {
            read_block(base_stick_id, num_rows_block, block_row_offset, block_row_size);
            block_row_offset += block_row_size;
        }

        if (block_row_leftover_size > 0) {
            read_block(base_stick_id, num_rows_block, block_row_offset, block_row_leftover_size);
            block_row_offset += block_row_size;
        }
    };

    for (uint32_t w = 0; w < num_unpadded_W; w++) {
        for (uint32_t z = 0; z < num_unpadded_Z; z++) {
            for (uint32_t y_t = 0; y_t < num_unpadded_Y / tile_height; y_t++) {
                read_block_rows(stick_id, tile_height);
                // Read fully padded blocks
                pad_blocks(num_blocks_w_diff);
                stick_id += tile_height;
            }

            if (num_leftover_Y > 0) {
                read_block_rows(stick_id, num_leftover_Y);
                // Read fully padded blocks
                pad_blocks(num_blocks_w_diff);
                stick_id += num_leftover_Y;
            }
            pad_blocks(padded_Y_diff_blocks);
        }
        pad_blocks(padded_Z_diff_blocks);
    }
    pad_blocks(padded_W_diff_blocks);

}
