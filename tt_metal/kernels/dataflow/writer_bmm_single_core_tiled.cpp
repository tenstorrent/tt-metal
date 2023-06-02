#include "dataflow_api.h"

void kernel_main() {

    // const uint32_t out_addr                   = get_arg_val<uint32_t>(0);
    // // uint32_t out_start_tile_id          = get_arg_val<uint32_t>(1);
    // const uint32_t out_stride_w               = get_arg_val<uint32_t>(2);
    // const uint32_t out_stride_h               = get_arg_val<uint32_t>(3);
    // const uint32_t out_next_subblock_stride_w = get_arg_val<uint32_t>(4);
    // const uint32_t out_next_subblock_stride_h = get_arg_val<uint32_t>(5);
    // const uint32_t out_subblock_w             = get_arg_val<uint32_t>(6);
    // const uint32_t out_subblock_h             = get_arg_val<uint32_t>(7);
    // const uint32_t out_subblock_tile_count    = get_arg_val<uint32_t>(8);
    // const uint32_t out_num_subblocks_w        = get_arg_val<uint32_t>(9);
    // const uint32_t out_num_subblocks_h        = get_arg_val<uint32_t>(10);
    // const uint32_t out_num_blocks_w           = get_arg_val<uint32_t>(11);
    // const uint32_t out_num_blocks_h           = get_arg_val<uint32_t>(12);

    uint32_t out_addr                   = get_arg_val<uint32_t>(0);
    uint32_t out_stride_w               = get_arg_val<uint32_t>(2);
    uint32_t out_stride_h               = get_arg_val<uint32_t>(3);
    uint32_t out_next_subblock_stride_w = get_arg_val<uint32_t>(4);
    uint32_t out_next_subblock_stride_h = get_arg_val<uint32_t>(5);
    uint32_t out_subblock_w             = get_arg_val<uint32_t>(6);
    uint32_t out_subblock_h             = get_arg_val<uint32_t>(7);
    uint32_t out_subblock_tile_count    = get_arg_val<uint32_t>(8);
    uint32_t out_num_subblocks_w        = get_arg_val<uint32_t>(9);
    uint32_t out_num_subblocks_h        = get_arg_val<uint32_t>(10);
    // uint32_t out_num_blocks_w           = get_arg_val<uint32_t>(11);
    // uint32_t out_num_blocks_h           = get_arg_val<uint32_t>(12);

    constexpr uint32_t out_cb_id = tt::CB::c_out0;

    constexpr uint32_t tile_size_pow2_exponent = 11;    // == 2^11 = 2048 = 2 * 32 * 32 (assuming dtype = 2 bytes)
    const InterleavedPow2AddrGen<true> s = {
        .bank_base_address = out_addr,
        .log_base_2_of_page_size = tile_size_pow2_exponent
    };
    const uint32_t tile_size_bytes = get_tile_size(out_cb_id);

    // DPRINT << "TILE_SIZE_BYTES: " << tile_size_bytes << " bytes" << ENDL();

    uint32_t out_sbh_start_tile_id = 0;
    for(uint32_t sbh = 0; sbh < out_num_subblocks_h; ++sbh) {
        uint32_t out_sbw_start_tile_id = out_sbh_start_tile_id;
        for(uint32_t sbw = 0; sbw < out_num_subblocks_w; ++sbw) {
            uint32_t out_sb_row_start_tile_id = out_sbw_start_tile_id;
            // wait for one subblock worth tiles
            // DPRINT << " ==== WAITING FOR OUT CB TILE COUNT: " << out_subblock_tile_count << ENDL();
            cb_wait_front(out_cb_id, out_subblock_tile_count);
            uint32_t l1_read_addr = get_read_ptr(out_cb_id);
            for(uint32_t h = 0; h < out_subblock_h; ++h) {
                uint32_t out_tile_id = out_sb_row_start_tile_id;
                for(uint32_t w = 0; w < out_subblock_w; ++w) {
                    uint64_t out_tile_noc_addr = get_noc_addr(out_tile_id, s);
                    ///////// for debug
                    // uint32_t addr = ((out_tile_id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) << tile_size_pow2_exponent) + out_addr;
                    // uint32_t noc_x = dram_bank_to_noc_x[out_tile_id];
                    // uint32_t noc_y = dram_bank_to_noc_y[out_tile_id];
                    // DPRINT << "SUBBLOCK: " << sbh << "," << sbw << " :: TILE: " << h << "," << w << " :: out_tile_id: " << out_tile_id << " :: addr: " << addr << " on: " << noc_x << "," << noc_y << " :: from " << l1_read_addr << ENDL();
                    ////////////////////
                    // DPRINT << "WRITING: " << tile_size_bytes << ENDL();
                    noc_async_write(l1_read_addr, out_tile_noc_addr, tile_size_bytes);
                    l1_read_addr += tile_size_bytes;
                    out_tile_id += out_stride_w;
                }
                out_sb_row_start_tile_id += out_stride_h;
            }
            noc_async_write_barrier();
            cb_pop_front(out_cb_id, out_subblock_tile_count);
            // DPRINT << " ==== POPPING OUT CB TILE COUNT: " << out_subblock_tile_count << ENDL();
            out_sbw_start_tile_id += out_next_subblock_stride_w;

            // DPRINT << "DONE: " << sbh << "," << sbw << ENDL();
        }
        out_sbh_start_tile_id += out_next_subblock_stride_h;
    }

    // DPRINT << "OOOOOUUUUUCCCCHHH!!! " << ENDL();
}
