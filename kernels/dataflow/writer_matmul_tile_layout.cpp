#include "dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {

    kernel_profiler::mark_time(4);

    // out tensor args
    uint32_t out_tensor_addr                         = get_arg_val<uint32_t>(0);
    uint32_t out_tensor_start_tile_id                = get_arg_val<uint32_t>(1);
    uint32_t out_tensor_stride_w                     = get_arg_val<uint32_t>(2);
    uint32_t out_tensor_stride_h                     = get_arg_val<uint32_t>(3);
    uint32_t out_tensor_next_subblock_stride_w       = get_arg_val<uint32_t>(4);
    uint32_t out_tensor_next_subblock_stride_h       = get_arg_val<uint32_t>(5);

    // out subblock args
    uint32_t out_subblock_w                   = get_arg_val<uint32_t>(6);
    uint32_t out_subblock_h                   = get_arg_val<uint32_t>(7);
    uint32_t out_subblock_tile_count          = get_arg_val<uint32_t>(8);
    uint32_t out_num_subblocks_w              = get_arg_val<uint32_t>(9);
    uint32_t out_num_subblocks_h              = get_arg_val<uint32_t>(10);

    // const args for tile-based bank-swizzled layout
    // could be added to the arg list in the future to test different
    // bank-swizzling configurations
    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    constexpr uint32_t tile_size_pow2_exponent = 11;

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);

    const InterleavedPow2AddrGen s = {
        .bank_base_address = out_tensor_addr,
        .num_used_banks = num_used_dram_ch,
        .log_base_2_of_num_used_banks = num_used_dram_ch_pow2_exponent,
        .log_base_2_of_bank_unit_size = tile_size_pow2_exponent
    };


    bool one_time_profile = true;
    uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
    for(uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
        uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
        for(uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
            uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

            cb_wait_front(cb_id_out0, out_subblock_tile_count);
            kernel_profiler::mark_time_once(5, &one_time_profile);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

            for(uint32_t h = 0; h < out_subblock_h; h++) {
                uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                for(uint32_t w = 0; w < out_subblock_w; w++) {
                    uint64_t out_tensor_tile_noc_addr = get_noc_addr(out_tensor_tile_id, s);

                    noc_async_write(l1_read_addr, out_tensor_tile_noc_addr, single_tile_size_bytes);
                    l1_read_addr+=single_tile_size_bytes;

                    out_tensor_tile_id += out_tensor_stride_w;
                }
                out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
            }

            noc_async_write_barrier();
            cb_pop_front(cb_id_out0, out_subblock_tile_count);
            out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
        }
        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
    }
    kernel_profiler::mark_time(6);
}
