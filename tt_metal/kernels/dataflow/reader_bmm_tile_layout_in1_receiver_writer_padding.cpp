#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // in0 tensor args
    uint32_t in0_tensor_addr                    = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id           = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w                = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_stride_h                = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride       = get_arg_val<uint32_t>(4);

    // in0 block args
    uint32_t in0_block_w                        = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h                        = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles                = get_arg_val<uint32_t>(7);

    // in1 tensor args
    uint32_t in1_tensor_addr                    = get_arg_val<uint32_t>(8);
    uint32_t in1_tensor_start_tile_id           = get_arg_val<uint32_t>(9);
    uint32_t in1_tensor_stride_w                = get_arg_val<uint32_t>(10);
    uint32_t in1_tensor_stride_h                = get_arg_val<uint32_t>(11);
    uint32_t in1_tensor_next_block_stride       = get_arg_val<uint32_t>(12);

    // in1 block args
    uint32_t in1_block_w                        = get_arg_val<uint32_t>(13);
    uint32_t in1_block_h                        = get_arg_val<uint32_t>(14);
    uint32_t in1_block_num_tiles                = get_arg_val<uint32_t>(15);

    // in0/in1 common args
    uint32_t num_blocks                         = get_arg_val<uint32_t>(16);

    // in0 mcast args
    uint32_t in0_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(17);
    uint32_t in0_mcast_dest_noc_start_y         = get_arg_val<uint32_t>(18);
    uint32_t in0_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(19);
    uint32_t in0_mcast_dest_noc_end_y           = get_arg_val<uint32_t>(20);
    uint32_t in0_mcast_num_dests                = get_arg_val<uint32_t>(21);
    uint32_t in0_mcast_sender_noc_x             = get_arg_val<uint32_t>(22);
    uint32_t in0_mcast_sender_noc_y             = get_arg_val<uint32_t>(23);
    uint32_t in0_mcast_sender_semaphore_addr    = get_arg_val<uint32_t>(24);
    uint32_t in0_mcast_receiver_semaphore_addr  = get_arg_val<uint32_t>(25);

    // in1 mcast args
    uint32_t in1_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(26);
    uint32_t in1_mcast_dest_noc_start_y         = get_arg_val<uint32_t>(27);
    uint32_t in1_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(28);
    uint32_t in1_mcast_dest_noc_end_y           = get_arg_val<uint32_t>(29);
    uint32_t in1_mcast_num_dests                = get_arg_val<uint32_t>(30);
    uint32_t in1_mcast_sender_noc_x             = get_arg_val<uint32_t>(31);
    uint32_t in1_mcast_sender_noc_y             = get_arg_val<uint32_t>(32);
    uint32_t in1_mcast_sender_semaphore_addr    = get_arg_val<uint32_t>(33);
    uint32_t in1_mcast_receiver_semaphore_addr  = get_arg_val<uint32_t>(34);

    // batch args
    uint32_t MtKt                               = get_arg_val<uint32_t>(35); // if 0
    uint32_t KtNt                               = get_arg_val<uint32_t>(36);
    uint32_t batch                              = get_arg_val<uint32_t>(37);
    uint32_t bcast_B                            = get_arg_val<uint32_t>(38);

    // WRITER
    // out tensor args
    uint32_t out_tensor_addr                    = get_arg_val<uint32_t>(39);
    uint32_t out_tensor_start_tile_id           = get_arg_val<uint32_t>(40);
    uint32_t out_tensor_stride_w                = get_arg_val<uint32_t>(41);
    uint32_t out_tensor_stride_h                = get_arg_val<uint32_t>(42);
    uint32_t out_tensor_next_subblock_stride_w  = get_arg_val<uint32_t>(43);
    uint32_t out_tensor_next_subblock_stride_h  = get_arg_val<uint32_t>(44);

    // out subblock args
    uint32_t out_subblock_w                     = get_arg_val<uint32_t>(45);
    uint32_t out_subblock_h                     = get_arg_val<uint32_t>(46);
    uint32_t out_subblock_tile_count            = get_arg_val<uint32_t>(47);
    uint32_t out_num_subblocks_w                = get_arg_val<uint32_t>(48);
    uint32_t out_num_subblocks_h                = get_arg_val<uint32_t>(49);

    // batch args
    uint32_t MtNt                               = get_arg_val<uint32_t>(50); // if 0
    // Don't need batch; same as batch from READER args

    // padding args (WRITER)
    uint32_t out_num_nonzero_subblocks_h        = get_arg_val<uint32_t>(51);
    uint32_t out_last_subblock_h                = get_arg_val<uint32_t>(52);
    uint32_t padded_block_tiles_h_skip          = get_arg_val<uint32_t>(53);
    uint32_t out_num_nonzero_subblocks_w        = get_arg_val<uint32_t>(54);
    uint32_t out_last_subblock_w                = get_arg_val<uint32_t>(55);
    uint32_t padded_subblock_tiles_addr_skip    = get_arg_val<uint32_t>(56);
    uint32_t padded_block_tiles_w_skip          = get_arg_val<uint32_t>(57);

    // WRITER
    // const args for tile-based bank-swizzled layout
    // could be added to the arg list in the future to test different
    // bank-swizzling configurations
    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    // WRITER
    constexpr uint32_t cb_id_out0 = 16;

    // WRITER
    // single-tile
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);

    volatile uint32_t* in1_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(in1_mcast_receiver_semaphore_addr);

    // WRITER
    #define tile_size_is_pow2 get_compile_time_arg_val(0) == 1
    #if (tile_size_is_pow2)
    constexpr uint32_t tile_size_pow2_exponent = get_compile_time_arg_val(1);
    const InterleavedPow2AddrGen s = {
        .bank_base_address = out_tensor_addr,
        .num_used_banks = num_used_dram_ch,
        .log_base_2_of_num_used_banks = num_used_dram_ch_pow2_exponent,
        .log_base_2_of_bank_unit_size = tile_size_pow2_exponent // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen s = {
        .bank_base_address = out_tensor_addr,
        .num_used_banks = num_used_dram_ch,
        .log_base_2_of_num_used_banks = num_used_dram_ch_pow2_exponent,
        .bank_unit_size = single_tile_size_bytes
    };
    #endif

    for (uint32_t b = 0; b < batch; b++) {
        for(uint32_t block = 0; block < num_blocks; block++) {
            // Operand 1
            cb_reserve_back(cb_id_in1, in1_block_num_tiles);

            // Set in1 semaphore value to INVALID
            noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, INVALID);

            // Atomic increment source core counter
            uint64_t in1_mcast_sender_semaphore_noc_addr = get_noc_addr(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, in1_mcast_sender_semaphore_addr);
            noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1);

            // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
            noc_semaphore_wait(in1_mcast_receiver_semaphore_addr_ptr, VALID);

            cb_push_back(cb_id_in1, in1_block_num_tiles);
        }
    }

    // WRITER
    for (uint32_t b = 0; b < batch; b++) {
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for(uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h; sbh++) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for(uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w; sbw++) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                uint32_t out_subblock_h_ = out_subblock_h;
                uint32_t out_subblock_w_ = out_subblock_w;
                uint32_t subblock_tiles_addr_skip = 0;
                if (sbh == out_num_nonzero_subblocks_h - 1) {
                    out_subblock_h_ = out_last_subblock_h;
                }
                if (sbw == out_num_nonzero_subblocks_w - 1) {
                    out_subblock_w_ = out_last_subblock_w;
                    subblock_tiles_addr_skip = padded_subblock_tiles_addr_skip;
                }

                cb_wait_front(cb_id_out0, out_subblock_tile_count);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                for(uint32_t h = 0; h < out_subblock_h_; h++) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for(uint32_t w = 0; w < out_subblock_w_; w++) {
                        uint64_t out_tensor_tile_noc_addr = get_noc_addr(out_tensor_tile_id, s);

                        noc_async_write(l1_read_addr, out_tensor_tile_noc_addr, single_tile_size_bytes);
                        l1_read_addr+=single_tile_size_bytes;

                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    // Skip padded tiles in subblock along row
                    l1_read_addr += subblock_tiles_addr_skip;
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc_async_write_barrier();
                cb_pop_front(cb_id_out0, out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            // Pop fully padded subblocks along the row
            cb_wait_front(cb_id_out0, padded_block_tiles_w_skip);
            cb_pop_front(cb_id_out0, padded_block_tiles_w_skip);
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        // Pop row(s) of fully padded subblocks
        cb_wait_front(cb_id_out0, padded_block_tiles_h_skip);
        cb_pop_front(cb_id_out0, padded_block_tiles_h_skip);
        out_tensor_start_tile_id += MtNt;
    }
}
