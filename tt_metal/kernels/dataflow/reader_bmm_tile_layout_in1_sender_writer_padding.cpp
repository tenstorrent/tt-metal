#include <stdint.h>
#include "dataflow_kernel_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // READER
    // in1 tensor args
    uint32_t in1_tensor_addr                    = get_arg_val<uint32_t>(0);
    uint32_t in1_tensor_start_tile_id           = get_arg_val<uint32_t>(1);
    // in1 mcast args
    uint32_t in1_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(2);
    uint32_t in1_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(3);

    // WRITER
    // out tensor args
    uint32_t out_tensor_addr                    = get_arg_val<uint32_t>(4);
    uint32_t out_tensor_start_tile_id           = get_arg_val<uint32_t>(5);

    // padding args (READER)
    uint32_t last_block_w                       = get_arg_val<uint32_t>(6);
    // padding args (WRITER)
    uint32_t out_num_nonzero_subblocks_h        = get_arg_val<uint32_t>(7);
    uint32_t out_last_subblock_h                = get_arg_val<uint32_t>(8);
    uint32_t padded_block_tiles_h_skip          = get_arg_val<uint32_t>(9);
    uint32_t out_num_nonzero_subblocks_w        = get_arg_val<uint32_t>(10);
    uint32_t out_last_subblock_w                = get_arg_val<uint32_t>(11);
    uint32_t padded_subblock_tiles_addr_skip    = get_arg_val<uint32_t>(12);
    uint32_t padded_block_tiles_w_skip          = get_arg_val<uint32_t>(13);

    // COMPILE TIME ARGS
    // dataflow::Interleaved accessor args
    // dataflow::Interleaved accessor args
    constexpr DataFormat data_format                      = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr bool in1_is_dram                            = get_compile_time_arg_val(1) == 1;
    constexpr bool out_is_dram                            = get_compile_time_arg_val(2) == 1;

    // READER
    // in1 tensor args
    constexpr uint32_t in1_tensor_stride_w                = get_compile_time_arg_val(3);
    constexpr uint32_t in1_tensor_stride_h                = get_compile_time_arg_val(4);
    constexpr uint32_t in1_tensor_next_block_stride       = get_compile_time_arg_val(5);
    // in1 block args
    constexpr uint32_t in1_block_w                        = get_compile_time_arg_val(6);
    constexpr uint32_t in1_block_h                        = get_compile_time_arg_val(7);
    constexpr uint32_t in1_block_num_tiles                = get_compile_time_arg_val(8);
    // in0/in1 common args
    constexpr uint32_t num_blocks                         = get_compile_time_arg_val(9);
    // in1 mcast args
    constexpr uint32_t in1_mcast_dest_noc_start_y         = get_compile_time_arg_val(10);
    constexpr uint32_t in1_mcast_dest_noc_end_y           = get_compile_time_arg_val(11);
    constexpr uint32_t in1_mcast_sender_semaphore_addr    = get_compile_time_arg_val(12);
    constexpr uint32_t in1_mcast_receiver_semaphore_addr  = get_compile_time_arg_val(13);
    constexpr uint32_t in1_mcast_num_dests                = get_compile_time_arg_val(14);
    // batch args
    constexpr uint32_t KtNt                               = get_compile_time_arg_val(15);
    constexpr uint32_t batch                              = get_compile_time_arg_val(16);
    constexpr uint32_t bcast_B                            = get_compile_time_arg_val(17);

    // WRITER
    // out tensor args
    constexpr uint32_t out_tensor_stride_w                = get_compile_time_arg_val(18);
    constexpr uint32_t out_tensor_stride_h                = get_compile_time_arg_val(19);
    constexpr uint32_t out_tensor_next_subblock_stride_w  = get_compile_time_arg_val(20);
    constexpr uint32_t out_tensor_next_subblock_stride_h  = get_compile_time_arg_val(21);
    // out subblock args
    constexpr uint32_t out_subblock_w                     = get_compile_time_arg_val(22);
    constexpr uint32_t out_subblock_h                     = get_compile_time_arg_val(23);
    constexpr uint32_t out_subblock_tile_count            = get_compile_time_arg_val(24);
    // batch args
    constexpr uint32_t MtNt                               = get_compile_time_arg_val(25); // if 0
    // Don't need batch; same as batch from READER args

    #ifdef FUSE_BIAS
        // in3 mcast args
        uint32_t in3_tensor_addr                    = get_arg_val<uint32_t>(14);
        uint32_t in3_tensor_start_tile_id           = get_arg_val<uint32_t>(15);
        uint32_t in3_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(16);
        uint32_t in3_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(17);
        // in3 mcast args
        constexpr bool in3_is_dram                            = get_compile_time_arg_val(26) == 1;
        constexpr uint32_t in3_tensor_stride_w                = get_compile_time_arg_val(27);
        constexpr uint32_t in3_mcast_dest_noc_start_y         = get_compile_time_arg_val(28);
        constexpr uint32_t in3_mcast_dest_noc_end_y           = get_compile_time_arg_val(29);
        constexpr uint32_t in3_mcast_sender_semaphore_addr    = get_compile_time_arg_val(30);
        constexpr uint32_t in3_mcast_receiver_semaphore_addr  = get_compile_time_arg_val(31);
        constexpr uint32_t in3_mcast_num_dests                = get_compile_time_arg_val(32);

        constexpr uint32_t cb_id_in3 = 3;

        uint32_t l1_write_addr_in3;
        volatile uint32_t* in3_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(in3_mcast_receiver_semaphore_addr);
        volatile uint32_t* in3_mcast_sender_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(in3_mcast_sender_semaphore_addr);
    #endif

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_in2 = 2; // Dummy cb containing one tile of zeros for padding

    // WRITER
    constexpr uint32_t cb_id_out0 = 16;

    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);
    // uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0); // Should be same

    uint32_t l1_write_addr_in1;
    uint32_t l1_zeros_addr_in2 = dataflow::get_write_ptr(cb_id_in2);

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile uint32_t* in1_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(in1_mcast_receiver_semaphore_addr);
    *(in1_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile uint32_t* in1_mcast_sender_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(in1_mcast_sender_semaphore_addr);

    const dataflow::InterleavedAddrGenFast<in1_is_dram> s1 = {
        .bank_base_address = in1_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = data_format
    };
    #ifdef FUSE_BIAS
        const dataflow::InterleavedAddrGenFast<in3_is_dram> s3 = {
            .bank_base_address = in3_tensor_addr,
            .page_size = single_tile_size_bytes,
            .data_format = data_format
        };
    #endif
    // WRITER
    const dataflow::InterleavedAddrGenFast<out_is_dram> s = {
        .bank_base_address = out_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = data_format
    };


    for (uint32_t b = 0; b < batch; b++) {
        uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
        for(uint32_t block = 0; block < num_blocks; block++) {
            // Operand 1
            dataflow::cb_reserve_back(cb_id_in1, in1_block_num_tiles);
            l1_write_addr_in1 = dataflow::get_write_ptr(cb_id_in1);

            uint32_t in1_start_address = l1_write_addr_in1; // copy start address of block, to be used for mcasting
            uint32_t in1_block_size_bytes = 0; // can be optimized later, pass it to kernel

            // Copy in1 block into CB, as the default kernel
            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for(uint32_t h = 0; h < in1_block_h; h++) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for(uint32_t w = 0; w < in1_block_w; w++) {
                    if (w < last_block_w) {
                        //uint64_t in1_tile_noc_address = dataflow::get_noc_addr(in1_tensor_tile_id, s1);
                        //noc_async_read(in1_tile_noc_address, l1_write_addr_in1, single_tile_size_bytes);
                        dataflow::noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);
                    }
                    else
                        dataflow::noc_async_read(l1_zeros_addr_in2, l1_write_addr_in1, single_tile_size_bytes);
                    l1_write_addr_in1 += single_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                    in1_block_size_bytes += single_tile_size_bytes;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

            // Barrier! make sure the reads are done
            dataflow::noc_async_read_barrier();

            // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr (i.e. its value should be in0_mcast_num_dests), then reset
            // the semaphore_addr value back to zero for the next block
            dataflow_internal::noc_semaphore_wait(in1_mcast_sender_semaphore_addr_ptr, in1_mcast_num_dests);
            dataflow_internal::noc_semaphore_set(in1_mcast_sender_semaphore_addr_ptr, 0);

            // Now we have the block in the CB address, we can mcast to dests!
            uint64_t in1_multicast_data_addr = dataflow_internal::get_noc_multicast_addr(
            in1_mcast_dest_noc_start_x,
            in1_mcast_dest_noc_start_y,
            in1_mcast_dest_noc_end_x,
            in1_mcast_dest_noc_end_y,
            in1_start_address);
            // num_dests must not include source, since we are NOT really doing a local copy!
            dataflow::noc_async_write_multicast(in1_start_address, in1_multicast_data_addr, in1_block_size_bytes, in1_mcast_num_dests);

            // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same cmd_buf
            // Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

            // We should also multicast the flag to destinations
            uint64_t in1_mcast_receiver_semaphore_noc_addr = dataflow_internal::get_noc_multicast_addr(
            in1_mcast_dest_noc_start_x,
            in1_mcast_dest_noc_start_y,
            in1_mcast_dest_noc_end_x,
            in1_mcast_dest_noc_end_y,
            in1_mcast_receiver_semaphore_addr);
            // num_dests must not include source, since we are NOT really doing a local copy!
            dataflow_internal::noc_semaphore_set_multicast(in1_mcast_receiver_semaphore_addr, in1_mcast_receiver_semaphore_noc_addr, in1_mcast_num_dests);

            dataflow::cb_push_back(cb_id_in1, in1_block_num_tiles);
        }
        #ifdef FUSE_BIAS
            *(in3_mcast_receiver_semaphore_addr_ptr) = VALID;
            // Operand 1
            dataflow::cb_reserve_back(cb_id_in3, in1_block_w);
            l1_write_addr_in3 = dataflow::get_write_ptr(cb_id_in3);

            uint32_t in3_start_address = l1_write_addr_in3; // copy start address of block, to be used for mcasting
            uint32_t in3_block_size_bytes = 0; // can be optimized later, pass it to kernel

            // Copy in1 block into CB, as the default kernel
            uint32_t in3_tensor_tile_id = in3_tensor_start_tile_id;
            for(uint32_t w = 0; w < in1_block_w; w++) {
                if (w < last_block_w) {
                    //uint64_t in1_tile_noc_address = dataflow::get_noc_addr(in1_tensor_tile_id, s1);
                    //noc_async_read(in1_tile_noc_address, l1_write_addr_in1, single_tile_size_bytes);
                    dataflow::noc_async_read_tile(in3_tensor_tile_id, s3, l1_write_addr_in3);
                }
                else
                    dataflow::noc_async_read(l1_zeros_addr_in2, l1_write_addr_in3, single_tile_size_bytes);
                l1_write_addr_in3 += single_tile_size_bytes;
                in3_tensor_tile_id += in3_tensor_stride_w;
                in3_block_size_bytes += single_tile_size_bytes;
            }
            // Barrier! make sure the reads are done
            dataflow::noc_async_read_barrier();

            // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr (i.e. its value should be in0_mcast_num_dests), then reset
            // the semaphore_addr value back to zero for the next block
            dataflow_internal::noc_semaphore_wait(in3_mcast_sender_semaphore_addr_ptr, in3_mcast_num_dests);
            dataflow_internal::noc_semaphore_set(in3_mcast_sender_semaphore_addr_ptr, 0);

            // Now we have the block in the CB address, we can mcast to dests!
            uint64_t in3_multicast_data_addr = dataflow_internal::get_noc_multicast_addr(
            in3_mcast_dest_noc_start_x,
            in3_mcast_dest_noc_start_y,
            in3_mcast_dest_noc_end_x,
            in3_mcast_dest_noc_end_y,
            in3_start_address);
            // num_dests must not include source, since we are NOT really doing a local copy!
            dataflow::noc_async_write_multicast(in3_start_address, in3_multicast_data_addr, in3_block_size_bytes, in3_mcast_num_dests);

            // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same cmd_buf
            // Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

            // We should also multicast the flag to destinations
            uint64_t in3_mcast_receiver_semaphore_noc_addr = dataflow_internal::get_noc_multicast_addr(
            in3_mcast_dest_noc_start_x,
            in3_mcast_dest_noc_start_y,
            in3_mcast_dest_noc_end_x,
            in3_mcast_dest_noc_end_y,
            in3_mcast_receiver_semaphore_addr);
            // num_dests must not include source, since we are NOT really doing a local copy!
            dataflow_internal::noc_semaphore_set_multicast(in3_mcast_receiver_semaphore_addr, in3_mcast_receiver_semaphore_noc_addr, in3_mcast_num_dests);

            dataflow::cb_push_back(cb_id_in3, in1_block_w);
        #endif
        if (bcast_B == 0) {
            in1_tensor_start_tile_id += KtNt;
        }

        // WRITER
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

                dataflow::cb_wait_front(cb_id_out0, out_subblock_tile_count);
                uint32_t l1_read_addr = dataflow::get_read_ptr(cb_id_out0);

                for(uint32_t h = 0; h < out_subblock_h_; h++) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for(uint32_t w = 0; w < out_subblock_w_; w++) {
                        //uint64_t out_tensor_tile_noc_addr = dataflow::get_noc_addr(out_tensor_tile_id, s);
                        //noc_async_write(l1_read_addr, out_tensor_tile_noc_addr, single_tile_size_bytes);
                        dataflow::noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);

                        l1_read_addr+=single_tile_size_bytes;

                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    // Skip padded tiles in subblock along row
                    l1_read_addr += subblock_tiles_addr_skip;
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                dataflow::noc_async_write_barrier();
                dataflow::cb_pop_front(cb_id_out0, out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            // Pop fully padded subblocks along the row
            dataflow::cb_wait_front(cb_id_out0, padded_block_tiles_w_skip);
            dataflow::cb_pop_front(cb_id_out0, padded_block_tiles_w_skip);
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        // Pop row(s) of fully padded subblocks
        dataflow::cb_wait_front(cb_id_out0, padded_block_tiles_h_skip);
        dataflow::cb_pop_front(cb_id_out0, padded_block_tiles_h_skip);
        out_tensor_start_tile_id += MtNt;
    }
}
