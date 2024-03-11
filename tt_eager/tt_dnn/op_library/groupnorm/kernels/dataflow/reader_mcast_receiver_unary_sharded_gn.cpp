// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// #include "debug/dprint.h"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t reduce_receiver_semaphore_addr  = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_addr    = get_compile_time_arg_val(1);
    constexpr uint32_t num_group                     = get_compile_time_arg_val(2);
    constexpr uint32_t num_batch                    = get_compile_time_arg_val(3);
    constexpr uint32_t per_core_N                     = get_compile_time_arg_val(4);
    constexpr uint32_t is_num_channel_div_by_tile     = get_compile_time_arg_val(5);
    constexpr uint32_t num_cols_per_group         = get_compile_time_arg_val(6);
    constexpr uint32_t group_offset         = get_compile_time_arg_val(7);
    constexpr uint32_t num_nz_rows_per_tile         = get_compile_time_arg_val(8);
    constexpr uint32_t batch_offset       = get_compile_time_arg_val(9);
    constexpr uint32_t block_h                        = get_compile_time_arg_val(10);
    constexpr uint32_t block_w                        = get_compile_time_arg_val(11);
    constexpr uint32_t block_h_offset                        = get_compile_time_arg_val(12);
    constexpr uint32_t block_w_offset                        = get_compile_time_arg_val(13);
    const uint32_t per_core_N_bytes = per_core_N * 2;

    const uint32_t mcast_sender_noc_x         = get_arg_val<uint32_t>(0);
    const uint32_t mcast_sender_noc_y         = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0; // E[x] partial reduce
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce
    constexpr uint32_t cb_in0 = tt::CB::c_in0; // sharded cb
    constexpr uint32_t cb_in = tt::CB::c_in7; // for pick values

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial); // tile size
    const DataFormat data_format = get_dataformat(cb_ex_partial); // data format

    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);

    const uint64_t reduce_receiver_semaphore_noc_addr = get_noc_addr(mcast_sender_noc_x, mcast_sender_noc_y, reduce_receiver_semaphore_addr);

    volatile tt_l1_ptr uint16_t* rptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_in0));
    uint32_t in0_l1_read_addr = get_read_ptr(cb_in0);

    uint32_t batch_index = 0;
    for (uint32_t i=0; i < num_batch; ++i) {
        uint32_t group_index = 0;
        for (uint32_t j=0; j < num_group; ++j) {

            // pick values from sharded input cb to cb
            uint32_t h_index = 0;
            for (uint32_t h=0; h < block_h; ++h) {
                uint32_t w_index = 0;
                for (uint32_t w=0; w < block_w; ++w) {
                    cb_reserve_back(cb_in, 1);

                    uint32_t group_batch_index_offset = batch_index + group_index + h_index + w_index;

                    if (w == block_w - 1 and not is_num_channel_div_by_tile) {
                        volatile tt_l1_ptr uint16_t* wptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_in));
                        uint32_t read_l1_index = group_batch_index_offset;
                        uint32_t write_l1_index = 0;
                        for (uint32_t t=0; t < num_nz_rows_per_tile; ++t) {
                            for (uint32_t c=0; c < num_cols_per_group; ++c) {
                                wptr[c + write_l1_index] = rptr[c + read_l1_index];
                            }
                            read_l1_index += per_core_N;
                            write_l1_index += 32;
                        }
                    } else {

                        volatile tt_l1_ptr uint16_t* wptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_in));
                        uint32_t read_l1_index = group_batch_index_offset;
                        uint32_t write_l1_index = 0;
                        for (uint32_t t=0; t < num_nz_rows_per_tile; ++t) {
                            for (uint32_t c=0; c < 32; ++c) {
                                wptr[c + write_l1_index] = rptr[c + read_l1_index];
                            }
                            read_l1_index += per_core_N;
                            write_l1_index += 32;
                        }
                    }

                    cb_push_back(cb_in, 1);
                    w_index += block_w_offset;
                }
                h_index += block_h_offset;
            }

            for (uint32_t i=0; i < 2; ++i) {
                // wait for local data ready
                cb_wait_front(cb_ex_partial, 1);
                noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
                cb_reserve_back(cb_ex_global, 1);
                noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
                noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
                cb_push_back(cb_ex_global, 1);
                cb_pop_front(cb_ex_partial, 1);
            }

            group_index += group_offset;
        }

        batch_index += batch_offset;
    }

}
