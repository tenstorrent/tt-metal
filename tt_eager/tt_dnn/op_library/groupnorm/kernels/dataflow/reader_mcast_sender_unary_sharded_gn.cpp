// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
// #include "debug/dprint.h"


// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t reduce_receiver_semaphore_addr = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_addr   = get_compile_time_arg_val(1);
    constexpr uint32_t num_mcast_cores                = get_compile_time_arg_val(2);
    constexpr uint32_t num_group                     = get_compile_time_arg_val(3);
    constexpr uint32_t num_batch                    = get_compile_time_arg_val(4);
    constexpr uint32_t per_core_N                     = get_compile_time_arg_val(5);
    constexpr uint32_t is_num_channel_div_by_tile     = get_compile_time_arg_val(6);
    constexpr uint32_t num_cols_per_group         = get_compile_time_arg_val(7);
    constexpr uint32_t group_offset         = get_compile_time_arg_val(8);
    constexpr uint32_t num_nz_rows_per_tile         = get_compile_time_arg_val(9);
    constexpr uint32_t batch_offset       = get_compile_time_arg_val(10);
    constexpr uint32_t block_h                        = get_compile_time_arg_val(11);
    constexpr uint32_t block_w                        = get_compile_time_arg_val(12);
    constexpr uint32_t block_h_offset                        = get_compile_time_arg_val(13);
    constexpr uint32_t block_w_offset                        = get_compile_time_arg_val(14);
    constexpr uint32_t datum_size_bytes                      = get_compile_time_arg_val(15);
    const uint32_t per_core_N_bytes = per_core_N * 2;

    // DPRINT << "num_mcast_cores " <<num_mcast_cores<<ENDL();
    // DPRINT << "per_core_N " <<per_core_N<<ENDL();
    // DPRINT << "is_num_channel_div_by_tile " <<is_num_channel_div_by_tile<<ENDL();
    // DPRINT << "num_cols_per_group " <<num_cols_per_group<<ENDL();
    // DPRINT << "num_group " <<num_group<<ENDL();
    // DPRINT << "num_batch " <<num_batch<<ENDL();
    // DPRINT << "group_offset " <<group_offset<<ENDL();
    // DPRINT << "batch_offset " <<batch_offset<<ENDL();

    // DPRINT << "block_h " <<block_h<<ENDL();
    // DPRINT << "block_w " <<block_w<<ENDL();
    // DPRINT << "block_h_offset " <<block_h_offset<<ENDL();
    // DPRINT << "block_w_offset " <<block_w_offset<<ENDL();

    const bool has_mcast_first_group                    = get_arg_val<uint32_t>(0);
    const bool has_mcast_last_group                     = get_arg_val<uint32_t>(1);

    // mid mcast group
    const uint32_t mcast_dest_noc_start_x               = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_start_y               = get_arg_val<uint32_t>(3);
    const uint32_t mcast_dest_noc_end_x                 = get_arg_val<uint32_t>(4);
    const uint32_t mcast_dest_noc_end_y                 = get_arg_val<uint32_t>(5);
    const uint32_t num_mcast_cores_mid_group            = get_arg_val<uint32_t>(6);

    // first mcast group
    uint32_t mcast_first_group_dest_noc_start_x;
    uint32_t mcast_first_group_dest_noc_start_y;
    uint32_t mcast_first_group_dest_noc_end_x;
    uint32_t mcast_first_group_dest_noc_end_y;
    // last mcast group
    uint32_t mcast_last_group_dest_noc_start_x;
    uint32_t mcast_last_group_dest_noc_start_y;
    uint32_t mcast_last_group_dest_noc_end_x;
    uint32_t mcast_last_group_dest_noc_end_y;
    // volatile tt_l1_ptr uint32_t * noc_coord;
    volatile tt_l1_ptr uint32_t * noc_coord_x;
    volatile tt_l1_ptr uint32_t * noc_coord_y;

    // number of cores in mcast groups
    uint32_t num_mcast_cores_first_group;
    uint32_t num_mcast_cores_last_group;

    // noc addrs for first and last groups
    uint64_t reduce_sender_first_group_semaphore_noc_addr;
    uint64_t multicast_first_group_data_noc;
    uint64_t reduce_sender_last_group_semaphore_noc_addr;
    uint64_t multicast_last_group_data_noc;

    if (has_mcast_first_group and has_mcast_last_group) {

        mcast_first_group_dest_noc_start_x               = get_arg_val<uint32_t>(7);
        mcast_first_group_dest_noc_start_y               = get_arg_val<uint32_t>(8);
        mcast_first_group_dest_noc_end_x                 = get_arg_val<uint32_t>(9);
        mcast_first_group_dest_noc_end_y                 = get_arg_val<uint32_t>(10);
        num_mcast_cores_first_group                      = get_arg_val<uint32_t>(11);

        mcast_last_group_dest_noc_start_x               = get_arg_val<uint32_t>(12);
        mcast_last_group_dest_noc_start_y               = get_arg_val<uint32_t>(13);
        mcast_last_group_dest_noc_end_x                 = get_arg_val<uint32_t>(14);
        mcast_last_group_dest_noc_end_y                 = get_arg_val<uint32_t>(15);
        num_mcast_cores_last_group                      = get_arg_val<uint32_t>(16);

        noc_coord_x             = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(17));
        noc_coord_y             = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(17+num_mcast_cores));

    } else if (has_mcast_first_group and not has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x               = get_arg_val<uint32_t>(7);
        mcast_first_group_dest_noc_start_y               = get_arg_val<uint32_t>(8);
        mcast_first_group_dest_noc_end_x                 = get_arg_val<uint32_t>(9);
        mcast_first_group_dest_noc_end_y                 = get_arg_val<uint32_t>(10);
        num_mcast_cores_first_group                      = get_arg_val<uint32_t>(11);

        noc_coord_x             = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(12));
        noc_coord_y             = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(12+num_mcast_cores));

    } else if (not has_mcast_first_group and has_mcast_last_group) {
        mcast_last_group_dest_noc_start_x               = get_arg_val<uint32_t>(7);
        mcast_last_group_dest_noc_start_y               = get_arg_val<uint32_t>(8);
        mcast_last_group_dest_noc_end_x                 = get_arg_val<uint32_t>(9);
        mcast_last_group_dest_noc_end_y                 = get_arg_val<uint32_t>(10);
        num_mcast_cores_last_group                      = get_arg_val<uint32_t>(11);

        noc_coord_x             = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(12));
        noc_coord_y             = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(12+num_mcast_cores));

    } else {
        noc_coord_x             = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(7));
        noc_coord_y             = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(7+num_mcast_cores));
    }

    const uint64_t reduce_sender_semaphore_noc_addr = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        reduce_sender_semaphore_addr);

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        0);

    if (has_mcast_first_group) {
        reduce_sender_first_group_semaphore_noc_addr = get_noc_multicast_addr(
            mcast_first_group_dest_noc_start_x,
            mcast_first_group_dest_noc_start_y,
            mcast_first_group_dest_noc_end_x,
            mcast_first_group_dest_noc_end_y,
            reduce_sender_semaphore_addr);

        multicast_first_group_data_noc = get_noc_multicast_addr(
            mcast_first_group_dest_noc_start_x,
            mcast_first_group_dest_noc_start_y,
            mcast_first_group_dest_noc_end_x,
            mcast_first_group_dest_noc_end_y,
            0);
    }
    if (has_mcast_last_group) {
        reduce_sender_last_group_semaphore_noc_addr = get_noc_multicast_addr(
            mcast_last_group_dest_noc_start_x,
            mcast_last_group_dest_noc_start_y,
            mcast_last_group_dest_noc_end_x,
            mcast_last_group_dest_noc_end_y,
            reduce_sender_semaphore_addr);

        multicast_last_group_data_noc = get_noc_multicast_addr(
            mcast_last_group_dest_noc_start_x,
            mcast_last_group_dest_noc_start_y,
            mcast_last_group_dest_noc_end_x,
            mcast_last_group_dest_noc_end_y,
            0);
    }

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    *reduce_sender_semaphore_addr_ptr = VALID;
    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);

    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0;
    constexpr uint32_t cb_ex = tt::CB::dataflow1;
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2;
    constexpr uint32_t cb_in0 = tt::CB::c_in0; // sharded cb
    constexpr uint32_t cb_in = tt::CB::c_in7; // for pick values

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial);
    const DataFormat data_format = get_dataformat(cb_ex_partial);
    // const uint32_t num_bytes_read = datum_size_bytes;
    const uint32_t num_bytes_read = single_tile_size_bytes;

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
                        write_l1_index = 0;
                        for (uint32_t t=0; t < num_nz_rows_per_tile; ++t) {
                            for (uint32_t c=num_cols_per_group; c < 32; ++c) {
                                wptr[c + write_l1_index] = 0;
                            }
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


            if constexpr(num_mcast_cores > 1) {
                for (uint32_t i=0; i < 2; ++i) {
                    // wait for local data ready
                    cb_wait_front(cb_ex_partial, 1);
                    uint32_t l1_read_addr_ex_par = get_read_ptr(cb_ex_partial);
                    {
                        uint32_t l1_write_addr_external = get_write_ptr(cb_ex_external);
                        cb_reserve_back(cb_ex_external, 1);
                        uint64_t noc_addr_ex_par = get_noc_addr(noc_coord_x[0], noc_coord_y[0], l1_read_addr_ex_par);
                        noc_async_read_one_packet(noc_addr_ex_par, l1_write_addr_external, num_bytes_read);
                        noc_async_read_barrier();
                        cb_push_back(cb_ex_external, 1);
                    }

                    // wait for all other cores data ready
                    noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_mcast_cores-1);
                    noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);

                    // read data from other cores
                    for(uint32_t i = 0; i < num_mcast_cores - 1; ++i) {
                        uint32_t l1_write_addr_external = get_write_ptr(cb_ex_external);
                        cb_reserve_back(cb_ex_external, 1);
                        uint64_t noc_addr_ex_par = get_noc_addr(noc_coord_x[i + 1], noc_coord_y[i + 1], l1_read_addr_ex_par);
                        noc_async_read_one_packet(noc_addr_ex_par, l1_write_addr_external, num_bytes_read);
                        noc_async_read_barrier();
                        cb_push_back(cb_ex_external, 1);
                    }

                    // wait for global reduce done
                    cb_wait_front(cb_ex, 1);
                    cb_pop_front(cb_ex_partial, 1);

                    // mcast to other cores
                    uint32_t l1_read_addr_ex = get_read_ptr(cb_ex);
                    noc_async_write_multicast(l1_read_addr_ex, multicast_data_noc | l1_read_addr_ex, num_bytes_read, num_mcast_cores_mid_group, true);
                    noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_mcast_cores_mid_group, false);

                    if (has_mcast_first_group) {
                        noc_async_write_multicast(l1_read_addr_ex, multicast_first_group_data_noc | l1_read_addr_ex, num_bytes_read, num_mcast_cores_first_group, true);
                        noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_first_group_semaphore_noc_addr, num_mcast_cores_first_group, false);
                    }

                    if (has_mcast_last_group) {
                        noc_async_write_multicast(l1_read_addr_ex, multicast_last_group_data_noc | l1_read_addr_ex, num_bytes_read, num_mcast_cores_last_group, true);
                        noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_last_group_semaphore_noc_addr, num_mcast_cores_last_group, false);
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_ex, 1);
                }
            }
            group_index += group_offset;
        }
        batch_index += batch_offset;
    }
}
