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
    constexpr uint32_t num_group_batch                = get_compile_time_arg_val(3);

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

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial);
    const DataFormat data_format = get_dataformat(cb_ex_partial);

    if constexpr(num_mcast_cores > 1) {
        for (uint32_t i=0; i < 2; ++i) {
            // wait for local data ready
            cb_wait_front(cb_ex_partial, num_group_batch);

            // wait for all other cores data ready
            noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_mcast_cores-1);
            noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);

            // read data from other cores
            uint32_t l1_read_addr_ex_par = get_read_ptr(cb_ex_partial);
            for (uint32_t bg = 0; bg < num_group_batch; ++bg) {
                for(uint32_t i = 0; i < num_mcast_cores; ++i) {
                    uint32_t l1_write_addr_external = get_write_ptr(cb_ex_external);
                    cb_reserve_back(cb_ex_external, 1);
                    uint64_t noc_addr_ex_par = get_noc_addr(noc_coord_x[i], noc_coord_y[i], l1_read_addr_ex_par);
                    noc_async_read_one_packet(noc_addr_ex_par, l1_write_addr_external, single_tile_size_bytes);
                    noc_async_read_barrier();
                    cb_push_back(cb_ex_external, 1);
                }
                l1_read_addr_ex_par += single_tile_size_bytes;
            }

            // wait for global reduce done
            cb_wait_front(cb_ex, num_group_batch);
            cb_pop_front(cb_ex_partial, num_group_batch);

            // mcast to other cores
            uint32_t l1_read_addr_ex = get_read_ptr(cb_ex);
            noc_async_write_multicast(l1_read_addr_ex, multicast_data_noc | l1_read_addr_ex, num_group_batch * single_tile_size_bytes, num_mcast_cores_mid_group, true);
            noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_mcast_cores_mid_group, false);

            if (has_mcast_first_group) {
                noc_async_write_multicast(l1_read_addr_ex, multicast_first_group_data_noc | l1_read_addr_ex, num_group_batch * single_tile_size_bytes, num_mcast_cores_first_group, true);
                noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_first_group_semaphore_noc_addr, num_mcast_cores_first_group, false);
            }

            if (has_mcast_last_group) {
                noc_async_write_multicast(l1_read_addr_ex, multicast_last_group_data_noc | l1_read_addr_ex, num_group_batch * single_tile_size_bytes, num_mcast_cores_last_group, true);
                noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_last_group_semaphore_noc_addr, num_mcast_cores_last_group, false);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_ex, num_group_batch);

        }
    }
}
