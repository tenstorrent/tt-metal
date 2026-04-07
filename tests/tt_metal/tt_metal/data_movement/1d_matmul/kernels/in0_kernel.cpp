#pragma once

#include "test_config.h"
#include "dataflow_api.h"
#include "risc_common.h"

using namespace TestConfig;

void in0_sender_receiver_run(
    const uint32_t origin_x_coord,
    const uint32_t origin_y_coord,
    const uint32_t phy_x_coord,
    const uint32_t phy_y_coord,
    const uint32_t start_x,
    const uint32_t start_y,
    const uint32_t end_x,
    const uint32_t end_y,
    const uint32_t mhartid) {
    uint64_t sender_id = (phy_y_coord - origin_y_coord) * (end_x - start_x + 1) + (phy_x_coord - origin_x_coord);
    /* Semaphore setup */
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile uint32_t* in0_mcast_receiver_semaphore_addr_ptr = (uint32_t*)(in0_mcast_receiver_semaphore_addr);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready  to
    // receive the mcast
    volatile uint32_t* in0_mcast_sender_semaphore_addr_ptr = (uint32_t*)(in0_mcast_sender_semaphore_addr);
    // Semaphore with valid value, used for multicasting
    volatile uint32_t* in0_mcast_sender_semaphore_valid_addr_ptr = (uint32_t*)(in0_mcast_sender_semaphore_valid_addr);
    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    in0_mcast_sender_semaphore_valid_addr_ptr[0] =
        1;  // Load const 1 to be used as semaphore valid value sent from sender to receivers
    in0_mcast_sender_semaphore_addr_ptr[0] = 0;

    /* Pre calculate receiver semaphore multicast address*/
    const uint64_t in0_multicast_data_noc = noc_index == 0 ? get_noc_multicast_addr(start_x, start_y, end_x, end_y, 0)
                                                           : get_noc_multicast_addr(end_x, end_y, start_x, start_y, 0);
    uint64_t in0_mcast_receiver_semaphore_noc_addr =
        in0_multicast_data_noc | (uint64_t)in0_mcast_receiver_semaphore_addr;
    /* Semaphore setup end */
    uint64_t remote_sender_noc_addrs[num_remote_senders];

    for (uint32_t i = 0; i < num_remote_senders; i++) {
        uint32_t cur_x = i % (end_x - start_x + 1);
        uint32_t cur_y = i / (end_x - start_x + 1);
        remote_sender_noc_addrs[i] =
            get_noc_addr(origin_x_coord + cur_x, origin_y_coord + cur_y, in0_mcast_sender_semaphore_addr);
    }

    noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, 1);

    DeviceTimestampedData("Number of transactions", num_subblocks_k_dim);
    DeviceTimestampedData("Transaction size in bytes", in0_column_block_size_bytes);

    /* Reserve output circular buffer */
    // CB::cb_reserve_back(cb_id_in5, in0_block_num_tiles);
    uint32_t in0_tensor_current_tensix_cluster_l1_input_addr = cb5_write_addr;  // CB::get_write_ptr(0, cb_id_in5);
    // Main loop for the blocks
    uint32_t current_t6 = 0;
    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t subblock_k_dim_index = 0; subblock_k_dim_index < num_subblocks_k_dim; subblock_k_dim_index++) {
            noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, 0);

            uint32_t block_id = subblock_k_dim_index / num_subblocks_k_dim_per_tensix_cluster;

            /* in the sending block mode*/
            if (block_id == sender_id) {
                uint32_t in0_tensor_local_l1_write_addr = 0x80000;

                noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests - 1);
                noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

                uint64_t in0_multicast_data_addr =
                    in0_multicast_data_noc | in0_tensor_local_l1_write_addr;  // This is address where we multicast data
                noc_async_write_multicast(
                    in0_tensor_current_tensix_cluster_l1_input_addr,
                    in0_multicast_data_addr,
                    in0_column_block_size_bytes,
                    num_of_dests);
                // This needs snoop bit enabled
                noc_semaphore_set_multicast_loopback_src(
                    in0_mcast_sender_semaphore_valid_addr, in0_mcast_receiver_semaphore_noc_addr, num_of_dests);
                // in0_tensor_current_tensix_cluster_l1_input_addr += in0_column_block_size_bytes;
            } else /* In the receiving block mode*/
            {
                uint64_t in0_mcast_sender_semaphore_noc_addr = remote_sender_noc_addrs[block_id];
                // Atomic increment source core counter
                // Snoop bit enabled
                noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);
            }
            noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, 1);
        }
    }
}
