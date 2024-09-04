// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

/**
 * A blocking call that waits for remote receiver to acknowledge that all data sent with eth_send_bytes since the last
 * reset_erisc_info call is no longer being used. Also, see \a eth_receiver_done().
 * This also syncs with the local receiver core using a semaphore and reads date from the receiver core
 *
 * Return value: None
 *
 * | Argument                    | Description                                               | Type     | Valid Range                                                   | Required |
 * |-----------------------------|-----------------------------------------------------------|----------|---------------------------------------------------------------|----------|
 * | sender_semaphore_addr_ptr   | Semaphore address in local L1 memory                      | uint32_t | 0..1MB                                                        | True     |
 * | receiver_semaphore_noc_addr | Encoding of the receiver semaphore location (x,y)+address | uint64_t | DOX-TODO(insert a reference to what constitutes valid coords) | True     |
 * | receiver_data_noc_addr      | Encoding of the receiver source location (x,y)+address    | uint64_t | DOX-TODO(ref to explain valid coords)                         | True     |
 * | dst_local_l1_addr           | Address in local L1 memory                                | uint32_t | 0..1MB                                                        | True     |
 * | size                        | Size of data transfer in bytes                            | uint32_t | 0..1MB                                                        | True     |
 */
template<bool write_barrier = false>
FORCE_INLINE
void eth_wait_for_remote_receiver_done_and_get_local_receiver_data(
    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr,
    uint64_t receiver_semaphore_noc_addr,
    uint64_t receiver_data_noc_addr,
    uint32_t local_eth_l1_curr_src_addr,
    uint32_t size
) {
    internal_::eth_send_packet(
        0,
        ((uint32_t)(&(erisc_info->channels[0].bytes_sent))) >> 4,
        ((uint32_t)(&(erisc_info->channels[0].bytes_sent))) >> 4,
        1);
    eth_noc_semaphore_wait(sender_semaphore_addr_ptr, 1);
    noc_async_read(receiver_data_noc_addr, local_eth_l1_curr_src_addr, size);
    noc_semaphore_set(sender_semaphore_addr_ptr, 0);
    eth_noc_async_read_barrier();
    if constexpr (write_barrier) {
        eth_noc_async_write_barrier();
    }
    noc_semaphore_inc(receiver_semaphore_noc_addr, 1);
    while (erisc_info->channels[0].bytes_sent != 0) {
        internal_::risc_context_switch();
    }
}
