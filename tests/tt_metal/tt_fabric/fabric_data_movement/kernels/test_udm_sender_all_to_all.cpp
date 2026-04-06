// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_udm_utils.hpp"

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);
constexpr uint32_t notification_mailbox_address = get_compile_time_arg_val(2);
constexpr uint32_t target_address_base = get_compile_time_arg_val(3);
constexpr NocSendType noc_send_type = static_cast<NocSendType>(get_compile_time_arg_val(4));
constexpr uint32_t source_l1_buffer_address = get_compile_time_arg_val(5);
constexpr uint16_t packet_payload_size_bytes = static_cast<uint16_t>(get_compile_time_arg_val(6));
constexpr uint32_t num_packets = get_compile_time_arg_val(7);
constexpr uint32_t time_seed_init = get_compile_time_arg_val(8);
constexpr uint32_t req_notification_size_bytes = get_compile_time_arg_val(9);
// Per-sender L1 region size on receiver
constexpr uint32_t per_sender_l1_size = get_compile_time_arg_val(10);
// Number of destinations (N-1 for all-to-all)
constexpr uint32_t num_destinations = get_compile_time_arg_val(11);
// This sender's device index - used directly as L1 slot index on receivers
constexpr uint32_t sender_device_idx = get_compile_time_arg_val(12);

/*
 * All-to-all sender kernel.
 * Sends packets to all other devices.
 * Uses simple indexing: sender device N writes to slot N on all receivers.
 *
 * Runtime args: for each destination: (noc_x, noc_y, dst_dev_id, dst_mesh_id)
 */
void kernel_main() {
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();
    uint64_t total_bytes_sent = 0;

    // This sender always writes to slot = sender_device_idx on receivers
    uint32_t target_address = target_address_base + sender_device_idx * per_sender_l1_size;

    // Use time_seed offset by sender_device_idx for unique data per sender
    uint32_t time_seed_base = time_seed_init + sender_device_idx;

    // Runtime args: for each destination: (noc_x, noc_y, dst_dev_id, dst_mesh_id)
    uint32_t arg_index = 0;

    // Loop over all destinations
    for (uint32_t dest = 0; dest < num_destinations; dest++) {
        uint32_t noc_x_start = get_arg_val<uint32_t>(arg_index++);
        uint32_t noc_y_start = get_arg_val<uint32_t>(arg_index++);
        uint32_t dst_dev_id = get_arg_val<uint32_t>(arg_index++);
        uint32_t dst_mesh_id = get_arg_val<uint32_t>(arg_index++);

        uint32_t time_seed = time_seed_base;
        uint32_t current_target_address = target_address;

        for (uint32_t i = 0; i < num_packets; i++) {
            time_seed = prng_next(time_seed);

            tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address);
            fill_packet_data(start_addr, packet_payload_size_bytes / 16, time_seed);

            switch (noc_send_type) {
                case NOC_UNICAST_WRITE: {
                    tt::tt_fabric::udm::fabric_fast_write_any_len(
                        dst_dev_id,
                        dst_mesh_id,
                        source_l1_buffer_address,
                        get_noc_addr(noc_x_start, noc_y_start, current_target_address),
                        packet_payload_size_bytes);
                } break;
                case NOC_UNICAST_INLINE_WRITE: {
                    uint32_t inline_value = time_seed;
                    tt::tt_fabric::udm::fabric_fast_write_dw_inline(
                        dst_dev_id,
                        dst_mesh_id,
                        inline_value,
                        get_noc_addr(noc_x_start, noc_y_start, current_target_address));
                } break;
                case NOC_UNICAST_ATOMIC_INC: {
                    uint32_t incr_value = time_seed;
                    tt::tt_fabric::udm::fabric_fast_atomic_inc(
                        dst_dev_id,
                        dst_mesh_id,
                        incr_value,
                        get_noc_addr(noc_x_start, noc_y_start, current_target_address));
                } break;
                default: {
                    ASSERT(false);
                } break;
            }

            switch (noc_send_type) {
                case NOC_UNICAST_WRITE:
                case NOC_UNICAST_INLINE_WRITE: {
                    tt::tt_fabric::udm::fabric_write_barrier();
                } break;
                case NOC_UNICAST_ATOMIC_INC: {
                    tt::tt_fabric::udm::fabric_atomic_barrier();
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
            noc_async_writes_flushed();
            current_target_address += packet_payload_size_bytes;
        }

        total_bytes_sent += packet_payload_size_bytes * num_packets;
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    noc_async_write_barrier();

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)total_bytes_sent;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = total_bytes_sent >> 32;
}
