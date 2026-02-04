// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_udm_utils.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include <type_traits>

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);
constexpr uint32_t notification_mailbox_address = get_compile_time_arg_val(2);
constexpr uint32_t target_address_base = get_compile_time_arg_val(3);
constexpr NocSendType noc_send_type = static_cast<NocSendType>(get_compile_time_arg_val(4));
constexpr uint16_t packet_payload_size_bytes = static_cast<uint16_t>(get_compile_time_arg_val(5));
constexpr uint32_t num_packets = get_compile_time_arg_val(6);
constexpr uint32_t time_seed_init = get_compile_time_arg_val(7);
constexpr uint32_t req_notification_size_bytes = get_compile_time_arg_val(8);
// Total number of devices in the mesh
constexpr uint32_t num_devices = get_compile_time_arg_val(9);
// Per-reader L1 region size
constexpr uint32_t per_reader_l1_size = get_compile_time_arg_val(10);
// This data provider's device index - skip this slot (no one reads from self)
constexpr uint32_t provider_device_idx = get_compile_time_arg_val(11);

/*
 * All-to-all read receiver (data provider) kernel.
 * Fills L1 with data for all readers (N-1 readers).
 * Uses simple indexing: slot i contains data for reader device i.
 * Skips slot = provider_device_idx (no one reads from self).
 *
 * Runtime args: for each reader: (noc_x, noc_y, dst_dev_id, dst_mesh_id)
 */
void kernel_main() {
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t bytes_filled = 0;

    // Fill data for each reader using simple indexing
    // Slot i contains data for reader device i
    for (uint32_t reader_idx = 0; reader_idx < num_devices; reader_idx++) {
        if (reader_idx == provider_device_idx) {
            continue;  // Skip self - no one reads from their own device
        }

        // Calculate L1 address for this reader's data
        uint32_t target_address = target_address_base + reader_idx * per_reader_l1_size;

        // Use time_seed offset by reader_idx for unique data per reader
        uint32_t time_seed = time_seed_init + reader_idx;

        // Fill all packets for this reader
        for (uint32_t i = 0; i < num_packets; i++) {
            time_seed = prng_next(time_seed);

            uint32_t curr_local_data_addr = target_address + (i * packet_payload_size_bytes);
            tt_l1_ptr uint32_t* buffer_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(curr_local_data_addr);
            fill_packet_data(buffer_addr, packet_payload_size_bytes / 16, time_seed);

            bytes_filled += packet_payload_size_bytes;
        }
    }

    // Notify all readers that data is ready
    // Each provider writes to slot = provider_device_idx on all readers
    // This way each reader can poll on different slots for each provider
    // Runtime args: for each reader: (noc_x, noc_y, dst_dev_id, dst_mesh_id)
    uint32_t arg_index = 0;
    uint32_t num_readers = num_devices - 1;  // All devices except self
    for (uint32_t reader = 0; reader < num_readers; reader++) {
        uint32_t noc_x = get_arg_val<uint32_t>(arg_index++);
        uint32_t noc_y = get_arg_val<uint32_t>(arg_index++);
        uint32_t dst_dev_id = get_arg_val<uint32_t>(arg_index++);
        uint32_t dst_mesh_id = get_arg_val<uint32_t>(arg_index++);

        uint32_t local_notification_buffer_addr = notification_mailbox_address;
        // Send notification to slot = provider_device_idx on the reader
        // Reader will poll on this slot to know data from this provider is ready
        uint32_t remote_notification_dest_addr =
            notification_mailbox_address + provider_device_idx * req_notification_size_bytes;

        notify_receiver(
            dst_dev_id,
            dst_mesh_id,
            noc_x,
            noc_y,
            local_notification_buffer_addr,
            remote_notification_dest_addr,
            time_seed_init,
            req_notification_size_bytes);
    }

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)bytes_filled;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = bytes_filled >> 32;
}
