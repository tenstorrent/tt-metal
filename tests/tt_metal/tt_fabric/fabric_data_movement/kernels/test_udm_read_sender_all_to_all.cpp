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
// Per-reader L1 region size on data provider
constexpr uint32_t per_reader_l1_size = get_compile_time_arg_val(10);
// Number of data providers to read from (N-1 for all-to-all)
constexpr uint32_t num_providers = get_compile_time_arg_val(11);
// This reader's device index - used directly as L1 slot index on data providers
constexpr uint32_t reader_device_idx = get_compile_time_arg_val(12);

/*
 * All-to-all read sender (reader) kernel.
 * Issues fabric read requests to all other devices.
 * Uses simple indexing: reads from slot = reader_device_idx on all data providers.
 *
 * Runtime args: for each provider: (noc_x, noc_y, dst_dev_id, dst_mesh_id)
 */
void kernel_main() {
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();
    uint64_t total_bytes_read = 0;
    bool match = true;
    uint32_t mismatch_addr, mismatch_val, expected_val;

    // This reader always reads from slot = reader_device_idx on data providers
    uint32_t remote_read_addr_base = target_address_base + reader_device_idx * per_reader_l1_size;

    // Use time_seed offset by reader_device_idx for expected data validation
    uint32_t time_seed_base = time_seed_init + reader_device_idx;

    // Runtime args: for each provider: (noc_x, noc_y, dst_dev_id, dst_mesh_id)
    uint32_t arg_index = 0;

    // Wait for notifications from all data providers
    // Each provider writes to slot = provider_device_idx on this reader
    // So we poll on all slots except our own (reader_device_idx)
    uint32_t num_devices = num_providers + 1;  // Total devices = providers + self
    for (uint32_t device_idx = 0; device_idx < num_devices; device_idx++) {
        if (device_idx == reader_device_idx) {
            continue;  // Skip our own slot - no one writes there
        }
        // Poll on slot for this device
        uint32_t notification_addr = notification_mailbox_address + device_idx * req_notification_size_bytes;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* received_header =
            wait_for_notification(notification_addr, time_seed_init, req_notification_size_bytes);
    }

    // Loop over all data providers
    for (uint32_t prov = 0; prov < num_providers; prov++) {
        uint32_t noc_x_start = get_arg_val<uint32_t>(arg_index++);
        uint32_t noc_y_start = get_arg_val<uint32_t>(arg_index++);
        uint32_t dst_dev_id = get_arg_val<uint32_t>(arg_index++);
        uint32_t dst_mesh_id = get_arg_val<uint32_t>(arg_index++);

        uint32_t remote_read_addr = remote_read_addr_base;
        uint32_t local_read_addr = source_l1_buffer_address;
        uint32_t time_seed = time_seed_base;

        for (uint32_t i = 0; i < num_packets; i++) {
            time_seed = prng_next(time_seed);

            switch (noc_send_type) {
                case NOC_UNICAST_READ: {
                    // Issue a read request to the remote device
                    tt::tt_fabric::udm::fabric_fast_read_any_len(
                        dst_dev_id,
                        dst_mesh_id,
                        get_noc_addr(noc_x_start, noc_y_start, remote_read_addr),
                        local_read_addr,
                        packet_payload_size_bytes);

                    // Wait for the read to complete
                    tt::tt_fabric::udm::fabric_read_barrier();

                    // Check the received data
                    match = check_packet_data(
                        reinterpret_cast<tt_l1_ptr uint32_t*>(local_read_addr),
                        packet_payload_size_bytes / 16,
                        time_seed,
                        mismatch_addr,
                        mismatch_val,
                        expected_val);

                    if (!match) {
                        DPRINT << "Data mismatch at provider " << prov << " packet " << i << "\n";
                        DPRINT << "  Mismatch addr: " << mismatch_addr << "\n";
                        DPRINT << "  Mismatch val: " << mismatch_val << "\n";
                        DPRINT << "  Expected val: " << expected_val << "\n";
                    }
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
            if (!match) {
                break;
            }
            noc_async_writes_flushed();
            remote_read_addr += packet_payload_size_bytes;
            local_read_addr += packet_payload_size_bytes;
        }

        total_bytes_read += packet_payload_size_bytes * num_packets;
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    noc_async_write_barrier();

    if (!match) {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_DATA_MISMATCH;
        test_results[TT_FABRIC_MISC_INDEX + 12] = mismatch_addr;
        test_results[TT_FABRIC_MISC_INDEX + 13] = mismatch_val;
        test_results[TT_FABRIC_MISC_INDEX + 14] = expected_val;
    } else {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    }

    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)total_bytes_read;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = total_bytes_read >> 32;
}
