// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm.hpp"
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm_impl.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "dataflow_api.h"
#include "debug/dprint.h"

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);
constexpr uint32_t notification_mailbox_address = get_compile_time_arg_val(2);
uint32_t target_address = get_compile_time_arg_val(3);
constexpr NocSendType noc_send_type = static_cast<NocSendType>(get_compile_time_arg_val(4));
constexpr uint32_t source_l1_buffer_address = get_compile_time_arg_val(5);
constexpr uint16_t packet_payload_size_bytes = static_cast<uint16_t>(get_compile_time_arg_val(6));
constexpr uint32_t num_packets = get_compile_time_arg_val(7);
constexpr uint32_t time_seed_init = get_compile_time_arg_val(8);
constexpr uint32_t noc_x_start = get_compile_time_arg_val(9);
constexpr uint32_t noc_y_start = get_compile_time_arg_val(10);
constexpr uint32_t dst_dev_id = get_compile_time_arg_val(11);
constexpr uint32_t dst_mesh_id = get_compile_time_arg_val(12);

// Function to fill packet payload with header info at the beginning
inline void fill_payload_with_header_and_data(
    uint32_t source_buffer_addr,
    uint32_t payload_size_bytes,
    uint32_t time_seed,
    uint32_t target_addr,
    uint16_t dst_device_id,
    uint16_t dst_mesh_id,
    uint32_t noc_x,
    uint32_t noc_y) {
    tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(source_buffer_addr);

    // First, fill the packet payload with the expected data pattern
    fill_packet_data(start_addr, payload_size_bytes / 16, time_seed);

    // Now we need to fill the top of the payload with packet header info
    // The packet header will be created by fabric_fast_write, but we need to
    // simulate what it will contain and copy that to the beginning of our payload

    // Get the packet header that will be used
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = tt::tt_fabric::udm::get_or_allocate_header();

    // Configure the header (this is what fabric_fast_write will do internally)
    packet_header->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(noc_x, noc_y, target_addr)}, payload_size_bytes);
    tt::tt_fabric::udm::fabric_write_set_unicast_route(
        packet_header, dst_device_id, dst_mesh_id, 0, 0);  // trid=0, posted=0

    // Copy the header data to the beginning of our payload
    // Since both source and destination are in local L1, we can use direct memory copy
    uint32_t header_size_bytes = sizeof(PACKET_HEADER_TYPE);
    uint32_t header_size_words = header_size_bytes / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* dest_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source_buffer_addr);
    volatile tt_l1_ptr uint32_t* src_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(packet_header);

    for (uint32_t i = 0; i < header_size_words; i++) {
        dest_ptr[i] = src_ptr[i];
    }
}

void kernel_main() {
    // TODO: move this into fw once consolidated
    tt::tt_fabric::udm::fabric_local_state_init();

    // Runtime args are set up by append_fabric_connection_rt_args and used internally by fabric_fast_write
    // We don't need to read them explicitly here
    uint32_t time_seed = time_seed_init;

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();

    for (uint32_t i = 0; i < num_packets; i++) {
        time_seed = prng_next(time_seed);

        // Fill payload with header at the beginning and data after
        fill_payload_with_header_and_data(
            source_l1_buffer_address,
            packet_payload_size_bytes,
            time_seed,
            target_address,
            dst_dev_id,
            dst_mesh_id,
            noc_x_start,
            noc_y_start);

        switch (noc_send_type) {
            case NOC_UNICAST_WRITE: {
                tt::tt_fabric::udm::fabric_fast_write_any_len(
                    dst_dev_id,
                    dst_mesh_id,
                    source_l1_buffer_address,
                    get_noc_addr(noc_x_start, noc_y_start, target_address),
                    packet_payload_size_bytes);

                tt::tt_fabric::udm::fabric_write_barrier();
            } break;
            default: {
                ASSERT(false);
            } break;
        }
        noc_async_writes_flushed();
        target_address += packet_payload_size_bytes;
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    noc_async_write_barrier();

    uint64_t bytes_sent = packet_payload_size_bytes * num_packets;

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)bytes_sent;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = bytes_sent >> 32;
}
