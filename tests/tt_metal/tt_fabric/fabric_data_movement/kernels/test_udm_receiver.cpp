// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm.hpp"
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm_impl.hpp"
#include "debug/dprint.h"
#include <type_traits>

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);
constexpr uint32_t notification_mailbox_address = get_compile_time_arg_val(2);
uint32_t target_address = get_compile_time_arg_val(3);
constexpr NocSendType noc_send_type = static_cast<NocSendType>(get_compile_time_arg_val(4));
constexpr uint16_t packet_payload_size_bytes = static_cast<uint16_t>(get_compile_time_arg_val(5));
constexpr uint32_t num_packets = get_compile_time_arg_val(6);
constexpr uint32_t time_seed_init = get_compile_time_arg_val(7);

// Helper function to print UDM control fields from received packet header
inline void print_udm_control_fields(volatile tt_l1_ptr uint32_t* packet_start_addr, uint32_t packet_index) {
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_start_addr);

    DPRINT << "UDM Control Fields (Packet " << packet_index << "):\n";
    DPRINT << "  src_chip_id: " << (uint32_t)header->udm_control.write.src_chip_id << "\n";
    DPRINT << "  src_mesh_id: " << (uint32_t)header->udm_control.write.src_mesh_id << "\n";
    DPRINT << "  src_noc_x: " << (uint32_t)header->udm_control.write.src_noc_x << "\n";
    DPRINT << "  src_noc_y: " << (uint32_t)header->udm_control.write.src_noc_y << "\n";
    DPRINT << "  risc_id: " << (uint32_t)header->udm_control.write.risc_id << "\n";
    DPRINT << "  transaction_id: " << (uint32_t)header->udm_control.write.transaction_id << "\n";
    DPRINT << "  posted: " << (uint32_t)header->udm_control.write.posted << "\n";
}

// Function to check packet data while skipping header bytes
inline bool check_packet_data_skip_header(
    tt_l1_ptr uint32_t* packet_start_addr,
    uint32_t payload_size_bytes,
    uint32_t time_seed,
    uint32_t& mismatch_addr,
    uint32_t& mismatch_val,
    uint32_t& expected_val) {
    // Skip the packet header bytes that were filled with header data
    constexpr uint32_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    // Calculate how many 16-byte words the header occupies
    constexpr uint32_t header_words = packet_header_size_bytes / PACKET_WORD_SIZE_BYTES;

    // Start checking from after the header bytes
    tt_l1_ptr uint32_t* check_start_addr = packet_start_addr + (packet_header_size_bytes / sizeof(uint32_t));

    // Adjust the number of words to check (exclude header words)
    uint32_t num_words_to_check = (payload_size_bytes / PACKET_WORD_SIZE_BYTES) - header_words;

    // Adjust the starting seed value to skip the header words
    uint32_t adjusted_time_seed = time_seed;
    for (uint32_t j = 0; j < header_words; j++) {
        adjusted_time_seed++;
    }

    return check_packet_data(
        check_start_addr, num_words_to_check, adjusted_time_seed, mismatch_addr, mismatch_val, expected_val);
}

void kernel_main() {
    uint32_t time_seed = time_seed_init;

    int32_t dest_bank_id;
    uint32_t dest_dram_addr;

    tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(target_address);
    volatile tt_l1_ptr uint32_t* poll_addr =
        reinterpret_cast<tt_l1_ptr uint32_t*>(target_address + packet_payload_size_bytes - 4);
    uint32_t mismatch_addr, mismatch_val, expected_val;
    bool match = true;
    uint64_t bytes_received = 0;

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    // poll for data
    for (uint32_t i = 0; i < num_packets; i++) {
        time_seed = prng_next(time_seed);
        uint32_t expected_val = time_seed + (packet_payload_size_bytes / 16) - 1;
        if constexpr (noc_send_type == NOC_UNICAST_INLINE_WRITE) {
            expected_val = 0xDEADBEEF;
        }

        WAYPOINT("FPW");
        while (expected_val != *poll_addr) {
            invalidate_l1_cache();
        }
        WAYPOINT("FPD");

        // Send write ACK back to the sender
        // The function will use the risc_id from the packet header as the dm_id
        volatile tt_l1_ptr PACKET_HEADER_TYPE* received_header =
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(start_addr);
        tt::tt_fabric::udm::fabric_fast_write_ack(received_header);

        // check for data correctness, skipping the header bytes
        match = check_packet_data_skip_header(
            start_addr, packet_payload_size_bytes, time_seed, mismatch_addr, mismatch_val, expected_val);
        if (!match) {
            break;
        }
        start_addr += packet_payload_size_bytes / 4;
        poll_addr += packet_payload_size_bytes / 4;
        bytes_received += packet_payload_size_bytes;
    }

    if (!match) {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_DATA_MISMATCH;
        test_results[TT_FABRIC_MISC_INDEX + 12] = mismatch_addr;
        test_results[TT_FABRIC_MISC_INDEX + 13] = mismatch_val;
        test_results[TT_FABRIC_MISC_INDEX + 14] = expected_val;
    } else {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    }

    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)bytes_received;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = bytes_received >> 32;
}
