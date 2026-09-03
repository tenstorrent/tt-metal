// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#if defined(FABRIC_2D)
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#endif
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"

constexpr uint32_t test_results_address = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
constexpr uint32_t receiver_slots_base_address = get_compile_time_arg_val(2);
constexpr uint32_t sender_credit_handshake_address = get_compile_time_arg_val(3);
constexpr bool is_2d_fabric = get_compile_time_arg_val(4) != 0;
// CT args 5-8 are sender-only (eager staging / pattern / status trid); keep indices aligned.
constexpr bool kRandomizePayloadSizeAndDelay = get_compile_time_arg_val(9) != 0;

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_packets = get_arg_val<uint32_t>(arg_idx++);
    uint32_t seed = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t return_credits_per_packet = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t sender_noc_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t sender_noc_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t num_receiver_slots = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t num_hops = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t dst_device_id = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));

    auto sender = tt::tt_fabric::FabricMuxV2Sender<>::build_from_args(arg_idx);

    auto test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_address);
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    zero_l1_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(packet_header_buffer_address), sizeof(PACKET_HEADER_TYPE));

    sender.open();

    uint32_t mismatch_addr = 0;
    uint32_t mismatch_val = 0;
    uint32_t expected_val = 0;
    bool match = true;
    uint64_t bytes_received = 0;
    const uint32_t max_packet_payload_size_bytes = packet_payload_size_bytes;
    uint32_t receiver_slot_id = 0;
    uint32_t pending_credits = 0;

    const uint64_t sender_credit_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_credit_handshake_address);

    for (uint32_t packet_idx = 0; packet_idx < num_packets; ++packet_idx) {
        seed = prng_next(seed);
        uint32_t this_packet_payload_size_bytes = max_packet_payload_size_bytes;
        if constexpr (kRandomizePayloadSizeAndDelay) {
            this_packet_payload_size_bytes = derive_aligned_payload_size_bytes(seed, max_packet_payload_size_bytes);
        }
        expected_val = seed + (this_packet_payload_size_bytes / 16) - 1;

        // Slots are always strided by the configured max payload size.
        const uint32_t packet_offset_bytes = receiver_slot_id * max_packet_payload_size_bytes;
        auto poll_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            receiver_slots_base_address + packet_offset_bytes + this_packet_payload_size_bytes - sizeof(uint32_t));
        while (*poll_addr != expected_val) {
            invalidate_l1_cache();
        }

        auto packet_start = reinterpret_cast<tt_l1_ptr uint32_t*>(receiver_slots_base_address + packet_offset_bytes);
        match = check_packet_data(
            packet_start, this_packet_payload_size_bytes / 16, seed, mismatch_addr, mismatch_val, expected_val);
        if (!match) {
            break;
        }

        bytes_received += this_packet_payload_size_bytes;
        receiver_slot_id += 1;
        if (receiver_slot_id == num_receiver_slots) {
            receiver_slot_id = 0;
        }

        pending_credits += 1;
        if (pending_credits == return_credits_per_packet) {
            const auto credit_header =
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_credit_noc_addr, pending_credits};
#if defined(FABRIC_2D)
            if constexpr (is_2d_fabric) {
                tt::tt_fabric::mesh::experimental::fabric_unicast_noc_unicast_atomic_inc(
                    &sender, packet_header, dst_device_id, dst_mesh_id, credit_header);
            } else
#endif
            {
                tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
                    &sender, packet_header, credit_header, num_hops);
            }
            pending_credits = 0;
        }
    }

    if (pending_credits > 0) {
        const auto credit_header =
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_credit_noc_addr, pending_credits};
#if defined(FABRIC_2D)
        if constexpr (is_2d_fabric) {
            tt::tt_fabric::mesh::experimental::fabric_unicast_noc_unicast_atomic_inc(
                &sender, packet_header, dst_device_id, dst_mesh_id, credit_header);
        } else
#endif
        {
            tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
                &sender, packet_header, credit_header, num_hops);
        }
    }

    noc_async_write_barrier();
    sender.close();

    if (!match) {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_DATA_MISMATCH;
        test_results[TT_FABRIC_MISC_INDEX + 12] = mismatch_addr;
        test_results[TT_FABRIC_MISC_INDEX + 13] = mismatch_val;
        test_results[TT_FABRIC_MISC_INDEX + 14] = expected_val;
    } else {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    }

    test_results[TT_FABRIC_WORD_CNT_INDEX] = static_cast<uint32_t>(bytes_received);
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = static_cast<uint32_t>(bytes_received >> 32);
}
