// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm.hpp"
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm_impl.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// Function to notify receiver about incoming data (for both reads and writes)
// Used by sender to inform receiver about incoming packets
inline void notify_receiver(
    uint32_t dst_dev_id,
    uint32_t dst_mesh_id,
    uint32_t noc_x,
    uint32_t noc_y,
    uint32_t notification_buffer_addr,
    uint32_t remote_notification_addr,
    uint32_t time_seed,
    uint32_t req_notification_size_bytes) {
    tt_l1_ptr uint32_t* notification_buffer = reinterpret_cast<tt_l1_ptr uint32_t*>(notification_buffer_addr);

    // First, copy the packet header that contains source information
    // For reads: This header will be used by fabric_fast_read_any_len_ack on the receiver side
    // For writes: This header contains the sender information for ACK
    volatile tt_l1_ptr PACKET_HEADER_TYPE* allocated_header = tt::tt_fabric::udm::get_or_allocate_header();
    uint32_t header_size_words = sizeof(PACKET_HEADER_TYPE) / sizeof(uint32_t);
    for (uint32_t j = 0; j < header_size_words; j++) {
        notification_buffer[j] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(allocated_header)[j];
    }

    // set the last word for polling - receiver checks this location
    notification_buffer[req_notification_size_bytes / 4 - 1] = time_seed + req_notification_size_bytes / 4 - 1;

    // Send the notification to receiver's remote_notification_addr as a posted write
    tt::tt_fabric::udm::fabric_fast_write_any_len(
        dst_dev_id,
        dst_mesh_id,
        notification_buffer_addr,
        get_noc_addr(noc_x, noc_y, remote_notification_addr),
        req_notification_size_bytes,
        false,  // multicast
        1,      // num_dests
        0,      // trid
        1);     // posted = 1 (no ack needed)
}

// Function to wait for notification from sender (for both reads and writes)
// Used by receiver to poll for incoming packet notifications
inline volatile tt_l1_ptr PACKET_HEADER_TYPE* wait_for_notification(
    uint32_t notification_addr, uint32_t expected_seed, uint32_t req_notification_size_bytes) {
    // Cast the notification address
    volatile tt_l1_ptr uint32_t* current_notification_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(notification_addr);

    // Poll on the last word of the notification packet
    volatile tt_l1_ptr uint32_t* poll_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(notification_addr + req_notification_size_bytes - 4);

    // Expected value for the last word (must match sender's calculation)
    uint32_t expected_val = expected_seed + req_notification_size_bytes / 4 - 1;

    WAYPOINT("FPW");
    // Wait for notification to arrive
    while (*poll_addr != expected_val) {
        invalidate_l1_cache();
    }
    WAYPOINT("FPD");

    // Return the notification as a packet header
    // For reads: used by fabric_fast_read_any_len_ack
    // For writes: used by fabric_fast_write_ack
    return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(current_notification_addr);
}

// Helper function to print UDM control fields from received packet header (for debugging)
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
