// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp"
// Include appropriate fabric packet header APIs

// Compile-time args (in order):
// 0: source_buffer_address (uint32_t)
// 1: packet_payload_size_bytes (uint32_t)
// 2: target_noc_encoding (uint32_t) - encoded NOC address of receiver
// 3: teardown_signal_address (uint32_t)
// 4: is_2d_fabric (uint32_t) - 0 for 1D, 1 for 2D

// Runtime args (in order):
// 0: dest_chip_id (uint32_t)
// 1: dest_mesh_id (uint32_t)
// 2: random_seed (uint32_t)

// Constants from fabric_traffic_generator_defs.hpp
constexpr uint32_t WORKER_KEEP_RUNNING = 0;
constexpr uint32_t WORKER_TEARDOWN = 1;

void kernel_main() {
    // Extract compile-time args
    constexpr uint32_t source_buffer_address = get_compile_time_arg_val(0);
    constexpr uint32_t packet_payload_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t target_noc_encoding = get_compile_time_arg_val(2);
    constexpr uint32_t teardown_signal_address = get_compile_time_arg_val(3);
    constexpr uint32_t is_2d_fabric = get_compile_time_arg_val(4);

    // Extract runtime args
    uint32_t dest_chip_id = get_arg_val<uint32_t>(0);
    uint32_t dest_mesh_id = get_arg_val<uint32_t>(1);
    uint32_t random_seed = get_arg_val<uint32_t>(2);

    // Initialize packet header
    auto* packet_header = PacketHeaderPool::allocate_header();

    // Point to teardown mailbox
    volatile uint32_t* teardown_signal =
        reinterpret_cast<volatile uint32_t*>(teardown_signal_address);

    // Main traffic loop
    uint32_t packet_count = 0;
    while (*teardown_signal == WORKER_KEEP_RUNNING) {
        // Build packet header with routing info
        packet_header->to_chip_unicast(
            dest_chip_id,
            dest_mesh_id,
            target_noc_encoding,
            packet_payload_size_bytes,
            /*tag=*/packet_count);

        // Send packet via fabric
        fabric_send_packet(
            source_buffer_address,
            packet_payload_size_bytes,
            packet_header);

        packet_count++;

        // Optional: small delay to avoid saturating fabric
        // (can be tuned based on test requirements)
    }

    // Graceful shutdown - no cleanup needed
}
