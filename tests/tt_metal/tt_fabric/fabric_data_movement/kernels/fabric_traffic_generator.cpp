// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"

// Compile-time args (in order):
// 0: source_buffer_address (uint32_t)
// 1: packet_payload_size_bytes (uint32_t)
// 2: teardown_signal_address (uint32_t)
// 3: virt_noc_x (uint32_t) - virtual NOC x coordinate of target
// 4: virt_noc_y (uint32_t) - virtual NOC y coordinate of target
// 5: remote_buffer_addr (uint32_t) - address of buffer on remote chip

// Runtime args (in order):
// 0: dest_chip_id (uint32_t)
// 1: dest_mesh_id (uint32_t)
// 2+: fabric connection args (appended by append_fabric_connection_rt_args)

// Constants for teardown protocol
constexpr uint32_t WORKER_KEEP_RUNNING = 0;
constexpr uint32_t WORKER_TEARDOWN = 1;

void kernel_main() {
    // Extract compile-time args
    [[maybe_unused]] constexpr uint32_t source_buffer_address = get_compile_time_arg_val(0);
    constexpr uint32_t packet_payload_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t teardown_signal_address = get_compile_time_arg_val(2);
    constexpr uint32_t virt_noc_x = get_compile_time_arg_val(3);
    constexpr uint32_t virt_noc_y = get_compile_time_arg_val(4);
    constexpr uint32_t remote_buffer_addr = get_compile_time_arg_val(5);

    // Extract runtime args
    size_t arg_idx = 0;
    uint32_t dest_chip_id = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] uint32_t dest_mesh_id = get_arg_val<uint32_t>(arg_idx++);

    // Build individual fabric connection from runtime args and open it
    auto fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    fabric_connection.open();

    // Initialize packet header from pool
    auto* packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
        PacketHeaderPool::allocate_header());

    // Point to teardown mailbox
    volatile uint32_t* teardown_signal =
        reinterpret_cast<volatile uint32_t*>(teardown_signal_address);
    *teardown_signal = 0;

    // Compute destination NOC address once
    uint64_t noc_addr = get_noc_addr(virt_noc_x, virt_noc_y, remote_buffer_addr);

    // Total size to send (header + payload)
    constexpr size_t total_send_size = packet_payload_size_bytes;

    // Main traffic generation loop
    while (*teardown_signal == WORKER_KEEP_RUNNING) {
        // Build packet header with routing info
        // Set chip routing (unicast to destination chip)
        packet_header->to_chip_unicast(1);
        // Set NOC write command
        packet_header->to_noc_unicast_write(
            tt::tt_fabric::NocUnicastCommandHeader{noc_addr},
            packet_payload_size_bytes);

        // Send via fabric connection
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_non_blocking_from_address(
            reinterpret_cast<uint32_t>(packet_header),
            total_send_size);
    }
    DPRINT << "TEARDOWN\n";

    // Graceful shutdown - close fabric connection
    fabric_connection.close();
}
