// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "dataflow_api.h"

#include <cstdint>
#include <cstddef>

constexpr bool use_mcast_mode = get_compile_time_arg_val(0) != 0;

void kernel_main() {
    using namespace tt::tt_fabric;
    size_t arg_idx = 0;

    const size_t dest_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const size_t dest_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const size_t dest_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t message_num_hops = get_arg_val<uint32_t>(arg_idx++);
    auto teardown_signal_addr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    const size_t packet_header_cb = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_header_size_in_headers = get_arg_val<uint32_t>(arg_idx++);
    const bool is_downstream = get_arg_val<uint32_t>(arg_idx++) != 0;
    const size_t latency_writer_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const size_t latency_writer_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t latency_writer_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const size_t hops_to_latency_writer = get_arg_val<uint32_t>(arg_idx++);
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    ASSERT(dest_bank_addr != 0);
    const size_t source_l1_buffer_address = dest_bank_addr;

    ASSERT(fabric_connection.is_logically_connected());
    if (!fabric_connection.is_logically_connected()) {
        return;
    }

    fabric_connection.open();
    cb_reserve_back(packet_header_cb, packet_header_size_in_headers);
    const auto packet_header_buffer_address = get_write_ptr(packet_header_cb);

    auto* packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* ready_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));
    // Setup data packet header
    if constexpr (use_mcast_mode) {
        packet_header->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(message_num_hops)});
    } else {
        packet_header->to_chip_unicast(static_cast<uint8_t>(message_num_hops));
    }

    auto noc0_dest_addr =
        safe_get_noc_addr(static_cast<uint8_t>(dest_noc_x), static_cast<uint8_t>(dest_noc_y), dest_bank_addr, 0);
    packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc0_dest_addr}, packet_payload_size_bytes);

    // Setup ready signal packet header
    ready_packet_header->to_chip_unicast(static_cast<uint8_t>(hops_to_latency_writer));
    auto ready_sem_noc_addr =
        safe_get_noc_addr(latency_writer_noc_x, latency_writer_noc_y, latency_writer_ready_sem, 0);
    ready_packet_header->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader{ready_sem_noc_addr, 1, std::numeric_limits<uint16_t>::max()});

    // Send initial traffic burst before signaling ready
    static constexpr size_t INITIAL_BURST_SIZE = 8;
    for (size_t i = 0; i < INITIAL_BURST_SIZE; i++) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            dest_bank_addr, packet_payload_size_bytes);
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
    }

    // Signal ready to latency writer
    auto& ready_connection =
        is_downstream ? fabric_connection.get_backward_connection() : fabric_connection.get_forward_connection();
    ready_connection.wait_for_empty_write_slot();
    ready_connection.send_payload_flush_non_blocking_from_address(
        (uint32_t)ready_packet_header, sizeof(PACKET_HEADER_TYPE));

    // Continue normal operation
    while (*teardown_signal_addr == 0) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            dest_bank_addr, packet_payload_size_bytes);
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
    }

    fabric_connection.close();
    noc_async_write_barrier();
}
