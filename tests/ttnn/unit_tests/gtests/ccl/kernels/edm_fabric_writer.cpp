// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/fabric_connection_manager.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "dataflow_api.h"

#include <cstdint>
#include <cstddef>

void kernel_main() {
    DeviceZoneScopedN("TEST-FULL");
    using namespace tt::fabric;
    size_t arg_idx = 0;

    const size_t dest_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const size_t dest_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const size_t dest_noc_y = get_arg_val<uint32_t>(arg_idx++);

    const size_t num_mcasts = get_arg_val<uint32_t>(arg_idx++);
    const size_t mcast_fwd_hops = get_arg_val<uint32_t>(arg_idx++);
    const size_t mcast_bwd_hops = get_arg_val<uint32_t>(arg_idx++);

    const size_t num_unicasts = get_arg_val<uint32_t>(arg_idx++);
    const size_t unicast_hops = get_arg_val<uint32_t>(arg_idx++);
    const bool unicast_is_fwd = get_arg_val<uint32_t>(arg_idx++) != 0;

    const size_t source_l1_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_header_cb = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_header_size_in_headers = get_arg_val<uint32_t>(arg_idx++);

    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    ASSERT(fabric_connection.is_logically_connected());

    if (!fabric_connection.is_logically_connected()) {
        return;
    }

    DPRINT << "Writing " << (uint32_t)num_mcasts << " mcasts\n";
    DPRINT << "Writing " << (uint32_t)num_unicasts << " unicasts\n";
    DPRINT << "Mcast fwd hops: " << (uint32_t)mcast_fwd_hops << "\n";
    DPRINT << "Mcast bwd hops: " << (uint32_t)mcast_bwd_hops << "\n";
    DPRINT << "Unicast hops: " << (uint32_t)unicast_hops << "\n";
    DPRINT << "Has fwd conn: " << (uint32_t)fabric_connection.has_forward_connection() << "\n";
    DPRINT << "Has bwd conn: " << (uint32_t)fabric_connection.has_backward_connection() << "\n";
    DPRINT << "Open connection\n";
    fabric_connection.open();

    cb_reserve_back(packet_header_cb, packet_header_size_in_headers);
    const auto packet_header_buffer_address = get_write_ptr(packet_header_cb);
    auto* mcast_fwd_packet_header = reinterpret_cast<PacketHeader*>(packet_header_buffer_address);
    auto* mcast_bwd_packet_header =
        reinterpret_cast<PacketHeader*>(packet_header_buffer_address + sizeof(tt::fabric::PacketHeader));
    auto* unicast_packet_header =
        reinterpret_cast<PacketHeader*>(packet_header_buffer_address + sizeof(tt::fabric::PacketHeader) * 2);

    mcast_fwd_packet_header->to_write().to_chip_multicast(
        MulticastRoutingCommandHeader{1, static_cast<uint8_t>(mcast_fwd_hops)});
    mcast_bwd_packet_header->to_write().to_chip_multicast(
        MulticastRoutingCommandHeader{1, static_cast<uint8_t>(mcast_bwd_hops)});
    unicast_packet_header->to_write().to_chip_unicast(UnicastRoutingCommandHeader{static_cast<uint8_t>(unicast_hops)});

    {
        DeviceZoneScopedN("MAIN-WRITE-ZONE");
        for (size_t i = 0; i < num_mcasts; i++) {
            noc_async_write(
                source_l1_buffer_address,
                safe_get_noc_addr(static_cast<uint8_t>(dest_noc_x), static_cast<uint8_t>(dest_noc_y), dest_bank_addr),
                packet_payload_size_bytes);
            if (fabric_connection.has_forward_connection()) {
                mcast_fwd_packet_header->to_noc_unicast(NocUnicastCommandHeader{
                    dest_bank_addr,
                    packet_payload_size_bytes,
                    static_cast<uint8_t>(dest_noc_x),
                    static_cast<uint8_t>(dest_noc_y)});
                // auto &fwd_conn = fabric_connection.get_forward_connection();
                DPRINT << "Wait EDMF\n";
                fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                DPRINT << "Got it\n";
                fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
                    source_l1_buffer_address, packet_payload_size_bytes);
                fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)mcast_fwd_packet_header, sizeof(tt::fabric::PacketHeader));
            }

            if (fabric_connection.has_backward_connection()) {
                mcast_bwd_packet_header->to_noc_unicast(NocUnicastCommandHeader{
                    dest_bank_addr,
                    packet_payload_size_bytes,
                    static_cast<uint8_t>(dest_noc_x),
                    static_cast<uint8_t>(dest_noc_y)});
                // auto &bwd_conn = fabric_connection.get_backward_connection();
                DPRINT << "Wait EDMR\n";
                fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                DPRINT << "Got it\n";
                fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
                    source_l1_buffer_address, packet_payload_size_bytes);
                fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)mcast_bwd_packet_header, sizeof(tt::fabric::PacketHeader));
            }
            noc_async_writes_flushed();
        }
    }

    for (size_t i = 0; i < num_unicasts; i++) {
        auto& fabric_conn =
            unicast_is_fwd ? fabric_connection.get_forward_connection() : fabric_connection.get_backward_connection();
        unicast_packet_header->to_noc_unicast(NocUnicastCommandHeader{
            dest_bank_addr,
            packet_payload_size_bytes,
            static_cast<uint8_t>(dest_noc_x),
            static_cast<uint8_t>(dest_noc_y)});
        fabric_conn.wait_for_empty_write_slot();
        fabric_conn.send_payload_without_header_non_blocking_from_address(
            source_l1_buffer_address, packet_payload_size_bytes);
        fabric_conn.send_payload_flush_blocking_from_address(
            (uint32_t)unicast_packet_header, sizeof(tt::fabric::PacketHeader));
    }

    // noc_async_write_barrier();
    fabric_connection.close();
    // noc_async_write_barrier();
}
