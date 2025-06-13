// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "dataflow_api.h"

#include <cstdint>
#include <cstddef>

constexpr bool enable_fused_payload_with_sync = get_compile_time_arg_val(0) != 0;
constexpr bool payloads_are_mcast = get_compile_time_arg_val(1) != 0;
constexpr bool sem_inc_only = get_compile_time_arg_val(2) != 0;

void kernel_main() {
    using namespace tt::tt_fabric;
    size_t arg_idx = 0;

    // A safe location to dump payload data
    const size_t dest_dummy_payload_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t semaphore_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const size_t payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const size_t burst_size = get_arg_val<uint32_t>(arg_idx++);
    const size_t num_bursts_to_send = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_header_cb = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_header_size_in_headers = get_arg_val<uint32_t>(arg_idx++);
    const size_t num_hops_over_loopback_fabric_to_self = get_arg_val<uint32_t>(arg_idx++);
    const size_t congestion_writers_ready_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const size_t num_congestion_writers = get_arg_val<uint32_t>(arg_idx++);

    // Wait for all congestion writers to be ready
    noc_semaphore_wait_min(
        reinterpret_cast<volatile uint32_t*>(congestion_writers_ready_semaphore), num_congestion_writers);

    size_t has_upstream_congestion_writer = get_arg_val<uint32_t>(arg_idx++) != 0;
    uint64_t upstream_congestion_writer_teardown_noc_addr = 0;
    if (has_upstream_congestion_writer) {
        size_t upstream_congestion_writer_teardown_bank_addr = get_arg_val<uint32_t>(arg_idx++);
        size_t upstream_congestion_writer_teardown_noc_x = get_arg_val<uint32_t>(arg_idx++);
        size_t upstream_congestion_writer_teardown_noc_y = get_arg_val<uint32_t>(arg_idx++);
        upstream_congestion_writer_teardown_noc_addr = safe_get_noc_addr(
            upstream_congestion_writer_teardown_noc_x,
            upstream_congestion_writer_teardown_noc_y,
            upstream_congestion_writer_teardown_bank_addr,
            0);
    }
    size_t num_downstream_congestion_writers = get_arg_val<uint32_t>(arg_idx++);
    size_t* downstream_congestion_writer_teardown_semaphore_addresses_ptr =
        reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_downstream_congestion_writers;
    size_t* downstream_congestion_writer_noc_x_list_ptr = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_downstream_congestion_writers;
    size_t* downstream_congestion_writer_noc_y_list_ptr = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_downstream_congestion_writers;
    size_t* downstream_congestion_writer_hop_distance_list_ptr = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_downstream_congestion_writers;

    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    for (size_t i = 0; i < num_downstream_congestion_writers; i++) {
        uint64_t downstream_noc_addr = safe_get_noc_addr(
            downstream_congestion_writer_noc_x_list_ptr[i],
            downstream_congestion_writer_noc_y_list_ptr[i],
            downstream_congestion_writer_teardown_semaphore_addresses_ptr[i],
            0);
    }

    ASSERT(fabric_connection.is_logically_connected());

    if (!fabric_connection.is_logically_connected()) {
        while (true) {
        }
    }

    fabric_connection.open();

    cb_reserve_back(packet_header_cb, packet_header_size_in_headers);
    const auto packet_header_buffer_address = get_write_ptr(packet_header_cb);

    auto* payload_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* sem_inc_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));

    auto wait_for_semaphore_then_reset = [semaphore_address](size_t target_value) {
        noc_semaphore_wait_min(reinterpret_cast<volatile uint32_t*>(semaphore_address), target_value);
        *reinterpret_cast<volatile uint32_t*>(semaphore_address) = 0;
    };

    // PACKET HEADER SETUP
    if constexpr (payloads_are_mcast) {
        auto mcast_hops = static_cast<uint8_t>(num_hops_over_loopback_fabric_to_self);
        payload_packet_header->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(mcast_hops)});
        sem_inc_packet_header->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(mcast_hops)});
    } else {
        payload_packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops_over_loopback_fabric_to_self));
        sem_inc_packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops_over_loopback_fabric_to_self));
    }
    auto dest_semaphore_noc_addr =
        safe_get_noc_addr(static_cast<uint8_t>(my_x[0]), static_cast<uint8_t>(my_y[0]), semaphore_address, 0);
    auto dest_payload_noc_addr = safe_get_noc_addr(
        static_cast<uint8_t>(my_x[0]), static_cast<uint8_t>(my_y[0]), dest_dummy_payload_buffer_address, 0);
    if constexpr (enable_fused_payload_with_sync) {
        payload_packet_header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                dest_payload_noc_addr, dest_semaphore_noc_addr, 1, std::numeric_limits<uint16_t>::max(), false},
            payload_size_bytes);
    } else {
        payload_packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_payload_noc_addr}, payload_size_bytes);
        sem_inc_packet_header->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{dest_semaphore_noc_addr, 1, std::numeric_limits<uint16_t>::max()});
    }

    auto send_seminc_packet = [&fabric_connection, sem_inc_packet_header]() {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        // print_pkt_header(sem_inc_packet_header);
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)sem_inc_packet_header, sizeof(PACKET_HEADER_TYPE));
    };
    auto send_payload_packet =
        [&fabric_connection, payload_packet_header, dest_dummy_payload_buffer_address, payload_size_bytes]() {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
                dest_dummy_payload_buffer_address, payload_size_bytes);
            fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
                (uint32_t)payload_packet_header, sizeof(PACKET_HEADER_TYPE));
        };
    // Flush the datapath
    {
        DeviceZoneScopedN("Flush");
        send_seminc_packet();
        wait_for_semaphore_then_reset(1);
    }

    auto payload_l1_ptr = reinterpret_cast<volatile uint32_t*>(dest_dummy_payload_buffer_address);
    {
        for (size_t i = 0; i < num_bursts_to_send; i++) {
            // Wait for the fabric endpoint to have a completely empty sender channel buffer
            if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
                if (burst_size > 1) {
                    DPRINT << "STUCK\n";
                    while (1);  // invalid config -- hang instead of reporting garbage numbers
                }
                // Initialize to i + 1 so we can safely reset to 0 and not invalidate the first
                // packet wait (without +1, we'd finish the loop before the first packet arrives
                // since we reset to 0)
                *payload_l1_ptr = i + 1;
            }

            // Burst
            {
                DeviceZoneScopedN("BURST-WRITE");
                for (size_t j = 0; j < burst_size; j++) {
                    if constexpr (enable_fused_payload_with_sync) {
                        send_payload_packet();
                    } else {
                        if constexpr (sem_inc_only) {
                            send_seminc_packet();
                        } else {
                            send_payload_packet();
                        }
                    }
                }
                // Don't want to include noc command buffer response time in the total latency measurement
                noc_async_writes_flushed();
                if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
                    // TODO: add separate src buffer -- technically a race but in practice this will never hit.
                    *payload_l1_ptr = 0;
                }

                {
                    DeviceZoneScopedN("WAIT-FOR-ALL-SEMAPHORES");
                    if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
                        noc_semaphore_wait_min(payload_l1_ptr, i + 1);
                    } else {
                        for (size_t j = 0; j < burst_size; j++) {
                            noc_semaphore_wait_min(reinterpret_cast<volatile uint32_t*>(semaphore_address), j + 1);
                        }
                    }
                }
                *reinterpret_cast<volatile uint32_t*>(semaphore_address) = 0;
            }
        }
    }

    auto send_teardown_message = [packet_header = payload_packet_header](
                                     tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
                                     uint64_t teardown_noc_addr,
                                     size_t num_hops_on_fabric) {
        // Now that we are done, we need to notify all other congestion writers to teardown
        packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops_on_fabric));
        packet_header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            teardown_noc_addr, 1, std::numeric_limits<uint16_t>::max()});

        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
    };
    if (has_upstream_congestion_writer) {
        send_teardown_message(
            fabric_connection.get_backward_connection(), upstream_congestion_writer_teardown_noc_addr, 1);
    }
    for (size_t i = 0; i < num_downstream_congestion_writers; i++) {
        uint64_t downstream_noc_addr = safe_get_noc_addr(
            downstream_congestion_writer_noc_x_list_ptr[i],
            downstream_congestion_writer_noc_y_list_ptr[i],
            downstream_congestion_writer_teardown_semaphore_addresses_ptr[i],
            0);

        send_teardown_message(
            fabric_connection.get_forward_connection(),
            downstream_noc_addr,
            downstream_congestion_writer_hop_distance_list_ptr[i]);
    }

    fabric_connection.close();
    noc_async_write_barrier();
}
