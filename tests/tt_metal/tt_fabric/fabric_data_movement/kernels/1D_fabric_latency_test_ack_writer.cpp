// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "dataflow_api.h"

#include <cstdint>
#include <cstddef>

// This kernel receives the message from the latency test writer and sends an ack back to it

constexpr bool enable_fused_payload_with_sync = get_compile_time_arg_val(0) != 0;
constexpr bool sem_inc_only = get_compile_time_arg_val(1) != 0;

void kernel_main() {
    using namespace tt::tt_fabric;
    size_t arg_idx = 0;

    // A safe location to dump payload data
    const size_t dest_dummy_payload_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t semaphore_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const size_t burst_size = get_arg_val<uint32_t>(arg_idx++);
    const size_t num_bursts_to_send = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_header_cb = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_header_size_in_headers = get_arg_val<uint32_t>(arg_idx++);
    const size_t num_hops_upstream = get_arg_val<uint32_t>(arg_idx++);

    auto fabric_connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

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

    payload_packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops_upstream));
    sem_inc_packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops_upstream));

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
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_non_blocking_from_address(
            (uint32_t)sem_inc_packet_header, sizeof(PACKET_HEADER_TYPE));
    };
    auto send_payload_packet =
        [&fabric_connection, payload_packet_header, dest_dummy_payload_buffer_address, payload_size_bytes]() {
            fabric_connection.wait_for_empty_write_slot();
            fabric_connection.send_payload_without_header_non_blocking_from_address(
                dest_dummy_payload_buffer_address, payload_size_bytes);
            fabric_connection.send_payload_flush_non_blocking_from_address(
                (uint32_t)payload_packet_header, sizeof(PACKET_HEADER_TYPE));
        };
    // Flush the datapath
    {
        wait_for_semaphore_then_reset(1);
        send_seminc_packet();
    }

    if (burst_size > 1) {
        DPRINT << "STUCK\n";
        while (1);  // invalid config -- hang instead of reporting garbage numbers
    }
    auto payload_l1_ptr = reinterpret_cast<volatile uint32_t*>(dest_dummy_payload_buffer_address);

    for (size_t i = 0; i < num_bursts_to_send; i++) {
        // Wait for the fabric endpoint to have a completely empty sender channel buffer
        if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
            noc_semaphore_wait(payload_l1_ptr, i + 1);
        } else {
            noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(semaphore_address), i + 1);
        }

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

    fabric_connection.close();
    noc_async_write_barrier();
}
