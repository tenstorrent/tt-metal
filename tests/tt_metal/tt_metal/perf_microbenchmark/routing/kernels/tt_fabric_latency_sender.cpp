// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "dataflow_api.h"

#include <cstdint>
#include <cstddef>

// Latency test sender kernel - measures round-trip latency by recording timestamps
// around packet send and ack receipt

constexpr bool enable_fused_payload_with_sync = get_compile_time_arg_val(0) != 0;
constexpr bool sem_inc_only = get_compile_time_arg_val(1) != 0;
constexpr bool is_2d_fabric = get_compile_time_arg_val(2) != 0;

void kernel_main() {
    using namespace tt::tt_fabric;
    size_t arg_idx = 0;

    DPRINT << "SENDER: Starting latency sender kernel\n";

    // Common runtime args
    const size_t result_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t semaphore_address =
        get_arg_val<uint32_t>(arg_idx++);  // Shared sync address (same offset on all devices)
    const size_t payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const size_t burst_size = get_arg_val<uint32_t>(arg_idx++);
    const size_t num_bursts = get_arg_val<uint32_t>(arg_idx++);
    const size_t scratch_buffer_address = get_arg_val<uint32_t>(arg_idx++);  // Sender's scratch (for receiving echo)
    const size_t responder_scratch_buffer_address =
        get_arg_val<uint32_t>(arg_idx++);  // Responder's scratch (for sending TO)

    // Topology-specific routing args
    uint32_t num_hops_to_responder = 0;
    uint32_t dst_device_id = 0;
    uint32_t dst_mesh_id = 0;

    if constexpr (!is_2d_fabric) {
        num_hops_to_responder = get_arg_val<uint32_t>(arg_idx++);
    } else {
        dst_device_id = get_arg_val<uint32_t>(arg_idx++);
        dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    }

    // Build fabric connection
    auto fabric_connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

    fabric_connection.open();

    // Allocate packet headers from pool
    auto* payload_packet_header = PacketHeaderPool::allocate_header();
    auto* sem_inc_packet_header = PacketHeaderPool::allocate_header();

    // Setup packet headers for routing
    if constexpr (!is_2d_fabric) {
        // 1D routing: use Low Latency header with hop count
        fabric_set_unicast_route<false>((LowLatencyPacketHeader*)payload_packet_header, num_hops_to_responder);
        fabric_set_unicast_route<false>((LowLatencyPacketHeader*)sem_inc_packet_header, num_hops_to_responder);
    } else {
        // 2D routing: use Hybrid Mesh header with device/mesh IDs (static routing)
        fabric_set_unicast_route((HybridMeshPacketHeader*)payload_packet_header, dst_device_id, dst_mesh_id);
        fabric_set_unicast_route((HybridMeshPacketHeader*)sem_inc_packet_header, dst_device_id, dst_mesh_id);
    }

    // Setup NOC addresses for destination (responder device)
    // Send payload to responder's scratch buffer (not its timestamp buffer)
    auto dest_semaphore_noc_addr =
        safe_get_noc_addr(static_cast<uint8_t>(my_x[0]), static_cast<uint8_t>(my_y[0]), semaphore_address, 0);
    auto dest_payload_noc_addr = safe_get_noc_addr(
        static_cast<uint8_t>(my_x[0]), static_cast<uint8_t>(my_y[0]), responder_scratch_buffer_address, 0);

    // Setup NOC command headers
    if constexpr (enable_fused_payload_with_sync) {
        payload_packet_header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                dest_payload_noc_addr, dest_semaphore_noc_addr, 1, false},
            payload_size_bytes);
    } else {
        payload_packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_payload_noc_addr}, payload_size_bytes);
        sem_inc_packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{dest_semaphore_noc_addr, 1});
    }

    auto send_seminc_packet = [&fabric_connection, sem_inc_packet_header]() {
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_non_blocking_from_address(
            (uint32_t)sem_inc_packet_header, sizeof(PACKET_HEADER_TYPE));
    };

    auto send_payload_packet =
        [&fabric_connection, payload_packet_header, scratch_buffer_address, payload_size_bytes]() {
            fabric_connection.wait_for_empty_write_slot();
            if (payload_size_bytes > 0) {
                fabric_connection.send_payload_without_header_non_blocking_from_address(
                    scratch_buffer_address, payload_size_bytes);
            }
            fabric_connection.send_payload_flush_non_blocking_from_address(
                (uint32_t)payload_packet_header, sizeof(PACKET_HEADER_TYPE));
        };

    auto wait_for_semaphore_then_reset = [semaphore_address](size_t target_value) {
        noc_semaphore_wait_min(reinterpret_cast<volatile uint32_t*>(semaphore_address), target_value);
        *reinterpret_cast<volatile uint32_t*>(semaphore_address) = 0;
    };

    // Warmup: flush the datapath
    {
        send_seminc_packet();
        wait_for_semaphore_then_reset(1);
    }

    // Validate burst size
    if (burst_size > 1) {
        DPRINT << "ERROR: burst_size > 1 not supported for accurate latency measurement\n";
        while (1);  // Invalid config - hang instead of reporting garbage numbers
    }

    // Main latency measurement loop
    // Store results as pairs of (start_timestamp, end_timestamp) in result buffer
    volatile uint64_t* result_ptr = reinterpret_cast<volatile uint64_t*>(result_buffer_address);

    // Use separate scratch buffer for payload echo to avoid corrupting timestamp data
    auto payload_l1_ptr = reinterpret_cast<volatile uint32_t*>(scratch_buffer_address);
    for (size_t burst_idx = 0; burst_idx < num_bursts; burst_idx++) {
        // Record start timestamp

        if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
            if (burst_size > 1) {
                DPRINT << "STUCK\n";
                while (1);  // invalid config -- hang instead of reporting garbage numbers
            }
            // Initialize to i + 1 so we can safely reset to 0 and not invalidate the first
            // packet wait (without +1, we'd finish the loop before the first packet arrives
            // since we reset to 0)
            *payload_l1_ptr = burst_idx + 1;
        }

        uint64_t start_timestamp = 0;  // get_timestamp();
        uint64_t end_timestamp = 0;    // get_timestamp();
        {
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
            start_timestamp = get_timestamp();
            if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
                *payload_l1_ptr = 0;
            }

            if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
                noc_semaphore_wait_min(payload_l1_ptr, burst_idx + 1);
            } else {
                for (size_t j = 0; j < burst_size; j++) {
                    noc_semaphore_wait_min(reinterpret_cast<volatile uint32_t*>(semaphore_address), j + 1);
                }
            }

            end_timestamp = get_timestamp();
            *reinterpret_cast<volatile uint32_t*>(semaphore_address) = 0;
        }

        // Store timestamp pair in result buffer
        result_ptr[burst_idx * 2] = start_timestamp;
        result_ptr[burst_idx * 2 + 1] = end_timestamp;
    }

    fabric_connection.close();

    noc_async_full_barrier();
}
