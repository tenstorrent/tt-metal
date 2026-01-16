// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "api/dataflow/dataflow_api.h"

#include <cstdint>
#include <cstddef>

// Latency test sender kernel - measures round-trip latency by recording timestamps
// around packet send and ack receipt

constexpr bool enable_fused_payload_with_sync = get_compile_time_arg_val(0) != 0;
constexpr bool sem_inc_only = get_compile_time_arg_val(1) != 0;
constexpr bool is_2d_fabric = get_compile_time_arg_val(2) != 0;

void kernel_main() {
    set_l1_data_cache<false>();
    using namespace tt::tt_fabric;
    size_t arg_idx = 0;

    // Common runtime args
    const size_t result_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t semaphore_address =
        get_arg_val<uint32_t>(arg_idx++);  // Shared sync address (same offset on all devices)
    const size_t payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const size_t num_samples = get_arg_val<uint32_t>(arg_idx++);
    const size_t send_buffer_address = get_arg_val<uint32_t>(arg_idx++);  // Sender's send buffer (write before sending)
    const size_t receive_buffer_address =
        get_arg_val<uint32_t>(arg_idx++);  // Sender's receive buffer (wait for response)
    const uint8_t responder_noc_x =
        static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));  // Responder's virtual NOC X
    const uint8_t responder_noc_y =
        static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));  // Responder's virtual NOC Y
    const size_t responder_receive_buffer_address =
        send_buffer_address;  // Responder receives from sender's send buffer

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
    // Use responder's virtual core coordinates (not sender's coordinates)
    // Send payload to responder's receive buffer (not its timestamp buffer)
    auto dest_semaphore_noc_addr = safe_get_noc_addr(responder_noc_x, responder_noc_y, semaphore_address, 0);
    auto dest_payload_noc_addr =
        safe_get_noc_addr(responder_noc_x, responder_noc_y, responder_receive_buffer_address, 0);

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

    auto send_payload_packet = [&fabric_connection, payload_packet_header, send_buffer_address, payload_size_bytes]() {
        fabric_connection.wait_for_empty_write_slot();
        if (payload_size_bytes > 0) {
            fabric_connection.send_payload_without_header_non_blocking_from_address(
                send_buffer_address, payload_size_bytes);
        }
        fabric_connection.send_payload_flush_non_blocking_from_address(
            (uint32_t)payload_packet_header, sizeof(PACKET_HEADER_TYPE));
    };

    auto wait_for_semaphore_then_reset = [semaphore_address](size_t target_value) {
        noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(semaphore_address), target_value);
        *reinterpret_cast<volatile uint32_t*>(semaphore_address) = 0;
    };

    // Warmup: flush the datapath
    {
        send_seminc_packet();
        wait_for_semaphore_then_reset(1);
    }

    // Store elapsed time (cycles) as uint32_t in result buffer
    volatile uint32_t* result_ptr = reinterpret_cast<volatile uint32_t*>(result_buffer_address);

    // Clear result buffer before writing elapsed times to avoid reading stale data
    uint32_t result_buffer_size = num_samples * sizeof(uint32_t);
    for (uint32_t i = 0; i < num_samples; i++) {
        result_ptr[i] = 0;
    }

    // Main latency measurement loop
    // Use separate send and receive buffers to avoid race conditions
    volatile uint32_t* send_buffer_ptr = reinterpret_cast<volatile uint32_t*>(send_buffer_address);
    volatile uint32_t* receive_buffer_ptr = reinterpret_cast<volatile uint32_t*>(receive_buffer_address);

    // Initialize receive buffer to 0
    *receive_buffer_ptr = 0;

    for (size_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
            // Write incrementing value to send buffer BEFORE sending
            *send_buffer_ptr = sample_idx + 1;
            *receive_buffer_ptr = 0;
        }

        // Send one message per sample
        if constexpr (enable_fused_payload_with_sync) {
            send_payload_packet();
        } else {
            if constexpr (sem_inc_only) {
                send_seminc_packet();
            } else {
                send_payload_packet();
            }
        }
        auto start_timestamp = get_timestamp();

        // Don't want to include noc command buffer response time in the total latency measurement
        noc_async_writes_flushed();

        // Wait on receive buffer for response from responder
        if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
            noc_semaphore_wait(receive_buffer_ptr, sample_idx + 1);
        } else {
            noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(semaphore_address), 1);
            *reinterpret_cast<volatile uint32_t*>(semaphore_address) = 0;
        }

        auto end_timestamp = get_timestamp();

        // Store elapsed time in cycles (truncated to uint32_t, sufficient for latency measurements)
        auto elapsed_cycles = end_timestamp - start_timestamp;
        result_ptr[sample_idx] = static_cast<uint32_t>(elapsed_cycles);
    }

    fabric_connection.close();

    noc_semaphore_set(reinterpret_cast<volatile uint32_t*>(semaphore_address), 0);
    noc_async_full_barrier();
}
