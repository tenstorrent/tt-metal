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

// Latency test responder kernel - receives packets from sender and immediately sends ack back

constexpr bool enable_fused_payload_with_sync = get_compile_time_arg_val(0) != 0;
constexpr bool sem_inc_only = get_compile_time_arg_val(1) != 0;
constexpr bool is_2d_fabric = get_compile_time_arg_val(2) != 0;

void kernel_main() {
    using namespace tt::tt_fabric;
    size_t arg_idx = 0;

    DPRINT << "RESPONDER: Starting latency responder kernel\n";

    // Common runtime args
    const size_t timestamp_buffer_address = get_arg_val<uint32_t>(arg_idx++);  // For storing response timestamps
    const size_t semaphore_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const size_t burst_size = get_arg_val<uint32_t>(arg_idx++);
    const size_t num_bursts = get_arg_val<uint32_t>(arg_idx++);
    const size_t sender_scratch_buffer_address = get_arg_val<uint32_t>(arg_idx++);     // For echoing back to sender
    const size_t responder_scratch_buffer_address = get_arg_val<uint32_t>(arg_idx++);  // For receiving payload

    // Topology-specific routing args (for sending back to sender)
    uint32_t num_hops_back_to_sender = 0;
    uint32_t dst_device_id = 0;
    uint32_t dst_mesh_id = 0;

    if constexpr (!is_2d_fabric) {
        num_hops_back_to_sender = get_arg_val<uint32_t>(arg_idx++);
    } else {
        dst_device_id = get_arg_val<uint32_t>(arg_idx++);
        dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    }

    DPRINT << "RESPONDER: Config - payload_size=" << (uint32_t)payload_size_bytes
           << " burst_size=" << (uint32_t)burst_size << " num_bursts=" << (uint32_t)num_bursts
           << " hops=" << (uint32_t)num_hops_back_to_sender << "\n";

    // Build fabric connection for sending response back
    auto fabric_connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

    // ASSERT(fabric_connection.is_logically_connected());
    // if (!fabric_connection.is_logically_connected()) {
    //     while (true) {
    //     }
    // }

    fabric_connection.open();
    DPRINT << "RESPONDER: Fabric connection opened\n";

    // Allocate packet headers from pool
    auto* payload_packet_header = PacketHeaderPool::allocate_header();
    auto* sem_inc_packet_header = PacketHeaderPool::allocate_header();

    // Setup packet headers for routing back to sender
    if constexpr (!is_2d_fabric) {
        // 1D routing: use Low Latency header with hop count
        fabric_set_unicast_route<false>((LowLatencyPacketHeader*)payload_packet_header, num_hops_back_to_sender);
        fabric_set_unicast_route<false>((LowLatencyPacketHeader*)sem_inc_packet_header, num_hops_back_to_sender);
    } else {
        // 2D routing: use Hybrid Mesh header with device/mesh IDs (static routing)
        fabric_set_unicast_route((HybridMeshPacketHeader*)payload_packet_header, dst_device_id, dst_mesh_id);
        fabric_set_unicast_route((HybridMeshPacketHeader*)sem_inc_packet_header, dst_device_id, dst_mesh_id);
    }

    // Setup NOC addresses for destination (sender device)
    auto dest_semaphore_noc_addr =
        safe_get_noc_addr(static_cast<uint8_t>(my_x[0]), static_cast<uint8_t>(my_y[0]), semaphore_address, 0);
    auto dest_payload_noc_addr = safe_get_noc_addr(
        static_cast<uint8_t>(my_x[0]), static_cast<uint8_t>(my_y[0]), sender_scratch_buffer_address, 0);

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
        [&fabric_connection, payload_packet_header, responder_scratch_buffer_address, payload_size_bytes]() {
            fabric_connection.wait_for_empty_write_slot();
            if (payload_size_bytes > 0) {
                fabric_connection.send_payload_without_header_non_blocking_from_address(
                    responder_scratch_buffer_address, payload_size_bytes);
            }
            fabric_connection.send_payload_flush_non_blocking_from_address(
                (uint32_t)payload_packet_header, sizeof(PACKET_HEADER_TYPE));
        };

    auto wait_for_semaphore_then_reset = [semaphore_address](size_t target_value) {
        noc_semaphore_wait_min(reinterpret_cast<volatile uint32_t*>(semaphore_address), target_value);
        *reinterpret_cast<volatile uint32_t*>(semaphore_address) = 0;
    };

    // Warmup: respond to flush packet
    {
        wait_for_semaphore_then_reset(1);
        send_seminc_packet();
    }

    // Validate burst size
    if (burst_size > 1) {
        DPRINT << "ERROR: burst_size > 1 not supported for accurate latency measurement\n";
        while (1);  // Invalid config - hang instead of reporting garbage numbers
    }

    // Main response loop
    // Wait for incoming packets and immediately send ack back
    // Use scratch buffer for receiving payloads to avoid overwriting timestamps
    auto payload_l1_ptr = reinterpret_cast<volatile uint32_t*>(responder_scratch_buffer_address);

    // Store response timestamps in timestamp buffer
    volatile uint64_t* result_ptr = reinterpret_cast<volatile uint64_t*>(timestamp_buffer_address);

    if (burst_size == 1) {
        // for burst size one, we can safely assume we don't need to wait for space because
        // we are only sending 1 packet at a time, and the fact that we received a packet
        // indicates our previous response packet was flushed through the system
        for (size_t burst_idx = 0; burst_idx < num_bursts; burst_idx++) {
            // Wait for incoming packet from sender
            // wait_for_semaphore_then_reset(burst_idx + 1);  // +2 because warmup used 1

            if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
                noc_semaphore_wait(payload_l1_ptr, burst_idx + 1);
            } else {
                noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(semaphore_address), burst_idx + 1);
            }

            // Capture start timestamp after receiving packet
            uint64_t start_timestamp = get_timestamp();

            if constexpr (enable_fused_payload_with_sync) {
                if (payload_size_bytes > 0) {
                    fabric_connection.send_payload_without_header_non_blocking_from_address(
                        responder_scratch_buffer_address, payload_size_bytes);
                }
                fabric_connection.send_payload_flush_non_blocking_from_address(
                    (uint32_t)payload_packet_header, sizeof(PACKET_HEADER_TYPE));
            } else {
                if constexpr (sem_inc_only) {
                    fabric_connection.send_payload_flush_non_blocking_from_address(
                        (uint32_t)sem_inc_packet_header, sizeof(PACKET_HEADER_TYPE));
                } else {
                    if (payload_size_bytes > 0) {
                        fabric_connection.send_payload_without_header_non_blocking_from_address(
                            responder_scratch_buffer_address, payload_size_bytes);
                    }
                    fabric_connection.send_payload_flush_non_blocking_from_address(
                        (uint32_t)payload_packet_header, sizeof(PACKET_HEADER_TYPE));
                }
            }

            // Capture end timestamp after sending response
            uint64_t end_timestamp = get_timestamp();

            // Store timestamp pair in result buffer
            result_ptr[burst_idx * 2] = start_timestamp;
            result_ptr[burst_idx * 2 + 1] = end_timestamp;
        }
    } else {
        for (size_t burst_idx = 0; burst_idx < num_bursts; burst_idx++) {
            // Wait for incoming packet from sender
            // wait_for_semaphore_then_reset(burst_idx + 1);  // +2 because warmup used 1
            if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
                noc_semaphore_wait(payload_l1_ptr, burst_idx + 1);
            } else {
                noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(semaphore_address), burst_idx + 1);
            }

            // Capture start timestamp after receiving packet
            uint64_t start_timestamp = get_timestamp();

            for (size_t i = 0; i < burst_size; i++) {
                if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
                    noc_semaphore_wait(payload_l1_ptr, burst_idx + 1);
                } else {
                    noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(semaphore_address), burst_idx + 1);
                }

                if constexpr (enable_fused_payload_with_sync) {
                    fabric_connection.wait_for_empty_write_slot();
                    if (payload_size_bytes > 0) {
                        fabric_connection.send_payload_without_header_non_blocking_from_address(
                            responder_scratch_buffer_address, payload_size_bytes);
                    }
                    fabric_connection.send_payload_flush_non_blocking_from_address(
                        (uint32_t)payload_packet_header, sizeof(PACKET_HEADER_TYPE));
                } else {
                    if constexpr (sem_inc_only) {
                        fabric_connection.wait_for_empty_write_slot();
                        fabric_connection.send_payload_flush_non_blocking_from_address(
                            (uint32_t)sem_inc_packet_header, sizeof(PACKET_HEADER_TYPE));
                    } else {
                        fabric_connection.wait_for_empty_write_slot();
                        if (payload_size_bytes > 0) {
                            fabric_connection.send_payload_without_header_non_blocking_from_address(
                                responder_scratch_buffer_address, payload_size_bytes);
                        }
                        fabric_connection.send_payload_flush_non_blocking_from_address(
                            (uint32_t)payload_packet_header, sizeof(PACKET_HEADER_TYPE));
                    }
                }
            }

            // Capture end timestamp after sending response
            uint64_t end_timestamp = get_timestamp();

            // Store timestamp pair in result buffer
            result_ptr[burst_idx * 2] = start_timestamp;
            result_ptr[burst_idx * 2 + 1] = end_timestamp;
        }
    }

    fabric_connection.close();

    noc_async_full_barrier();
}
