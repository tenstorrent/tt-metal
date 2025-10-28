// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "core_config.h"
#include "dev_mem_map.h"
#include "noc_nonblocking_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include <type_traits>

/**
 * @brief Number of Data Mover (DM) processors for fabric operations
 *
 * This is equivalent to NUM_NOCS for NoC operations but specifically for fabric.
 * The value comes from the architecture-specific core_config.h file:
 * - Wormhole: 2 DMs (DM0 and DM1)
 * - Blackhole: 2 DMs (DM0 and DM1)
 * - Quasar: 8 DMs
 */
#define NUM_DMS MaxDMProcessorsPerCoreType

/**
 * @brief Fabric transaction counters - equivalent to NoC counters
 *
 * These counters track fabric read/write operations similar to how NoC tracks them.
 * They mirror the NoC counters from noc_nonblocking_api.h but for fabric operations.
 *
 * The counters are used to track:
 * - Number of read requests issued through the fabric
 * - Number of non-posted write requests issued (require acknowledgment)
 * - Number of non-posted write acknowledgments received
 * - Number of non-posted atomic operation acknowledgments received
 * - Number of posted write requests issued (no acknowledgment required)
 *
 * Each processor maintains its own counter.
 */
extern uint32_t fabric_reads_num_acked;
extern uint32_t fabric_nonposted_writes_acked;
extern uint32_t fabric_nonposted_atomics_acked;

/**
 * @brief Fabric barrier types - equivalent to NocBarrierType
 *
 * Enumeration of different barrier types for fabric operations.
 * These are used to synchronize different types of fabric transactions.
 */
enum class FabricBarrierType : uint8_t {
    READS_NUM_ACKED,          // Track number of read requests issued
    NONPOSTED_WRITES_ACKED,   // Track number of non-posted write acknowledgments
    NONPOSTED_ATOMICS_ACKED,  // Track number of non-posted atomic acknowledgments
    COUNT                     // Total number of barrier types
};

static constexpr uint8_t NUM_FABRIC_BARRIER_TYPES = static_cast<uint8_t>(FabricBarrierType::COUNT);

/**
 * @brief Get the L1 memory address for a specific fabric counter
 *
 * @tparam dm_id The DM processor ID (0 to NUM_DMS-1)
 * @tparam barrier_type The type of barrier/counter
 * @return The L1 memory address of the counter
 */
template <uint8_t dm_id, FabricBarrierType barrier_type>
inline __attribute__((always_inline)) uint32_t get_fabric_counter_address() {
    static_assert(dm_id < MaxDMProcessorsPerCoreType);
    static_assert(static_cast<uint8_t>(barrier_type) < NUM_FABRIC_BARRIER_TYPES);
    constexpr uint32_t offset =
        MEM_FABRIC_COUNTER_BASE +
        (dm_id * NUM_FABRIC_BARRIER_TYPES + static_cast<uint8_t>(barrier_type)) * MEM_FABRIC_COUNTER_SIZE;
    return offset;
}

/**
 * @brief Get the value of a fabric counter from L1 memory
 */
template <uint8_t dm_id, FabricBarrierType barrier_type>
inline __attribute__((always_inline)) uint32_t get_fabric_counter_val() {
    uint32_t counter_addr = get_fabric_counter_address<dm_id, barrier_type>();
    return *reinterpret_cast<volatile uint32_t*>(counter_addr);
}

/**
 * @brief Increment a fabric counter value in L1 memory
 */
template <uint8_t dm_id, FabricBarrierType barrier_type>
inline __attribute__((always_inline)) void inc_fabric_counter_val(uint32_t inc = 1) {
    uint32_t counter_addr = get_fabric_counter_address<dm_id, barrier_type>();
    *reinterpret_cast<volatile uint32_t*>(counter_addr) += inc;
}

/**
 * @brief Set a fabric counter value in L1 memory
 */
template <uint8_t dm_id, FabricBarrierType barrier_type>
inline __attribute__((always_inline)) void set_fabric_counter_val(uint32_t val) {
    uint32_t counter_addr = get_fabric_counter_address<dm_id, barrier_type>();
    *reinterpret_cast<volatile uint32_t*>(counter_addr) = val;
}

/**
 * @brief Check if all fabric reads have been acknowledged
 *
 * Compares the L1 memory counter (updated by hardware) with the software counter to determine
 * if all issued read requests have been acknowledged.
 *
 * @return true if all reads have been acknowledged, false otherwise
 */
inline __attribute__((always_inline)) bool fabric_reads_flushed() {
    uint32_t status_counter = get_fabric_counter_val<proc_type, FabricBarrierType::READS_NUM_ACKED>();
    return (status_counter == fabric_reads_num_acked);
}

/**
 * @brief Check if all fabric non-posted writes have been acknowledged
 *
 * @return true if all non-posted writes have been acknowledged, false otherwise
 */
inline __attribute__((always_inline)) bool fabric_nonposted_writes_flushed() {
    uint32_t status_counter = get_fabric_counter_val<proc_type, FabricBarrierType::NONPOSTED_WRITES_ACKED>();
    return (status_counter == fabric_nonposted_writes_acked);
}

/**
 * @brief Check if all fabric non-posted atomics have been acknowledged
 *
 * @return true if all non-posted atomics have been acknowledged, false otherwise
 */
inline __attribute__((always_inline)) bool fabric_nonposted_atomics_flushed() {
    uint32_t status_counter = get_fabric_counter_val<proc_type, FabricBarrierType::NONPOSTED_ATOMICS_ACKED>();
    return (status_counter == fabric_nonposted_atomics_acked);
}

/**
 * @brief Initialize fabric counters for the current processor
 *
 * Reads the current hardware status from L1 memory and initializes the software counters
 * to match, establishing a baseline for tracking fabric operations.
 * Uses proc_type which is determined at compile time (DM0 for BRISC, DM1 for others).
 * This follows the same pattern as noc_local_state_init.
 */
inline __attribute__((always_inline)) void fabric_local_state_init() {
    // Read current counter values from L1 memory (hide latency by reading all first, then writing)
    uint32_t reads_ack = get_fabric_counter_val<proc_type, FabricBarrierType::READS_NUM_ACKED>();
    uint32_t nonposted_writes_ack = get_fabric_counter_val<proc_type, FabricBarrierType::NONPOSTED_WRITES_ACKED>();
    uint32_t nonposted_atomics_ack = get_fabric_counter_val<proc_type, FabricBarrierType::NONPOSTED_ATOMICS_ACKED>();

    // Initialize software counters to match hardware state
    fabric_reads_num_acked = reads_ack;
    fabric_nonposted_writes_acked = nonposted_writes_ack;
    fabric_nonposted_atomics_acked = nonposted_atomics_ack;
}

/**
 * @brief Full synchronization barrier for fabric operations on current processor
 *
 * Waits for all pending fabric operations to complete on the current DM processor.
 */
inline __attribute__((always_inline)) void fabric_full_sync() {
    while (!fabric_reads_flushed());
    while (!fabric_nonposted_writes_flushed());
    while (!fabric_nonposted_atomics_flushed());
}

// Forward declarations for fabric connection management
namespace tt::tt_fabric {
class WorkerToFabricEdmSender;
}

// Helper template for static_assert
template <typename T>
inline constexpr bool always_false_v = false;

/**
 * @brief Helper function to set unicast routing based on packet header type
 *
 * This function provides a unified interface for setting up fabric unicast routing
 * across different packet header types (UDMHybridMeshPacketHeader, UDMLowLatencyPacketHeader).
 * It automatically selects the appropriate routing API based on the packet header type.
 *
 * @tparam PACKET_HEADER_TYPE The type of packet header (auto-deduced)
 * @param packet_header Pointer to the packet header
 * @param dst_dev_id Destination device ID
 * @param dst_mesh_id Destination mesh ID
 * @param trid Transaction ID for UDM operations
 */
template <typename PACKET_HEADER_TYPE>
FORCE_INLINE void udm_fabric_set_unicast_route(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header, uint16_t dst_dev_id, uint16_t dst_mesh_id, uint16_t trid) {
    uint16_t my_noc_xy = my_x[edm_to_local_chip_noc] | my_y[edm_to_local_chip_noc] << 8;

    // Create udm_fields struct with the necessary fields
    tt::tt_fabric::udm_fields udm = {my_noc_xy, proc_type, trid};

    if constexpr (std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::UDMHybridMeshPacketHeader>) {
        fabric_set_unicast_route(packet_header, udm, dst_dev_id, dst_mesh_id);
    } else if constexpr (std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::UDMLowLatencyPacketHeader>) {
        fabric_set_unicast_route(packet_header, udm, dst_dev_id);
    } else {
        static_assert(always_false_v<PACKET_HEADER_TYPE>, "Unsupported PACKET_HEADER_TYPE for fabric routing");
    }
}

/**
 * @brief Get or create a singleton fabric connection for the current worker
 *
 * This function manages a static connection instance that persists across calls.
 * The connection is initialized on first use and reused for subsequent operations.
 *
 * @param rt_args_idx Reference to runtime args index (will be updated if connection needs to be built)
 * @return Reference to the fabric connection
 */
inline tt::tt_fabric::WorkerToFabricEdmSender& get_fabric_connection(size_t& rt_args_idx) {
    static tt::tt_fabric::WorkerToFabricEdmSender* connection = nullptr;
    static bool initialized = false;

    if (!initialized) {
        // Build connection from runtime args on first use
        // This assumes the runtime args are set up properly by the host
        // TODO: instead of using rt args, use the reserved L1 region for get the correct ETH channel, and semaphore
        // addresses.
        static tt::tt_fabric::WorkerToFabricEdmSender conn;
        conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
        conn.open();
        connection = &conn;
        initialized = true;
    }

    return *connection;
}

/**
 * @brief Write data to a remote location through the fabric (similar to ncrisc_noc_fast_write)
 *
 * This function sends data through the fabric network to a remote destination.
 * It automatically handles the fabric connection setup and management internally.
 *
 * @param src_addr Source address in local L1 memory
 * @param dest_addr Destination NOC address (encoded with x,y coordinates and local address)
 * @param len_bytes Number of bytes to write
 * @param dst_dev_id Destination device ID for fabric routing
 * @param dst_mesh_id Destination mesh ID for fabric routing
 * @param multicast Whether this is a multicast operation
 * @param num_dests Number of destinations (for multicast)
 * @param trid transaction id
 */
inline __attribute__((always_inline)) void fabric_fast_write(
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    bool multicast = false,
    uint32_t num_dests = 1,
    uint16_t trid = 0) {
    // Get or create the singleton fabric connection
    // TODO: instead of using rt args, use the reserved L1 region for get the correct ETH channel, and semaphore
    // addresses.
    size_t rt_args_idx = 0;  // This would need to be set appropriately
    auto& connection = get_fabric_connection(rt_args_idx);

    // Allocate packet header from pool
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::allocate_header();

    // Configure packet header based on operation type
    if (multicast) {
        // TODO: Set up multicast header with proper routing

    } else {
        packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_addr}, len_bytes);
        udm_fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id, trid);
    }

    connection.wait_for_empty_write_slot();

    connection.send_payload_without_header_non_blocking_from_address(src_addr, len_bytes);
    connection.send_payload_blocking_from_address(
        reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));

    fabric_nonposted_writes_acked += num_dests;

    // Return packet header to pool
    PacketHeaderPool::release_header(reinterpret_cast<volatile tt_l1_ptr uint8_t*>(packet_header));
}
