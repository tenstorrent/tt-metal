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
#include "tt_fabric_udm_impl.hpp"

namespace tt::tt_fabric::udm {

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
 * - fabric_reads_num_acked: Number of read acknowledgments received
 * - fabric_nonposted_writes_acked: Number of non-posted write acknowledgments received
 * - fabric_nonposted_atomics_acked: Number of non-posted atomic operation acknowledgments received
 *
 * Each processor maintains its own counter.
 */
inline uint32_t fabric_reads_num_acked __attribute__((used)) = 0;
inline uint32_t fabric_nonposted_writes_acked __attribute__((used)) = 0;
inline uint32_t fabric_nonposted_atomics_acked __attribute__((used)) = 0;

/**
 * @brief Fabric barrier types - equivalent to NocBarrierType
 *
 * Enumeration of different barrier types for fabric operations.
 * These are used to synchronize different types of fabric transactions.
 */
enum class FabricBarrierType : uint8_t {
    READS_NUM_ACKED,          // Track number of read acknowledgments received
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
 * @brief Get the L1 memory address for a specific fabric counter (runtime dm_id version)
 *
 * @tparam barrier_type The type of barrier/counter
 * @param dm_id The DM processor ID (0 to NUM_DMS-1)
 * @return The L1 memory address of the counter
 */
template <FabricBarrierType barrier_type>
inline __attribute__((always_inline)) uint32_t get_fabric_counter_address(uint8_t dm_id) {
    static_assert(static_cast<uint8_t>(barrier_type) < NUM_FABRIC_BARRIER_TYPES);
    uint32_t offset = MEM_FABRIC_COUNTER_BASE +
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
 * @brief Wait for all non-posted fabric writes to be acknowledged
 *
 * This function blocks until all outstanding non-posted writes have been acknowledged.
 * It polls the fabric counter and compares it with the local software counter
 * (fabric_nonposted_writes_acked) that tracks expected acknowledgments.
 */
inline __attribute__((always_inline)) void fabric_write_barrier() {
    do {
        invalidate_l1_cache();
    } while (!fabric_nonposted_writes_flushed());
}

/**
 * @brief Wait for all fabric reads to be acknowledged
 *
 * This function blocks until all outstanding reads have been acknowledged.
 * It polls the fabric counter and compares it with the local software counter
 * (fabric_reads_num_acked) that tracks expected acknowledgments.
 */
inline __attribute__((always_inline)) void fabric_read_barrier() {
    do {
        invalidate_l1_cache();
    } while (!fabric_reads_flushed());
}

/**
 * @brief Wait for all non-posted fabric atomics to be acknowledged
 *
 * This function blocks until all outstanding non-posted atomic operations have been acknowledged.
 * It polls the fabric counter and compares it with the local software counter
 * (fabric_nonposted_atomics_acked) that tracks expected acknowledgments.
 */
inline __attribute__((always_inline)) void fabric_atomic_barrier() {
    do {
        invalidate_l1_cache();
    } while (!fabric_nonposted_atomics_flushed());
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

/**
 * @brief Write data to a remote location through the fabric (similar to ncrisc_noc_fast_write)
 *
 * This function sends data through the fabric network to a remote destination.
 * It automatically handles the fabric connection setup and management internally.
 *
 * @param dst_dev_id Destination device ID for fabric routing
 * @param dst_mesh_id Destination mesh ID for fabric routing
 * @param src_addr Source address in local L1 memory
 * @param dest_addr Destination NOC address (encoded with x,y coordinates and local address)
 * @param len_bytes Number of bytes to write
 * @param multicast Whether this is a multicast operation
 * @param num_dests Number of destinations (for multicast)
 * @param trid Transaction ID for UDM operations
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
    auto& connection = get_or_open_fabric_connection();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = get_or_allocate_header();

    if (multicast) {
        // TODO: Set up multicast header with proper routing

    } else {
        packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_addr}, len_bytes);
        fabric_write_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id, trid);
    }

    connection.wait_for_empty_write_slot();

    connection.send_payload_without_header_non_blocking_from_address(src_addr, len_bytes);
    connection.send_payload_blocking_from_address(
        reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));

    fabric_nonposted_writes_acked += num_dests;
}

/**
 * @brief Send a write acknowledgment back to the sender using atomic increment
 *
 * This function sends an atomic increment to the sender's NONPOSTED_WRITES_ACKED counter
 * to acknowledge receipt of a write packet. It extracts the sender's information from
 * the received packet's UDM control fields and sends an atomic increment to the
 * appropriate counter. The dm_id is extracted from the packet header's risc_id field.
 *
 * @param received_header Pointer to the received packet header containing UDM control fields
 * @param increment_value The value to increment the counter by (default 1)
 * @param flush Whether to flush the atomic operation (default false)
 */
inline __attribute__((always_inline)) void fabric_fast_write_ack(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* received_header, uint32_t increment_value = 1, bool flush = false) {
    auto& connection = get_or_open_fabric_connection();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* ack_header = get_or_allocate_header();

    uint8_t src_chip_id = received_header->udm_control.write.src_chip_id;
    uint16_t src_mesh_id = received_header->udm_control.write.src_mesh_id;
    uint16_t src_noc_xy = received_header->udm_control.write.src_noc_xy;
    uint8_t src_risc_id = received_header->udm_control.write.risc_id;
    uint8_t src_noc_x = src_noc_xy & 0xFF;
    uint8_t src_noc_y = (src_noc_xy >> 8) & 0xFF;

    uint32_t counter_addr = get_fabric_counter_address<FabricBarrierType::NONPOSTED_WRITES_ACKED>(src_risc_id);

    ack_header->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader(get_noc_addr(src_noc_x, src_noc_y, counter_addr), increment_value, flush));
    fabric_write_set_unicast_route(ack_header, src_chip_id, src_mesh_id, 0);

    connection.wait_for_empty_write_slot();
    connection.send_payload_blocking_from_address(reinterpret_cast<uint32_t>(ack_header), sizeof(PACKET_HEADER_TYPE));
}

}  // namespace tt::tt_fabric::udm
