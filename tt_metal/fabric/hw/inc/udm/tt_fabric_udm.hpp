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
#include "tt_metal/fabric/hw/inc/udm/udm_memory_pool.hpp"
#include <type_traits>
#include "tt_fabric_udm_impl.hpp"
#include "api/debug/dprint.h"

namespace tt::tt_fabric::udm {

// Placeholder max page size for the addrgen until the page size is properly visible by the worker
// https://github.com/tenstorrent/tt-metal/issues/25966
static constexpr uint32_t max_fabric_payload_size = 4352;

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
FORCE_INLINE uint32_t get_fabric_counter_address() {
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
FORCE_INLINE uint32_t get_fabric_counter_address(uint8_t dm_id) {
    static_assert(static_cast<uint8_t>(barrier_type) < NUM_FABRIC_BARRIER_TYPES);
    uint32_t offset = MEM_FABRIC_COUNTER_BASE +
                      (dm_id * NUM_FABRIC_BARRIER_TYPES + static_cast<uint8_t>(barrier_type)) * MEM_FABRIC_COUNTER_SIZE;
    return offset;
}

/**
 * @brief Get the value of a fabric counter from L1 memory
 */
template <uint8_t dm_id, FabricBarrierType barrier_type>
FORCE_INLINE uint32_t get_fabric_counter_val() {
    uint32_t counter_addr = get_fabric_counter_address<dm_id, barrier_type>();
    return *reinterpret_cast<volatile uint32_t*>(counter_addr);
}

/**
 * @brief Increment a fabric counter value in L1 memory
 */
template <uint8_t dm_id, FabricBarrierType barrier_type>
FORCE_INLINE void inc_fabric_counter_val(uint32_t inc = 1) {
    uint32_t counter_addr = get_fabric_counter_address<dm_id, barrier_type>();
    *reinterpret_cast<volatile uint32_t*>(counter_addr) += inc;
}

/**
 * @brief Set a fabric counter value in L1 memory
 */
template <uint8_t dm_id, FabricBarrierType barrier_type>
FORCE_INLINE void set_fabric_counter_val(uint32_t val) {
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
FORCE_INLINE bool fabric_reads_flushed() {
    uint32_t status_counter = get_fabric_counter_val<proc_type, FabricBarrierType::READS_NUM_ACKED>();
    return (status_counter == fabric_reads_num_acked);
}

/**
 * @brief Check if all fabric non-posted writes have been acknowledged
 *
 * @return true if all non-posted writes have been acknowledged, false otherwise
 */
FORCE_INLINE bool fabric_nonposted_writes_flushed() {
    uint32_t status_counter = get_fabric_counter_val<proc_type, FabricBarrierType::NONPOSTED_WRITES_ACKED>();
    return (status_counter == fabric_nonposted_writes_acked);
}

/**
 * @brief Check if all fabric non-posted atomics have been acknowledged
 *
 * @return true if all non-posted atomics have been acknowledged, false otherwise
 */
FORCE_INLINE bool fabric_nonposted_atomics_flushed() {
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
FORCE_INLINE void fabric_write_barrier() {
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
FORCE_INLINE void fabric_read_barrier() {
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
FORCE_INLINE void fabric_atomic_barrier() {
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
FORCE_INLINE void fabric_local_state_init() {
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
FORCE_INLINE void fabric_full_sync() {
    while (!fabric_reads_flushed());
    while (!fabric_nonposted_writes_flushed());
    while (!fabric_nonposted_atomics_flushed());
}

/**
 * @brief Internal helper for sending a single fabric packet
 *
 * @param connection Fabric connection to use
 * @param packet_header Packet header with routing already configured
 * @param src_addr Source address in local L1 memory
 * @param dest_addr Destination NOC address
 * @param len_bytes Number of bytes to write (must be <= max_fabric_payload_size)
 * @param multicast Whether this is a multicast operation
 * @param num_dests Number of destinations (for multicast)
 */
FORCE_INLINE void fabric_fast_write(
    WorkerToFabricEdmSender& connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    bool multicast = false,
    uint32_t num_dests = 1) {
    if (multicast) {
        // TODO: Set up multicast header with proper routing
        ASSERT(false);
        while (1) {
        }
    } else {
        packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_addr}, len_bytes);
    }

    connection.wait_for_empty_write_slot();
    connection.send_payload_without_header_non_blocking_from_address(src_addr, len_bytes);
    connection.send_payload_flush_non_blocking_from_address(
        reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));

    noc_async_writes_flushed();
}

/**
 * @brief Write data to a remote location through the fabric (similar to ncrisc_noc_fast_write)
 *
 * This function sends data through the fabric network to a remote destination.
 * It automatically handles the fabric connection setup and management internally.
 * When sending multiple packets, all packets except the last one are sent as posted writes
 * (no acknowledgment needed), and only the last packet is sent as non-posted to ensure
 * all data has been received.
 *
 * @param dst_dev_id Destination device ID for fabric routing
 * @param dst_mesh_id Destination mesh ID for fabric routing
 * @param src_addr Source address in local L1 memory
 * @param dest_addr Destination NOC address (encoded with x,y coordinates and local address)
 * @param len_bytes Number of bytes to write
 * @param multicast Whether this is a multicast operation
 * @param num_dests Number of destinations (for multicast)
 * @param trid Transaction ID for UDM operations
 * @param posted Whether to use posted writes (1) or non-posted writes (0) for single packet case
 */
FORCE_INLINE void fabric_fast_write_any_len(
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    bool multicast = false,
    uint32_t num_dests = 1,
    uint8_t trid = 0,
    uint8_t posted = 0) {
    auto& connection = get_or_open_fabric_connection();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = get_or_allocate_header();

    // For optimization purpose, the non-last fabric writes will be posted, and only the last fabric write is determined
    // by the user posted arg.
    fabric_write_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id, trid, 1 /* posted */);
    while (len_bytes > max_fabric_payload_size) {
        fabric_fast_write(
            connection, packet_header, src_addr, dest_addr, max_fabric_payload_size, multicast, num_dests);

        src_addr += max_fabric_payload_size;
        dest_addr += max_fabric_payload_size;
        len_bytes -= max_fabric_payload_size;
    }
    fabric_write_set_unicast_route_control_field<UDM_CONTROL_FIELD::POSTED>(packet_header, posted);
    fabric_fast_write(connection, packet_header, src_addr, dest_addr, len_bytes, multicast, num_dests);

    if (!posted) {
        fabric_nonposted_writes_acked += num_dests;
    }

    release_fabric_connection();
}

/**
 * @brief Write a single 32-bit value to a remote location through the fabric using inline write
 *
 * This function sends a 32-bit value through the fabric network to a remote destination
 * using an inline write. Inline writes are efficient for small data as the value is embedded
 * directly in the packet header rather than sent as separate payload.
 *
 * @param dst_dev_id Destination device ID for fabric routing
 * @param dst_mesh_id Destination mesh ID for fabric routing
 * @param val The 32-bit value to write
 * @param dest_addr Destination NOC address (encoded with x,y coordinates and local address)
 * @param multicast Whether this is a multicast operation
 * @param num_dests Number of destinations (for multicast)
 * @param trid Transaction ID for UDM operations
 * @param posted Whether to use posted writes (1) or non-posted writes (0)
 */
FORCE_INLINE void fabric_fast_write_dw_inline(
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t val,
    uint64_t dest_addr,
    bool multicast = false,
    uint32_t num_dests = 1,
    uint8_t trid = 0,
    uint8_t posted = 0) {
    auto& connection = get_or_open_fabric_connection();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = get_or_allocate_header();

    fabric_write_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id, trid, posted);

    if (multicast) {
        // TODO: Set up multicast header with proper routing
        ASSERT(false);
        while (1) {
        }
    } else {
        packet_header->to_noc_unicast_inline_write(NocUnicastInlineWriteCommandHeader{dest_addr, val});
    }

    connection.wait_for_empty_write_slot();
    connection.send_payload_flush_non_blocking_from_address(
        reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));

    if (!posted) {
        fabric_nonposted_writes_acked += num_dests;
    }

    noc_async_writes_flushed();

    release_fabric_connection();
}

/**
 * @brief Perform an atomic increment to a remote location through the fabric using inline atomic
 *
 * This function sends an atomic increment command through the fabric network to a remote destination.
 * The atomic increment is efficient as the increment value is embedded directly in the packet header
 * rather than sent as separate payload. The operation atomically increments a counter at the destination.
 * We reuse UDMWriteControlHeader for tracking source information since atomic increments follow
 * similar acknowledgment patterns as writes.
 *
 * @param dst_dev_id Destination device ID for fabric routing
 * @param dst_mesh_id Destination mesh ID for fabric routing
 * @param incr_val The value to atomically add to the destination
 * @param dest_addr Destination NOC address (encoded with x,y coordinates and local address)
 * @param trid Transaction ID for UDM operations
 * @param posted Whether to use posted atomics (1) or non-posted atomics (0, default)
 * @param flush Whether to flush the atomic operation (default true)
 */
FORCE_INLINE void fabric_fast_atomic_inc(
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t incr_val,
    uint64_t dest_addr,
    uint8_t trid = 0,
    uint8_t posted = 0,
    bool flush = true) {
    auto& connection = get_or_open_fabric_connection();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = get_or_allocate_header();

    fabric_write_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id, trid, posted);
    packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(dest_addr, incr_val, flush));

    connection.wait_for_empty_write_slot();
    connection.send_payload_flush_non_blocking_from_address(
        reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));

    if (!posted) {
        fabric_nonposted_atomics_acked += 1;
    }

    noc_async_writes_flushed();

    release_fabric_connection();
}

// Map downstream direction to mux array index [0-2], excluding my_direction
// Examples:
// - EAST Mux (my_direction=0): WEST(1)→0, NORTH(2)→1, SOUTH(3)→2
// - WEST Mux (my_direction=1): EAST(0)→0, NORTH(2)→1, SOUTH(3)→2
// - NORTH Mux (my_direction=2): EAST(0)→0, WEST(1)→1, SOUTH(3)→2
// - SOUTH Mux (my_direction=3): EAST(0)→0, WEST(1)→1, NORTH(2)→2
constexpr uint32_t direction_to_mux_index_map[eth_chan_directions::COUNT][eth_chan_directions::COUNT] = {
    {0, 0, 1, 2},  // EAST Mux -> WEST, NORTH, SOUTH
    {0, 0, 1, 2},  // WEST Mux -> EAST, NORTH, SOUTH
    {0, 1, 0, 2},  // NORTH Mux -> EAST, WEST, SOUTH
    {0, 1, 2, 0},  // SOUTH Mux -> EAST, WEST, NORTH
};

template <
    uint32_t Direction,
    typename FabricConnectionType,
    typename DownstreamMuxConnectionType,
    size_t NumConnections>
FORCE_INLINE bool forward_to_downstream_mux_or_local_router(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    FabricConnectionType& fabric_connection,
    std::array<DownstreamMuxConnectionType, NumConnections>& downstream_mux_connections) {
    uint32_t mux_dir;
    NocSendType packet_type = packet_header->get_noc_send_type();
    // TODO: remove this check when we commonize some common fields between read/write
    if (packet_type == NOC_UNICAST_READ) {
        mux_dir = packet_header->udm_control.read.initial_direction;
    } else {
        mux_dir = packet_header->udm_control.write.initial_direction;
    }

    bool can_forward = true;
    if (Direction != mux_dir) {
        // Forward to the correct downstream mux
        uint32_t mux_index = direction_to_mux_index_map[Direction][mux_dir];
        can_forward = downstream_mux_connections[mux_index].edm_has_space_for_packet();
        if (can_forward) {
            downstream_mux_connections[mux_index].send_payload_flush_non_blocking_from_address(
                (uint32_t)packet_header, packet_header->get_payload_size_including_header());
        }
    } else {
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_non_blocking_from_address(
            (uint32_t)packet_header, packet_header->get_payload_size_including_header());
    }
    return can_forward;
}

/**
 * @brief Select the appropriate mux connection index based on relay direction and destination
 *
 * For EW relays, determines if ACK needs NS routing and selects the appropriate perpendicular connection.
 * For NS relays, always returns local connection (index 0).
 *
 * @tparam Direction Direction of this relay (EAST=0, WEST=1, NORTH=2, SOUTH=3)
 * @param dst_chip_id Destination chip ID
 * @return Mux connection index: 0=local, 1=downstream_en, 2=downstream_ws
 */
template <uint32_t Direction>
FORCE_INLINE uint32_t select_relay_to_mux_connection(uint16_t dst_chip_id) {
    uint32_t mux_idx = 0;  // Default: use local connection (index 0)
    if constexpr (Direction == eth_chan_directions::EAST || Direction == eth_chan_directions::WEST) {
        // For EW relays, check if ACK packet needs NS routing by querying routing info
        auto* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
        auto* routing_info = reinterpret_cast<tt_l1_ptr intra_mesh_routing_path_t<2, true>*>(ROUTING_PATH_BASE_2D);
        uint16_t my_chip_id = routing_table->my_device_id;

        const auto& compressed_route = routing_info->paths[dst_chip_id];
        uint8_t ns_hops = compressed_route.get_ns_hops();
        if (ns_hops > 0) {
            // is there another way to know whether it's north or south hops?
            if (dst_chip_id < my_chip_id) {
                mux_idx = 1;
            } else {
                mux_idx = 2;
            }
        }
    }
    return mux_idx;
}

/**
 * @brief Send an acknowledgment back to the sender using atomic increment
 *
 * This function sends an atomic increment to the sender's counter to acknowledge
 * receipt of a packet (write, atomic, etc.). It extracts the sender's information from
 * the registered response and sends an atomic increment to the appropriate counter
 * based on the FabricBarrierType template parameter.
 *
 * Note: The mux connection is already selected by the caller based on routing logic.
 * Only non-posted operations are registered, so no posted flag check is needed.
 *
 * @tparam BarrierType The type of barrier counter to increment (e.g., NONPOSTED_WRITES_ACKED, NONPOSTED_ATOMICS_ACKED)
 * @tparam FabricConnectionType The connection type
 * @param connection Single mux connection (already selected by caller)
 * @param response Pointer to the registered response containing sender information
 * @param increment_value The value to increment the counter by (default 1)
 * @param flush Whether to flush the atomic operation (default false)
 */
template <FabricBarrierType BarrierType, typename FabricConnectionType>
FORCE_INLINE void fabric_fast_ack(
    FabricConnectionType& connection,
    volatile RegisteredResponse* response,
    uint32_t increment_value = 1,
    bool flush = false) {
    ASSERT(connection.edm_has_space_for_packet());
    volatile tt_l1_ptr PACKET_HEADER_TYPE* ack_header = get_or_allocate_header();

    uint8_t src_chip_id = response->src_chip_id;
    uint16_t src_mesh_id = response->src_mesh_id;
    uint8_t src_noc_x = response->src_noc_x;
    uint8_t src_noc_y = response->src_noc_y;
    uint8_t src_risc_id = response->risc_id;

    uint32_t counter_addr = get_fabric_counter_address<BarrierType>(src_risc_id);

    ack_header->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader(get_noc_addr(src_noc_x, src_noc_y, counter_addr), increment_value, flush));
    fabric_write_set_unicast_route(ack_header, src_chip_id, src_mesh_id, 0, 1);  // trid=0, posted=1

    connection.send_payload_blocking_from_address(reinterpret_cast<uint32_t>(ack_header), sizeof(PACKET_HEADER_TYPE));
}

/**
 * @brief Send a write acknowledgment back to the sender
 *
 * Wrapper function that calls fabric_fast_ack with NONPOSTED_WRITES_ACKED barrier type.
 * Accepts a RegisteredResponse containing sender information and a pre-selected mux connection.
 *
 * @tparam FabricConnectionType The connection type
 * @param connection Single mux connection (already selected by caller)
 * @param response Pointer to the registered response containing sender information
 * @param increment_value The value to increment the counter by (default 1)
 * @param flush Whether to flush the atomic operation (default false)
 */
template <typename FabricConnectionType>
FORCE_INLINE void fabric_fast_write_ack(
    FabricConnectionType& connection,
    volatile RegisteredResponse* response,
    uint32_t increment_value = 1,
    bool flush = false) {
    fabric_fast_ack<FabricBarrierType::NONPOSTED_WRITES_ACKED>(connection, response, increment_value, flush);
}

/**
 * @brief Send an atomic increment acknowledgment back to the sender
 *
 * Wrapper function that calls fabric_fast_ack with NONPOSTED_ATOMICS_ACKED barrier type.
 * Accepts a RegisteredResponse containing sender information and a pre-selected mux connection.
 *
 * @tparam FabricConnectionType The connection type
 * @param connection Single mux connection (already selected by caller)
 * @param response Pointer to the registered response containing sender information
 * @param increment_value The value to increment the counter by (default 1)
 * @param flush Whether to flush the atomic operation (default false)
 */
template <typename FabricConnectionType>
FORCE_INLINE void fabric_fast_atomic_ack(
    FabricConnectionType& connection,
    volatile RegisteredResponse* response,
    uint32_t increment_value = 1,
    bool flush = false) {
    fabric_fast_ack<FabricBarrierType::NONPOSTED_ATOMICS_ACKED>(connection, response, increment_value, flush);
}

/**
 * @brief Initiate a read request through the fabric for any length
 *
 * This function sends a single read request packet through the fabric network,
 * regardless of the data size. The receiver is responsible for breaking up the
 * data into appropriate chunks if it exceeds the maximum fabric payload size.
 *
 * @param dst_dev_id Destination device ID for fabric routing
 * @param dst_mesh_id Destination mesh ID for fabric routing
 * @param dest_addr Remote NOC address to read from (encoded with x,y coordinates and local address)
 * @param src_l1_addr Local L1 memory address where the data should be written when received
 * @param size_bytes Total number of bytes to read (can be any size)
 * @param trid Transaction ID for tracking the read operation
 */
FORCE_INLINE void fabric_fast_read_any_len(
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint64_t dest_addr,
    uint32_t src_l1_addr,
    uint32_t size_bytes,
    uint8_t trid = 0) {
    auto& connection = get_or_open_fabric_connection();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = get_or_allocate_header();

    fabric_read_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id, src_l1_addr, size_bytes, trid);
    packet_header->to_noc_unicast_read(NocUnicastCommandHeader{dest_addr}, 0);
    connection.wait_for_empty_write_slot();
    connection.send_payload_flush_non_blocking_from_address(
        reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));

    fabric_reads_num_acked++;
    noc_async_writes_flushed();

    release_fabric_connection();
}

/**
 * @brief Send ONE read response slot back to the requester
 *
 * Sends a single slot from the memory pool to the requester. Automatically uses
 * fused atomic increment on the last packet (detected via response->is_last_slot()).
 *
 * @tparam FabricConnectionType The connection type
 * @tparam UDMMemoryPoolType The memory pool type
 * @param connection Single mux connection (already selected by caller)
 * @param response Pointer to the registered response containing sender information
 * @param memory_pool Reference to the UDM memory pool instance
 */
template <typename FabricConnectionType, typename UDMMemoryPoolType>
FORCE_INLINE void fabric_fast_read_ack(
    FabricConnectionType& connection, volatile RegisteredResponse* response, UDMMemoryPoolType& memory_pool) {
    ASSERT(connection.edm_has_space_for_packet());

    uint8_t src_chip_id = response->src_chip_id;
    uint16_t src_mesh_id = response->src_mesh_id;
    uint8_t src_noc_x = response->src_noc_x;
    uint8_t src_noc_y = response->src_noc_y;
    uint8_t src_risc_id = response->risc_id;
    uint8_t transaction_id = response->transaction_id;
    uint32_t src_l1_address = response->src_l1_address;

    volatile tt_l1_ptr PACKET_HEADER_TYPE* response_header = get_or_allocate_header();

    // Set up routing
    fabric_write_set_unicast_route(response_header, src_chip_id, src_mesh_id, transaction_id, 1 /* posted */);

    // Get source address from memory pool and destination address from response
    uint32_t src_addr = memory_pool.get_slot_addr(memory_pool.get_rd_slot_idx());
    uint64_t dest_addr = get_noc_addr(src_noc_x, src_noc_y, src_l1_address);
    uint32_t slot_size = memory_pool.get_slot_size();

    // Bytes to send: min of bytes in pool and slot size (handles partial last slot)
    uint32_t bytes_to_send = std::min((uint32_t)response->bytes_remaining, slot_size);

    if (response->is_last_send(slot_size)) {
        // Last packet - use fused atomic increment
        uint32_t counter_addr = get_fabric_counter_address<FabricBarrierType::READS_NUM_ACKED>(src_risc_id);
        uint64_t counter_noc_addr = get_noc_addr(src_noc_x, src_noc_y, counter_addr);
        response_header->to_noc_fused_unicast_write_atomic_inc(
            NocUnicastAtomicIncFusedCommandHeader(dest_addr, counter_noc_addr, 1, true /* flush */), bytes_to_send);
    } else {
        response_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_addr}, bytes_to_send);
    }

    connection.send_payload_without_header_non_blocking_from_address(src_addr, bytes_to_send);
    // Use blocking mode to barrier before issue the credits to the receiver
    connection.send_payload_blocking_from_address(
        reinterpret_cast<uint32_t>(response_header), sizeof(PACKET_HEADER_TYPE));
}

}  // namespace tt::tt_fabric::udm
