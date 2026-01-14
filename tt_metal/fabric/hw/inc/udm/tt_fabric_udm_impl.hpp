// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file contains implementation details and types for UDM fabric functions.
// It is included at the top of tt_fabric_udm.h

// Required headers for the implementations
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "dev_mem_map.h"
#include <type_traits>

namespace tt::tt_fabric::udm {

// Verify that connection object fits in reserved L1 storage
static_assert(
    sizeof(tt::tt_fabric::WorkerToFabricEdmSender) <= tt::tt_fabric::FABRIC_CONNECTION_OBJECT_SIZE,
    "WorkerToFabricEdmSender is too large for reserved L1 storage");

// Acquire lock using atomic exchange - spin until we get it
// On Wormhole, this is a no-op
FORCE_INLINE void acquire_lock(volatile uint32_t* lock) {
#ifndef ARCH_WORMHOLE
    while (__atomic_exchange_n(lock, 1, __ATOMIC_ACQUIRE) != 0) {
        // Spin waiting for lock to become available
    }
#endif  // ARCH_WORMHOLE
}

// Release lock using atomic store
// On Wormhole, this is a no-op
FORCE_INLINE void release_lock(volatile uint32_t* lock) {
#ifndef ARCH_WORMHOLE
    __atomic_store_n(lock, 0, __ATOMIC_RELEASE);
#endif  // ARCH_WORMHOLE
}

/**
 * @brief Enum for UDM control fields
 */
enum class UDM_CONTROL_FIELD {
    POSTED,
    SRC_CHIP_ID,
    SRC_MESH_ID,
    SRC_NOC_X,
    SRC_NOC_Y,
    RISC_ID,
    TRANSACTION_ID,
    INITIAL_DIRECTION
};

/**
 * @brief UDM (Unified Data Movement) fields for fabric write operations
 *
 * Contains the source information needed for UDM write operations:
 * - src_noc_x: Source NOC X coordinate
 * - src_noc_y: Source NOC Y coordinate
 * - risc_id: Processor ID (BRISC, NCRISC, etc.)
 * - trid: Transaction ID for tracking operations
 * - posted: Whether this is a posted write (1) or non-posted write (0)
 */
struct udm_write_fields {
    uint8_t src_noc_x;
    uint8_t src_noc_y;
    uint8_t risc_id;
    uint8_t trid;
    uint8_t posted;
};

/**
 * @brief UDM fields for fabric read operations
 *
 * Contains the source information needed for UDM read operations:
 * - src_noc_x: Source NOC X coordinate
 * - src_noc_y: Source NOC Y coordinate
 * - src_l1_address: Source L1 memory address where data should be written
 * - size_bytes: Number of bytes to read
 * - risc_id: Processor ID (BRISC, NCRISC, etc.)
 * - trid: Transaction ID for tracking operations
 */
struct udm_read_fields {
    uint8_t src_noc_x;
    uint8_t src_noc_y;
    uint32_t src_l1_address;
    uint32_t size_bytes;
    uint8_t risc_id;
    uint8_t trid;
};

/**
 * @brief Calculate the initial direction for a packet based on routing path
 *
 * Determines which direction (EAST, WEST, NORTH, SOUTH) a packet should initially
 * take to reach the destination chip in a 2D mesh topology.
 *
 * @param dst_chip_id Destination chip ID
 * @param my_chip_id Source chip ID
 * @return Initial direction as uint32_t (eth_chan_directions enum value)
 */
FORCE_INLINE uint32_t calculate_initial_direction(uint16_t dst_chip_id, uint16_t my_chip_id) {
    auto* routing_info = reinterpret_cast<tt_l1_ptr intra_mesh_routing_path_t<2, true>*>(ROUTING_PATH_BASE_2D);

    uint32_t initial_dir = static_cast<uint32_t>(eth_chan_directions::EAST);

    const auto& compressed_route = routing_info->paths[dst_chip_id];
    uint8_t ns_hops = compressed_route.get_ns_hops();
    uint8_t ew_hops = compressed_route.get_ew_hops();

    if (ns_hops > 0) {
        // is there another way to know whether it's north or south hops?
        if (dst_chip_id < my_chip_id) {
            initial_dir = static_cast<uint32_t>(eth_chan_directions::NORTH);
        } else {
            initial_dir = static_cast<uint32_t>(eth_chan_directions::SOUTH);
        }
    } else if (ew_hops > 0) {
        // is there another way to know whether it's east or west hops?
        if (dst_chip_id < my_chip_id) {
            initial_dir = static_cast<uint32_t>(eth_chan_directions::WEST);
        } else {
            initial_dir = static_cast<uint32_t>(eth_chan_directions::EAST);
        }
    }

    return initial_dir;
}

/**
 * @brief Set unicast route for UDMHybridMeshPacketHeader with UDM fields
 *
 * Configures routing and UDM control fields for hybrid mesh packet headers.
 * This overload handles 2D mesh routing with UDM-specific metadata.
 *
 * @param packet_header Pointer to the UDM hybrid mesh packet header
 * @param udm UDM fields containing source information (including posted flag)
 * @param dst_dev_id Destination device ID
 * @param dst_mesh_id Destination mesh ID
 * @return true if route was successfully set, false otherwise
 */
FORCE_INLINE bool fabric_write_set_unicast_route_impl(
    volatile tt_l1_ptr UDMHybridMeshPacketHeader* packet_header,
    udm_write_fields& udm,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id) {
    tt_l1_ptr routing_l1_info_t* routing_table =
        reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(MEM_TENSIX_ROUTING_TABLE_BASE);
    uint16_t my_chip_id = routing_table->my_device_id;
    uint16_t my_mesh_id = routing_table->my_mesh_id;

    ASSERT(my_mesh_id == dst_mesh_id);  // we dont support inter-mesh for UDM mode yet

    // Calculate initial direction based on routing path
    uint32_t initial_dir = calculate_initial_direction(dst_dev_id, my_chip_id);

    // First call the original API by casting to base type
    bool result = fabric_set_unicast_route(
        static_cast<volatile tt_l1_ptr HybridMeshPacketHeader*>(packet_header), dst_dev_id, dst_mesh_id);

    // Set UDM control fields for write operations using routing table info
    packet_header->udm_control.write.src_chip_id = my_chip_id;
    packet_header->udm_control.write.src_mesh_id = my_mesh_id;
    packet_header->udm_control.write.src_noc_x = udm.src_noc_x;
    packet_header->udm_control.write.src_noc_y = udm.src_noc_y;
    packet_header->udm_control.write.risc_id = udm.risc_id;
    packet_header->udm_control.write.transaction_id = udm.trid;
    packet_header->udm_control.write.posted = udm.posted;
    packet_header->udm_control.write.initial_direction = initial_dir;
    return result;
}

/**
 * @brief Helper function to set unicast routing based on packet header type
 *
 * This function provides a unified interface for setting up fabric unicast routing.
 * It creates the necessary UDM fields and calls the appropriate routing function
 * based on the packet header type defined at compile time.
 *
 * @param packet_header Pointer to the packet header
 * @param dst_dev_id Destination device ID
 * @param dst_mesh_id Destination mesh ID
 * @param trid Transaction ID for UDM operations
 * @param posted Whether this is a posted write (1) or non-posted write (0, default)
 */
template <typename T = PACKET_HEADER_TYPE>
FORCE_INLINE void fabric_write_set_unicast_route(
    volatile tt_l1_ptr T* packet_header, uint16_t dst_dev_id, uint16_t dst_mesh_id, uint8_t trid, uint8_t posted) {
    udm_write_fields udm = {my_x[edm_to_local_chip_noc], my_y[edm_to_local_chip_noc], proc_type, trid, posted};

    if constexpr (std::is_same_v<T, tt::tt_fabric::UDMHybridMeshPacketHeader>) {
        fabric_write_set_unicast_route_impl(packet_header, udm, dst_dev_id, dst_mesh_id);
    } else {
        // Compile error for unsupported types
        static_assert(
            std::is_same_v<T, tt::tt_fabric::UDMHybridMeshPacketHeader>,
            "Unsupported packet header type for fabric_write_set_unicast_route - only UDMHybridMeshPacketHeader is "
            "supported");
    }
}

/**
 * @brief Set unicast route for UDMHybridMeshPacketHeader with UDM read fields
 *
 * Configures routing and UDM control fields for hybrid mesh packet headers for read operations.
 * This overload handles 2D mesh routing with UDM-specific read metadata.
 *
 * @param packet_header Pointer to the UDM hybrid mesh packet header
 * @param udm UDM fields containing source information for read operation
 * @param dst_dev_id Destination device ID
 * @param dst_mesh_id Destination mesh ID
 * @return true if route was successfully set, false otherwise
 */
FORCE_INLINE bool fabric_read_set_unicast_route_impl(
    volatile tt_l1_ptr UDMHybridMeshPacketHeader* packet_header,
    udm_read_fields& udm,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id) {
    tt_l1_ptr routing_l1_info_t* routing_table =
        reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(MEM_TENSIX_ROUTING_TABLE_BASE);
    uint16_t my_chip_id = routing_table->my_device_id;
    uint16_t my_mesh_id = routing_table->my_mesh_id;

    ASSERT(my_mesh_id == dst_mesh_id);  // we dont support inter-mesh for UDM mode yet

    // Calculate initial direction based on routing path
    uint32_t initial_dir = calculate_initial_direction(dst_dev_id, my_chip_id);

    // First call the original API by casting to base type
    bool result = fabric_set_unicast_route(
        static_cast<volatile tt_l1_ptr HybridMeshPacketHeader*>(packet_header), dst_dev_id, dst_mesh_id);

    // Set UDM control fields for read operations using routing table info
    packet_header->udm_control.read.src_chip_id = my_chip_id;
    packet_header->udm_control.read.src_mesh_id = my_mesh_id;
    packet_header->udm_control.read.src_noc_x = udm.src_noc_x;
    packet_header->udm_control.read.src_noc_y = udm.src_noc_y;
    packet_header->udm_control.read.src_l1_address = udm.src_l1_address;
    packet_header->udm_control.read.size_bytes = udm.size_bytes;
    packet_header->udm_control.read.risc_id = udm.risc_id;
    packet_header->udm_control.read.transaction_id = udm.trid;
    packet_header->udm_control.read.initial_direction = initial_dir;

    return result;
}

/**
 * @brief Helper function to set unicast routing for read operations based on packet header type
 *
 * This function provides a unified interface for setting up fabric unicast routing for read operations.
 * It creates the necessary UDM fields and calls the appropriate routing function
 * based on the packet header type defined at compile time.
 *
 * @param packet_header Pointer to the packet header
 * @param dst_dev_id Destination device ID
 * @param dst_mesh_id Destination mesh ID
 * @param src_l1_addr Source L1 memory address where data should be written
 * @param size_bytes Number of bytes to read
 * @param trid Transaction ID for UDM operations
 */
template <typename T = PACKET_HEADER_TYPE>
FORCE_INLINE void fabric_read_set_unicast_route(
    volatile tt_l1_ptr T* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_l1_addr,
    uint32_t size_bytes,
    uint8_t trid) {
    udm_read_fields udm = {
        my_x[edm_to_local_chip_noc], my_y[edm_to_local_chip_noc], src_l1_addr, size_bytes, proc_type, trid};

    if constexpr (std::is_same_v<T, tt::tt_fabric::UDMHybridMeshPacketHeader>) {
        fabric_read_set_unicast_route_impl(packet_header, udm, dst_dev_id, dst_mesh_id);
    } else {
        // Compile error for unsupported types
        static_assert(
            std::is_same_v<T, tt::tt_fabric::UDMHybridMeshPacketHeader>,
            "Unsupported packet header type for fabric_read_set_unicast_route - only UDMHybridMeshPacketHeader is "
            "supported");
    }
}

/**
 * @brief Set individual UDM control fields in the packet header
 *
 * This template function allows setting specific UDM control fields individually
 * without affecting other fields. It provides fine-grained control over the
 * packet header configuration.
 *
 * @tparam Field The control field to set (from UDM_CONTROL_FIELD enum)
 * @tparam T The packet header type (auto-deduced)
 * @tparam V The value type (auto-deduced)
 * @param packet_header Pointer to the packet header (UDMHybridMeshPacketHeader)
 * @param value The value to set for the specified field
 *
 */
template <UDM_CONTROL_FIELD Field, typename T, typename V>
FORCE_INLINE void fabric_write_set_unicast_route_control_field(volatile tt_l1_ptr T* packet_header, V value) {
    // Ensure the packet header type has UDM control fields
    static_assert(
        std::is_same_v<T, UDMHybridMeshPacketHeader> || std::is_same_v<T, PACKET_HEADER_TYPE>,
        "Packet header must be UDMHybridMeshPacketHeader");

    if constexpr (Field == UDM_CONTROL_FIELD::POSTED) {
        packet_header->udm_control.write.posted = static_cast<uint8_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::SRC_CHIP_ID) {
        packet_header->udm_control.write.src_chip_id = static_cast<uint8_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::SRC_MESH_ID) {
        packet_header->udm_control.write.src_mesh_id = static_cast<uint16_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::SRC_NOC_X) {
        packet_header->udm_control.write.src_noc_x = static_cast<uint8_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::SRC_NOC_Y) {
        packet_header->udm_control.write.src_noc_y = static_cast<uint8_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::RISC_ID) {
        packet_header->udm_control.write.risc_id = static_cast<uint8_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::TRANSACTION_ID) {
        packet_header->udm_control.write.transaction_id = static_cast<uint8_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::INITIAL_DIRECTION) {
        packet_header->udm_control.write.initial_direction = static_cast<uint8_t>(value);
    }
}

WorkerToFabricEdmSender build_from_reserved_l1_info() {
    constexpr bool is_persistent_fabric = true;
    const StreamId my_fc_stream_channel_id = StreamId{std::numeric_limits<uint32_t>::max()};

    tt_l1_ptr tensix_fabric_connections_l1_info_t* connection_info =
        reinterpret_cast<tt_l1_ptr tensix_fabric_connections_l1_info_t*>(MEM_TENSIX_FABRIC_CONNECTIONS_BASE);
    constexpr uint32_t eth_channel = 0;  // always use channel 0 for UDM mode
    const auto conn = &connection_info->read_only[eth_channel];
    const auto aligned_conn = &connection_info->read_write[eth_channel];
    uint8_t direction = conn->edm_direction;
    uint8_t edm_worker_x = conn->edm_noc_x;
    uint8_t edm_worker_y = conn->edm_noc_y;
    uint32_t edm_buffer_base_addr = conn->edm_buffer_base_addr;
    uint8_t num_buffers_per_channel = conn->num_buffers_per_channel;
    uint32_t edm_connection_handshake_l1_addr = conn->edm_connection_handshake_addr;
    uint32_t edm_worker_location_info_addr = conn->edm_worker_location_info_addr;
    uint16_t buffer_size_bytes = conn->buffer_size_bytes;
    uint32_t edm_copy_of_wr_counter_addr = conn->buffer_index_semaphore_id;
    volatile uint32_t* writer_send_sem_addr =
        reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uintptr_t>(&aligned_conn->worker_flow_control_semaphore));
    uint32_t worker_free_slots_stream_id = static_cast<uint32_t>(conn->worker_free_slots_stream_id);

    // Use second region for worker_teardown_sem_addr
    constexpr uint32_t eth_channel_teardown = 1;
    static_assert(
        eth_channel_teardown < tensix_fabric_connections_l1_info_t::MAX_FABRIC_ENDPOINTS,
        "eth_channel_teardown out of bounds");
    const auto aligned_conn_teardown = &connection_info->read_write[eth_channel_teardown];
    volatile uint32_t* worker_teardown_sem_addr = reinterpret_cast<volatile uint32_t*>(
        reinterpret_cast<uintptr_t>(&aligned_conn_teardown->worker_flow_control_semaphore));

    // Use third region for worker_buffer_index_semaphore_addr
    constexpr uint32_t eth_channel_buffer_index = 2;
    static_assert(
        eth_channel_buffer_index < tensix_fabric_connections_l1_info_t::MAX_FABRIC_ENDPOINTS,
        "eth_channel_buffer_index out of bounds");
    const auto aligned_conn_buffer_index = &connection_info->read_write[eth_channel_buffer_index];
    uint32_t worker_buffer_index_semaphore_addr =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&aligned_conn_buffer_index->worker_flow_control_semaphore));

    return WorkerToFabricEdmSender(
        is_persistent_fabric,
        edm_worker_x,
        edm_worker_y,
        edm_buffer_base_addr,
        num_buffers_per_channel,
        edm_connection_handshake_l1_addr,
        edm_worker_location_info_addr,
        buffer_size_bytes,
        edm_copy_of_wr_counter_addr,
        writer_send_sem_addr,
        worker_teardown_sem_addr,
        worker_buffer_index_semaphore_addr,
        worker_free_slots_stream_id,
        my_fc_stream_channel_id);
}

/**
 * @brief Get the sync region for fabric connection state
 *
 * Uses dedicated MEM_FABRIC_CONNECTION_LOCK_BASE address in L1.
 * The region contains a spinlock and initialized flag.
 *
 * @return Pointer to the sync region in L1
 */
FORCE_INLINE volatile tt::tt_fabric::fabric_connection_sync_t* get_fabric_connection_sync() {
    return reinterpret_cast<volatile tt::tt_fabric::fabric_connection_sync_t*>(MEM_FABRIC_CONNECTION_LOCK_BASE);
}

/**
 * @brief Get or create a singleton fabric connection for the current worker
 *
 * This function manages a static connection instance that persists across calls.
 * The connection is initialized on first use and reused for subsequent operations.
 * On non-Wormhole, uses atomic spinlock to synchronize access across BRISC and NCRISC.
 * The lock is acquired on EVERY call and must be released via release_fabric_connection() after each use.
 * The initialized flag is stored in L1 (shared across RISCs on multi-RISC architectures).
 *
 * @return Reference to the fabric connection
 */
FORCE_INLINE tt::tt_fabric::WorkerToFabricEdmSender& get_or_open_fabric_connection() {
    // Get sync region pointer (just address calculation, no memory read)
    auto* sync = get_fabric_connection_sync();

#ifndef ARCH_WORMHOLE
    // Lock must be acquired BEFORE checking initialized to ensure mutual exclusion
    acquire_lock(&sync->lock);
#endif

    // Get connection pointer from L1 storage (at fixed offset after sync struct)
    auto* connection = reinterpret_cast<tt::tt_fabric::WorkerToFabricEdmSender*>(
        MEM_FABRIC_CONNECTION_LOCK_BASE + tt::tt_fabric::FABRIC_CONNECTION_OBJECT_OFFSET);

    if (!sync->initialized) {
        // Build connection in L1 storage using placement new
        new (connection) tt::tt_fabric::WorkerToFabricEdmSender(build_from_reserved_l1_info());
        connection->open();
        sync->initialized = 1;
    }

    return *connection;
}

/**
 * @brief Release the fabric connection lock
 *
 * MUST be called after EVERY call to get_or_open_fabric_connection() to release the lock.
 * Failure to call this will cause deadlocks on subsequent calls.
 * On Wormhole, this is a no-op.
 */
FORCE_INLINE void release_fabric_connection() {
#ifndef ARCH_WORMHOLE
    auto* sync = get_fabric_connection_sync();
    release_lock(&sync->lock);
#endif
}

/**
 * @brief Get or allocate a packet header for UDM operations
 *
 * Returns a singleton packet header for the current processor.
 * If a header has already been allocated, returns the existing one.
 * Otherwise allocates a new header from the PacketHeaderPool.
 *
 * This ensures we reuse the same header location for all operations,
 * similar to the get_fabric_connection pattern. This is more efficient
 * than allocating and releasing headers for each operation.
 *
 * @return Pointer to the allocated packet header
 */
FORCE_INLINE volatile tt_l1_ptr PACKET_HEADER_TYPE* get_or_allocate_header() {
    static volatile tt_l1_ptr PACKET_HEADER_TYPE* singleton_header = nullptr;
    if (singleton_header == nullptr) {
        singleton_header = PacketHeaderPool::allocate_header();
    }
    return singleton_header;
}

}  // namespace tt::tt_fabric::udm
