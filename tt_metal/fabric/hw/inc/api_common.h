// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

namespace tt::tt_fabric::common::experimental {

// clang-format off
/**
 * Field update masks for Linear Fabric APIs (compile-time bitfields).
 * These flags are used with the *_set_state and *_with_state helpers to control
 * which fields of the packet header should be updated at submission time.
 */
// clang-format on
// Field update masks for linear fabric APIs (compile-time bitfields)
enum class UnicastWriteUpdateMask : uint32_t {
    None = 0,
    DstAddr = 1u << 0,
    PayloadSize = 1u << 1,
};

enum class UnicastInlineWriteUpdateMask : uint32_t {
    None = 0,
    DstAddr = 1u << 0,
    Value = 1u << 1,
};

enum class UnicastAtomicIncUpdateMask : uint32_t {
    None = 0,
    DstAddr = 1u << 0,
    Wrap = 1u << 1,
    Val = 1u << 2,
    Flush = 1u << 3,
};

// Scatter write dynamic mask (coarse-grained)
enum class UnicastScatterWriteUpdateMask : uint32_t {
    None = 0,
    DstAddrs = 1u << 0,    // update all noc_address[i]
    ChunkSizes = 1u << 1,  // update all chunk_size[i]
    PayloadSize = 1u << 2,
};

// Fused write+atomic inc dynamic mask
enum class UnicastFusedAtomicIncUpdateMask : uint32_t {
    None = 0,
    WriteDstAddr = 1u << 0,
    SemaphoreAddr = 1u << 1,
    Wrap = 1u << 2,
    Val = 1u << 3,
    Flush = 1u << 4,
    PayloadSize = 1u << 5,
};

// Bitwise helpers for enum class flags
constexpr inline UnicastWriteUpdateMask operator|(UnicastWriteUpdateMask a, UnicastWriteUpdateMask b) {
    return static_cast<UnicastWriteUpdateMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastWriteUpdateMask operator&(UnicastWriteUpdateMask a, UnicastWriteUpdateMask b) {
    return static_cast<UnicastWriteUpdateMask>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastWriteUpdateMask mask, UnicastWriteUpdateMask bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

constexpr inline UnicastInlineWriteUpdateMask operator|(
    UnicastInlineWriteUpdateMask a, UnicastInlineWriteUpdateMask b) {
    return static_cast<UnicastInlineWriteUpdateMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastInlineWriteUpdateMask operator&(
    UnicastInlineWriteUpdateMask a, UnicastInlineWriteUpdateMask b) {
    return static_cast<UnicastInlineWriteUpdateMask>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastInlineWriteUpdateMask mask, UnicastInlineWriteUpdateMask bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

constexpr inline UnicastAtomicIncUpdateMask operator|(UnicastAtomicIncUpdateMask a, UnicastAtomicIncUpdateMask b) {
    return static_cast<UnicastAtomicIncUpdateMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastAtomicIncUpdateMask operator&(UnicastAtomicIncUpdateMask a, UnicastAtomicIncUpdateMask b) {
    return static_cast<UnicastAtomicIncUpdateMask>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastAtomicIncUpdateMask mask, UnicastAtomicIncUpdateMask bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

constexpr inline UnicastScatterWriteUpdateMask operator|(
    UnicastScatterWriteUpdateMask a, UnicastScatterWriteUpdateMask b) {
    return static_cast<UnicastScatterWriteUpdateMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastScatterWriteUpdateMask operator&(
    UnicastScatterWriteUpdateMask a, UnicastScatterWriteUpdateMask b) {
    return static_cast<UnicastScatterWriteUpdateMask>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastScatterWriteUpdateMask mask, UnicastScatterWriteUpdateMask bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

constexpr inline UnicastFusedAtomicIncUpdateMask operator|(
    UnicastFusedAtomicIncUpdateMask a, UnicastFusedAtomicIncUpdateMask b) {
    return static_cast<UnicastFusedAtomicIncUpdateMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastFusedAtomicIncUpdateMask operator&(
    UnicastFusedAtomicIncUpdateMask a, UnicastFusedAtomicIncUpdateMask b) {
    return static_cast<UnicastFusedAtomicIncUpdateMask>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastFusedAtomicIncUpdateMask mask, UnicastFusedAtomicIncUpdateMask bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

// ========================
// Fabric Sender Type Traits
// ========================

// Type trait to detect if a type is a WorkerToFabricMuxSender
template <typename T>
struct is_mux_sender : std::false_type {};

template <uint8_t N>
struct is_mux_sender<tt::tt_fabric::WorkerToFabricMuxSender<N>> : std::true_type {};

template <typename T>
constexpr bool is_mux_sender_v = is_mux_sender<T>::value;

// Type trait to detect if a type is a WorkerToFabricEdmSender
template <typename T>
struct is_edm_sender : std::false_type {};

template <>
struct is_edm_sender<tt::tt_fabric::WorkerToFabricEdmSender> : std::true_type {};

template <typename T>
constexpr bool is_edm_sender_v = is_edm_sender<T>::value;

template <typename FabricSenderType>
struct CheckFabricSenderType {
    static_assert(
        is_edm_sender_v<FabricSenderType> || is_mux_sender_v<FabricSenderType>,
        "FabricSenderType must be WorkerToFabricEdmSender or WorkerToFabricMuxSender");
};

// ========================
// Common populate helpers
// ========================

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_write_fields(
    volatile PACKET_HEADER_TYPE* packet_header, uint16_t packet_size_bytes, const CommandHeaderT& command_header) {
    if constexpr (has_flag(UpdateMask, UnicastWriteUpdateMask::DstAddr)) {
        auto comps = get_noc_address_components(command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_write.noc_address = noc_addr;
    }
    if constexpr (has_flag(UpdateMask, UnicastWriteUpdateMask::PayloadSize)) {
        packet_header->payload_size_bytes = packet_size_bytes;
    }
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_inline_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& command_header) {
    if constexpr (has_flag(UpdateMask, UnicastInlineWriteUpdateMask::DstAddr)) {
        auto comps = get_noc_address_components(command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_inline_write.noc_address = noc_addr;
    }
    if constexpr (has_flag(UpdateMask, UnicastInlineWriteUpdateMask::Value)) {
        packet_header->command_fields.unicast_inline_write.value = command_header.value;
    }
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_atomic_inc_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& command_header) {
    if constexpr (has_flag(UpdateMask, UnicastAtomicIncUpdateMask::DstAddr)) {
        auto comps = get_noc_address_components(command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_seminc.noc_address = noc_addr;
    }
    if constexpr (has_flag(UpdateMask, UnicastAtomicIncUpdateMask::Wrap)) {
        packet_header->command_fields.unicast_seminc.wrap = command_header.wrap;
    }
    if constexpr (has_flag(UpdateMask, UnicastAtomicIncUpdateMask::Val)) {
        packet_header->command_fields.unicast_seminc.val = command_header.val;
    }
    if constexpr (has_flag(UpdateMask, UnicastAtomicIncUpdateMask::Flush)) {
        packet_header->command_fields.unicast_seminc.flush = command_header.flush;
    }
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_scatter_write_fields(
    volatile PACKET_HEADER_TYPE* packet_header, uint16_t packet_size_bytes, const CommandHeaderT& command_header) {
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::DstAddrs)) {
        for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS; i++) {
            auto comps = get_noc_address_components(command_header.noc_address[i]);
            auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
            packet_header->command_fields.unicast_scatter_write.noc_address[i] = noc_addr;
        }
    }
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::ChunkSizes)) {
        for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS - 1; i++) {
            packet_header->command_fields.unicast_scatter_write.chunk_size[i] = command_header.chunk_size[i];
        }
    }
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::PayloadSize)) {
        packet_header->payload_size_bytes = packet_size_bytes;
    }
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_fused_atomic_inc_fields(
    volatile PACKET_HEADER_TYPE* packet_header, uint16_t packet_size_bytes, const CommandHeaderT& command_header) {
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::WriteDstAddr)) {
        auto comps = get_noc_address_components(command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_seminc_fused.noc_address = noc_addr;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::SemaphoreAddr)) {
        auto scomps = get_noc_address_components(command_header.semaphore_noc_address);
        auto snoc = safe_get_noc_addr(scomps.first.x, scomps.first.y, scomps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_seminc_fused.semaphore_noc_address = snoc;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::Wrap)) {
        packet_header->command_fields.unicast_seminc_fused.wrap = command_header.wrap;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::Val)) {
        packet_header->command_fields.unicast_seminc_fused.val = command_header.val;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::Flush)) {
        packet_header->command_fields.unicast_seminc_fused.flush = command_header.flush;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::PayloadSize)) {
        packet_header->payload_size_bytes = packet_size_bytes;
    }
}

// clang-format off
/**
 * Opens fabric routing-plane connections for all headers associated with the given route.
 * Reads connection parameters from runtime arguments and initializes headers with routing metadata.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Connection manager to build and open    | RoutingPlaneConnectionManager&                 | True     |
 * | num_connections_to_build              | Number of connections to build/open     | uint32_t                                       | True     |
 * | rt_arg_idx                            | Runtime-args cursor (advanced as parsed)| size_t&                                        | True     |
 */
// clang-format on
FORCE_INLINE void open_connections(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint32_t num_connections_to_build,
    size_t& rt_arg_idx) {
    connection_manager = tt::tt_fabric::RoutingPlaneConnectionManager::template build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION>(rt_arg_idx, num_connections_to_build);
}

// clang-format off
/**
 * Closes all connections owned by the provided connection manager.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Connection manager to be closed         | RoutingPlaneConnectionManager&                 | True     |
 */
// clang-format on
FORCE_INLINE void close_connections(tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager) {
    connection_manager.close();
}

}  // namespace tt::tt_fabric::common::experimental
