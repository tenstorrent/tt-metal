// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

namespace tt::tt_fabric::linear::experimental {

// Dynamic field masks for linear fabric APIs (compile-time bitfields)
enum class UnicastWriteDynamic : uint32_t {
    None = 0,
    DstAddr = 1u << 0,
    PayloadSize = 1u << 1,
};

enum class UnicastInlineWriteDynamic : uint32_t {
    None = 0,
    DstAddr = 1u << 0,
    Value = 1u << 1,
};

enum class UnicastAtomicIncDynamic : uint32_t {
    None = 0,
    DstAddr = 1u << 0,
    Wrap = 1u << 1,
    Val = 1u << 2,
    Flush = 1u << 3,
};

// Scatter write dynamic mask (coarse-grained)
enum class UnicastScatterWriteDynamic : uint32_t {
    None = 0,
    DstAddrs = 1u << 0,    // update all noc_address[i]
    ChunkSizes = 1u << 1,  // update all chunk_size[i]
    PayloadSize = 1u << 2,
};

// Fused write+atomic inc dynamic mask
enum class UnicastFusedAtomicIncDynamic : uint32_t {
    None = 0,
    WriteDstAddr = 1u << 0,
    SemaphoreAddr = 1u << 1,
    Wrap = 1u << 2,
    Val = 1u << 3,
    Flush = 1u << 4,
    PayloadSize = 1u << 5,
};

// Bitwise helpers for enum class flags
constexpr inline UnicastWriteDynamic operator|(UnicastWriteDynamic a, UnicastWriteDynamic b) {
    return static_cast<UnicastWriteDynamic>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastWriteDynamic operator&(UnicastWriteDynamic a, UnicastWriteDynamic b) {
    return static_cast<UnicastWriteDynamic>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastWriteDynamic mask, UnicastWriteDynamic bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

constexpr inline UnicastInlineWriteDynamic operator|(UnicastInlineWriteDynamic a, UnicastInlineWriteDynamic b) {
    return static_cast<UnicastInlineWriteDynamic>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastInlineWriteDynamic operator&(UnicastInlineWriteDynamic a, UnicastInlineWriteDynamic b) {
    return static_cast<UnicastInlineWriteDynamic>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastInlineWriteDynamic mask, UnicastInlineWriteDynamic bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

constexpr inline UnicastAtomicIncDynamic operator|(UnicastAtomicIncDynamic a, UnicastAtomicIncDynamic b) {
    return static_cast<UnicastAtomicIncDynamic>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastAtomicIncDynamic operator&(UnicastAtomicIncDynamic a, UnicastAtomicIncDynamic b) {
    return static_cast<UnicastAtomicIncDynamic>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastAtomicIncDynamic mask, UnicastAtomicIncDynamic bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

constexpr inline UnicastScatterWriteDynamic operator|(UnicastScatterWriteDynamic a, UnicastScatterWriteDynamic b) {
    return static_cast<UnicastScatterWriteDynamic>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastScatterWriteDynamic operator&(UnicastScatterWriteDynamic a, UnicastScatterWriteDynamic b) {
    return static_cast<UnicastScatterWriteDynamic>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastScatterWriteDynamic mask, UnicastScatterWriteDynamic bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

constexpr inline UnicastFusedAtomicIncDynamic operator|(
    UnicastFusedAtomicIncDynamic a, UnicastFusedAtomicIncDynamic b) {
    return static_cast<UnicastFusedAtomicIncDynamic>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastFusedAtomicIncDynamic operator&(
    UnicastFusedAtomicIncDynamic a, UnicastFusedAtomicIncDynamic b) {
    return static_cast<UnicastFusedAtomicIncDynamic>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastFusedAtomicIncDynamic mask, UnicastFusedAtomicIncDynamic bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

// ========================
// Common populate helpers
// ========================

template <UnicastWriteDynamic DynamicMask, uint32_t PACKET_SIZE, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_write_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& header) {
    if constexpr (has_flag(DynamicMask, UnicastWriteDynamic::DstAddr)) {
        auto comps = get_noc_address_components(header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_write.noc_address = noc_addr;
    }
    if constexpr (has_flag(DynamicMask, UnicastWriteDynamic::PayloadSize)) {
        packet_header->payload_size_bytes = PACKET_SIZE;
    }
}

template <UnicastInlineWriteDynamic DynamicMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_inline_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& header) {
    if constexpr (has_flag(DynamicMask, UnicastInlineWriteDynamic::DstAddr)) {
        auto comps = get_noc_address_components(header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_inline_write.noc_address = noc_addr;
    }
    if constexpr (has_flag(DynamicMask, UnicastInlineWriteDynamic::Value)) {
        packet_header->command_fields.unicast_inline_write.value = header.value;
    }
}

template <UnicastAtomicIncDynamic DynamicMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_atomic_inc_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& header) {
    if constexpr (has_flag(DynamicMask, UnicastAtomicIncDynamic::DstAddr)) {
        auto comps = get_noc_address_components(header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_seminc.noc_address = noc_addr;
    }
    if constexpr (has_flag(DynamicMask, UnicastAtomicIncDynamic::Wrap)) {
        packet_header->command_fields.unicast_seminc.wrap = header.wrap;
    }
    if constexpr (has_flag(DynamicMask, UnicastAtomicIncDynamic::Val)) {
        packet_header->command_fields.unicast_seminc.val = header.val;
    }
    if constexpr (has_flag(DynamicMask, UnicastAtomicIncDynamic::Flush)) {
        packet_header->command_fields.unicast_seminc.flush = header.flush;
    }
}

// Scatter write populate (no mask for now; fully copies fields from header)
template <UnicastScatterWriteDynamic DynamicMask, uint32_t PACKET_SIZE, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_scatter_write_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& header) {
    if constexpr (has_flag(DynamicMask, UnicastScatterWriteDynamic::DstAddrs)) {
        for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS; i++) {
            auto comps = get_noc_address_components(header.noc_address[i]);
            auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
            packet_header->command_fields.unicast_scatter_write.noc_address[i] = noc_addr;
        }
    }
    if constexpr (has_flag(DynamicMask, UnicastScatterWriteDynamic::ChunkSizes)) {
        for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS - 1; i++) {
            packet_header->command_fields.unicast_scatter_write.chunk_size[i] = header.chunk_size[i];
        }
    }
    if constexpr (has_flag(DynamicMask, UnicastScatterWriteDynamic::PayloadSize)) {
        packet_header->payload_size_bytes = PACKET_SIZE;
    }
}

// Fused write+atomic_inc populate (no mask for now; fully copies fields from header)
template <UnicastFusedAtomicIncDynamic DynamicMask, uint32_t PACKET_SIZE, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_fused_atomic_inc_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& header) {
    if constexpr (has_flag(DynamicMask, UnicastFusedAtomicIncDynamic::WriteDstAddr)) {
        auto comps = get_noc_address_components(header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_seminc_fused.noc_address = noc_addr;
    }
    if constexpr (has_flag(DynamicMask, UnicastFusedAtomicIncDynamic::SemaphoreAddr)) {
        auto scomps = get_noc_address_components(header.semaphore_noc_address);
        auto snoc = safe_get_noc_addr(scomps.first.x, scomps.first.y, scomps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_seminc_fused.semaphore_noc_address = snoc;
    }
    if constexpr (has_flag(DynamicMask, UnicastFusedAtomicIncDynamic::Wrap)) {
        packet_header->command_fields.unicast_seminc_fused.wrap = header.wrap;
    }
    if constexpr (has_flag(DynamicMask, UnicastFusedAtomicIncDynamic::Val)) {
        packet_header->command_fields.unicast_seminc_fused.val = header.val;
    }
    if constexpr (has_flag(DynamicMask, UnicastFusedAtomicIncDynamic::Flush)) {
        packet_header->command_fields.unicast_seminc_fused.flush = header.flush;
    }
    if constexpr (has_flag(DynamicMask, UnicastFusedAtomicIncDynamic::PayloadSize)) {
        packet_header->payload_size_bytes = PACKET_SIZE;
    }
}

template <size_t num_send_dir>
FORCE_INLINE void open_connections(
    tt::tt_fabric::WorkerToFabricEdmSender (&client_interfaces)[num_send_dir], size_t& rt_arg_idx) {
    for (size_t i = 0; i < num_send_dir; i++) {
        client_interfaces[i] =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_arg_idx);
        client_interfaces[i].open();
    }
}

template <size_t num_send_dir>
FORCE_INLINE void close_connections(tt::tt_fabric::WorkerToFabricEdmSender (&client_interfaces)[num_send_dir]) {
    for (size_t i = 0; i < num_send_dir; i++) {
        client_interfaces[i].close();
    }
}

FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t num_hops) {
    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_unicast_write(noc_unicast_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_unicast_write(
            &client_interfaces[i], packet_header, src_addr, size, noc_unicast_command_header, num_hops[i]);
    });
}

// Templated with-state variant allowing compile-time control over dynamic fields
template <UnicastWriteDynamic DynamicMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    populate_unicast_write_fields<DynamicMask, PACKET_SIZE>(packet_header, noc_unicast_command_header);

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastWriteDynamic DynamicMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_unicast_write_with_state<DynamicMask, PACKET_SIZE>(
            &client_interfaces[i], packet_header, src_addr, noc_unicast_command_header);
    });
}

// Templated set-state variant to preconfigure static fields
template <UnicastWriteDynamic DynamicMask, uint32_t PACKET_SIZE, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    uint8_t route_id, uint8_t* num_hops, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
        populate_unicast_write_fields<DynamicMask, PACKET_SIZE>(packet_header, header);
    });
}

FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t num_hops) {
    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_unicast_atomic_inc(noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_unicast_atomic_inc(
            &client_interfaces[i], packet_header, noc_unicast_atomic_inc_command_header, num_hops[i]);
    });
}

// Templated with-state variant for Unicast Atomic Inc
template <UnicastAtomicIncDynamic DynamicMask>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    populate_unicast_atomic_inc_fields<DynamicMask>(packet_header, noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastAtomicIncDynamic DynamicMask>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_unicast_atomic_inc_with_state<DynamicMask>(
            &client_interfaces[i], packet_header, noc_unicast_atomic_inc_command_header);
    });
}

// Templated set-state variant for Unicast Atomic Inc
template <UnicastAtomicIncDynamic DynamicMask = UnicastAtomicIncDynamic::None, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    uint8_t route_id, uint8_t* num_hops, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
        packet_header->payload_size_bytes = 0;
        populate_unicast_atomic_inc_fields<DynamicMask>(packet_header, header);
    });
}

FORCE_INLINE void fabric_unicast_noc_scatter_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t num_hops) {
    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_unicast_scatter_write(noc_unicast_scatter_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_scatter_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_scatter_write(
            &client_interfaces[i], packet_header, src_addr, size, noc_unicast_scatter_command_header, num_hops[i]);
    });
}

template <UnicastScatterWriteDynamic DynamicMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    populate_unicast_scatter_write_fields<DynamicMask, PACKET_SIZE>(packet_header, noc_unicast_scatter_command_header);

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastScatterWriteDynamic DynamicMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_scatter_write_with_state<DynamicMask, PACKET_SIZE>(
            &client_interfaces[i], packet_header, src_addr, noc_unicast_scatter_command_header);
    });
}

template <UnicastScatterWriteDynamic DynamicMask, uint32_t PACKET_SIZE, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    uint8_t route_id, uint8_t* num_hops, CommandHeaderT header = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
        if constexpr (!has_flag(DynamicMask, UnicastScatterWriteDynamic::PayloadSize)) {
            packet_header->payload_size_bytes = PACKET_SIZE;
        }
        populate_unicast_scatter_write_fields<DynamicMask, PACKET_SIZE>(packet_header, header);
    });
}

FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t num_hops) {
    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_unicast_inline_write(noc_unicast_inline_write_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_unicast_inline_write(
            &client_interfaces[i], packet_header, noc_unicast_inline_write_command_header, num_hops[i]);
    });
}

// Templated with-state variant for Unicast Inline Write
template <UnicastInlineWriteDynamic DynamicMask>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    populate_unicast_inline_fields<DynamicMask>(packet_header, noc_unicast_inline_write_command_header);

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastInlineWriteDynamic DynamicMask>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_unicast_inline_write_with_state<DynamicMask>(
            &client_interfaces[i], packet_header, noc_unicast_inline_write_command_header);
    });
}

template <UnicastInlineWriteDynamic DynamicMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    uint8_t route_id, uint8_t* num_hops, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_INLINE_WRITE;
        packet_header->payload_size_bytes = 0;
        populate_unicast_inline_fields<DynamicMask>(packet_header, header);
    });
}

FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t num_hops) {
    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_fused_unicast_write_atomic_inc(noc_fused_unicast_atomic_inc_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_fused_unicast_with_atomic_inc(
            &client_interfaces[i],
            packet_header,
            src_addr,
            size,
            noc_fused_unicast_atomic_inc_command_header,
            num_hops[i]);
    });
}

template <UnicastFusedAtomicIncDynamic DynamicMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    populate_unicast_fused_atomic_inc_fields<DynamicMask, PACKET_SIZE>(
        packet_header, noc_fused_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastFusedAtomicIncDynamic DynamicMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<DynamicMask, PACKET_SIZE>(
            &client_interfaces[i], packet_header, src_addr, noc_fused_unicast_atomic_inc_command_header);
    });
}

template <UnicastFusedAtomicIncDynamic DynamicMask, uint32_t PACKET_SIZE, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    uint8_t route_id, uint8_t* num_hops, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
        if constexpr (!has_flag(DynamicMask, UnicastFusedAtomicIncDynamic::PayloadSize)) {
            packet_header->payload_size_bytes = PACKET_SIZE;
        }
        populate_unicast_fused_atomic_inc_fields<DynamicMask, PACKET_SIZE>(packet_header, header);
    });
}

FORCE_INLINE void fabric_multicast_noc_unicast_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t start_distance,
    uint8_t range) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->to_noc_unicast_write(noc_unicast_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_unicast_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t* start_distance,
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_unicast_write(
            &client_interfaces[i],
            packet_header,
            src_addr,
            size,
            noc_unicast_command_header,
            start_distance[i],
            range[i]);
    });
}

// Multicast templated variants for Unicast Write
template <UnicastWriteDynamic DynamicMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t runtime_size_bytes,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    populate_unicast_write_fields<DynamicMask, PACKET_SIZE>(packet_header, noc_unicast_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastWriteDynamic DynamicMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t runtime_size_bytes,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_unicast_write_with_state<DynamicMask, PACKET_SIZE>(
            &client_interfaces[i], packet_header, src_addr, runtime_size_bytes, noc_unicast_command_header);
    });
}

template <UnicastWriteDynamic DynamicMask, uint32_t PACKET_SIZE, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
        populate_unicast_write_fields<DynamicMask, PACKET_SIZE>(packet_header, header);
    });
}

FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t start_distance,
    uint8_t range) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->to_noc_unicast_atomic_inc(noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t* start_distance,
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_unicast_atomic_inc(
            &client_interfaces[i], packet_header, noc_unicast_atomic_inc_command_header, start_distance[i], range[i]);
    });
}

template <UnicastAtomicIncDynamic DynamicMask = UnicastAtomicIncDynamic::None, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range, CommandHeaderT header = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
        packet_header->payload_size_bytes = 0;
        populate_unicast_atomic_inc_fields<DynamicMask>(packet_header, header);
    });
}

FORCE_INLINE void fabric_multicast_noc_scatter_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t start_distance,
    uint8_t range) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->to_noc_unicast_scatter_write(noc_unicast_scatter_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_scatter_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t* start_distance,
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_scatter_write(
            &client_interfaces[i],
            packet_header,
            src_addr,
            size,
            noc_unicast_scatter_command_header,
            start_distance[i],
            range[i]);
    });
}

template <UnicastScatterWriteDynamic DynamicMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    populate_unicast_scatter_write_fields<DynamicMask, PACKET_SIZE>(packet_header, noc_unicast_scatter_command_header);

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastScatterWriteDynamic DynamicMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_scatter_write_with_state<DynamicMask, PACKET_SIZE>(
            &client_interfaces[i], packet_header, src_addr, noc_unicast_scatter_command_header);
    });
}

template <UnicastScatterWriteDynamic DynamicMask, uint32_t PACKET_SIZE, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range, CommandHeaderT header = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
        if constexpr (!has_flag(DynamicMask, UnicastScatterWriteDynamic::PayloadSize)) {
            packet_header->payload_size_bytes = PACKET_SIZE;
        }
        populate_unicast_scatter_write_fields<DynamicMask, PACKET_SIZE>(packet_header, header);
    });
}

FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t start_distance,
    uint8_t range) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->to_noc_unicast_inline_write(noc_unicast_inline_write_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t* start_distance,
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_unicast_inline_write(
            &client_interfaces[i], packet_header, noc_unicast_inline_write_command_header, start_distance[i], range[i]);
    });
}

// Multicast templated variants for Unicast Inline Write
template <UnicastInlineWriteDynamic DynamicMask>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    populate_unicast_inline_fields<DynamicMask>(packet_header, noc_unicast_inline_write_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastInlineWriteDynamic DynamicMask>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_unicast_inline_write_with_state<DynamicMask>(
            &client_interfaces[i], packet_header, noc_unicast_inline_write_command_header);
    });
}

template <UnicastInlineWriteDynamic DynamicMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_INLINE_WRITE;
        packet_header->payload_size_bytes = 0;
        populate_unicast_inline_fields<DynamicMask>(packet_header, header);
    });
}

FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t start_distance,
    uint8_t range) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->to_noc_fused_unicast_write_atomic_inc(noc_fused_unicast_atomic_inc_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t* start_distance,
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_fused_unicast_with_atomic_inc(
            &client_interfaces[i],
            packet_header,
            src_addr,
            size,
            noc_fused_unicast_atomic_inc_command_header,
            start_distance[i],
            range[i]);
    });
}

template <UnicastFusedAtomicIncDynamic DynamicMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    populate_unicast_fused_atomic_inc_fields<DynamicMask, PACKET_SIZE>(
        packet_header, noc_fused_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastFusedAtomicIncDynamic DynamicMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<DynamicMask, PACKET_SIZE>(
            &client_interfaces[i], packet_header, src_addr, noc_fused_unicast_atomic_inc_command_header);
    });
}

template <UnicastFusedAtomicIncDynamic DynamicMask, uint32_t PACKET_SIZE, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
        if constexpr (!has_flag(DynamicMask, UnicastFusedAtomicIncDynamic::PayloadSize)) {
            packet_header->payload_size_bytes = PACKET_SIZE;
        }
        populate_unicast_fused_atomic_inc_fields<DynamicMask, PACKET_SIZE>(packet_header, header);
    });
}

}  // namespace tt::tt_fabric::linear::experimental
