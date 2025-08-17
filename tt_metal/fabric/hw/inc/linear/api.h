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
// Common populate helpers
// ========================

template <UnicastWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_write_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& header) {
    if constexpr (has_flag(UpdateMask, UnicastWriteUpdateMask::DstAddr)) {
        auto comps = get_noc_address_components(header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_write.noc_address = noc_addr;
    }
    if constexpr (has_flag(UpdateMask, UnicastWriteUpdateMask::PayloadSize)) {
        packet_header->payload_size_bytes = PACKET_SIZE;
    }
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_inline_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& header) {
    if constexpr (has_flag(UpdateMask, UnicastInlineWriteUpdateMask::DstAddr)) {
        auto comps = get_noc_address_components(header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_inline_write.noc_address = noc_addr;
    }
    if constexpr (has_flag(UpdateMask, UnicastInlineWriteUpdateMask::Value)) {
        packet_header->command_fields.unicast_inline_write.value = header.value;
    }
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_atomic_inc_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& header) {
    if constexpr (has_flag(UpdateMask, UnicastAtomicIncUpdateMask::DstAddr)) {
        auto comps = get_noc_address_components(header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_seminc.noc_address = noc_addr;
    }
    if constexpr (has_flag(UpdateMask, UnicastAtomicIncUpdateMask::Wrap)) {
        packet_header->command_fields.unicast_seminc.wrap = header.wrap;
    }
    if constexpr (has_flag(UpdateMask, UnicastAtomicIncUpdateMask::Val)) {
        packet_header->command_fields.unicast_seminc.val = header.val;
    }
    if constexpr (has_flag(UpdateMask, UnicastAtomicIncUpdateMask::Flush)) {
        packet_header->command_fields.unicast_seminc.flush = header.flush;
    }
}

// Scatter write populate (no mask for now; fully copies fields from header)
template <UnicastScatterWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_scatter_write_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& header) {
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::DstAddrs)) {
        for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS; i++) {
            auto comps = get_noc_address_components(header.noc_address[i]);
            auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
            packet_header->command_fields.unicast_scatter_write.noc_address[i] = noc_addr;
        }
    }
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::ChunkSizes)) {
        for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS - 1; i++) {
            packet_header->command_fields.unicast_scatter_write.chunk_size[i] = header.chunk_size[i];
        }
    }
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::PayloadSize)) {
        packet_header->payload_size_bytes = PACKET_SIZE;
    }
}

// Fused write+atomic_inc populate (no mask for now; fully copies fields from header)
template <UnicastFusedAtomicIncUpdateMask UpdateMask, uint32_t PACKET_SIZE, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_fused_atomic_inc_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& header) {
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::WriteDstAddr)) {
        auto comps = get_noc_address_components(header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_seminc_fused.noc_address = noc_addr;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::SemaphoreAddr)) {
        auto scomps = get_noc_address_components(header.semaphore_noc_address);
        auto snoc = safe_get_noc_addr(scomps.first.x, scomps.first.y, scomps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_seminc_fused.semaphore_noc_address = snoc;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::Wrap)) {
        packet_header->command_fields.unicast_seminc_fused.wrap = header.wrap;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::Val)) {
        packet_header->command_fields.unicast_seminc_fused.val = header.val;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::Flush)) {
        packet_header->command_fields.unicast_seminc_fused.flush = header.flush;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::PayloadSize)) {
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

// Templated with-state variant allowing compile-time control over which fields to update
template <UnicastWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    populate_unicast_write_fields<UpdateMask, PACKET_SIZE>(packet_header, noc_unicast_command_header);

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_unicast_write_with_state<UpdateMask, PACKET_SIZE>(
            &client_interfaces[i], packet_header, src_addr, noc_unicast_command_header);
    });
}

// Templated set-state variant to preconfigure static fields
template <UnicastWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    uint8_t route_id, uint8_t* num_hops, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
        populate_unicast_write_fields<UpdateMask, PACKET_SIZE>(packet_header, header);
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
template <UnicastAtomicIncUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastAtomicIncUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_unicast_atomic_inc_with_state<UpdateMask>(
            &client_interfaces[i], packet_header, noc_unicast_atomic_inc_command_header);
    });
}

// Templated set-state variant for Unicast Atomic Inc
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    uint8_t route_id, uint8_t* num_hops, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
        packet_header->payload_size_bytes = 0;
        populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, header);
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

template <UnicastScatterWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    populate_unicast_scatter_write_fields<UpdateMask, PACKET_SIZE>(packet_header, noc_unicast_scatter_command_header);

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastScatterWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_scatter_write_with_state<UpdateMask, PACKET_SIZE>(
            &client_interfaces[i], packet_header, src_addr, noc_unicast_scatter_command_header);
    });
}

template <UnicastScatterWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    uint8_t route_id, uint8_t* num_hops, CommandHeaderT header = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
        populate_unicast_scatter_write_fields<UpdateMask, PACKET_SIZE>(packet_header, header);
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
template <UnicastInlineWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    populate_unicast_inline_fields<UpdateMask>(packet_header, noc_unicast_inline_write_command_header);

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastInlineWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_unicast_inline_write_with_state<UpdateMask>(
            &client_interfaces[i], packet_header, noc_unicast_inline_write_command_header);
    });
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    uint8_t route_id, uint8_t* num_hops, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_INLINE_WRITE;
        packet_header->payload_size_bytes = 0;
        populate_unicast_inline_fields<UpdateMask>(packet_header, header);
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

template <UnicastFusedAtomicIncUpdateMask UpdateMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    populate_unicast_fused_atomic_inc_fields<UpdateMask, PACKET_SIZE>(
        packet_header, noc_fused_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<UpdateMask, PACKET_SIZE>(
            &client_interfaces[i], packet_header, src_addr, noc_fused_unicast_atomic_inc_command_header);
    });
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, uint32_t PACKET_SIZE, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    uint8_t route_id, uint8_t* num_hops, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
        populate_unicast_fused_atomic_inc_fields<UpdateMask, PACKET_SIZE>(packet_header, header);
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
template <UnicastWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t runtime_size_bytes,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    populate_unicast_write_fields<UpdateMask, PACKET_SIZE>(packet_header, noc_unicast_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t runtime_size_bytes,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_unicast_write_with_state<UpdateMask, PACKET_SIZE>(
            &client_interfaces[i], packet_header, src_addr, runtime_size_bytes, noc_unicast_command_header);
    });
}

template <UnicastWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
        populate_unicast_write_fields<UpdateMask, PACKET_SIZE>(packet_header, header);
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

// Multicast templated variants for Unicast Atomic Inc (with state)
template <UnicastAtomicIncUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastAtomicIncUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_unicast_atomic_inc_with_state<UpdateMask>(
            &client_interfaces[i], packet_header, noc_unicast_atomic_inc_command_header);
    });
}

template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range, CommandHeaderT header = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
        packet_header->payload_size_bytes = 0;
        populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, header);
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

template <UnicastScatterWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    populate_unicast_scatter_write_fields<UpdateMask, PACKET_SIZE>(packet_header, noc_unicast_scatter_command_header);

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastScatterWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_scatter_write_with_state<UpdateMask, PACKET_SIZE>(
            &client_interfaces[i], packet_header, src_addr, noc_unicast_scatter_command_header);
    });
}

template <UnicastScatterWriteUpdateMask UpdateMask, uint32_t PACKET_SIZE, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range, CommandHeaderT header = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
        populate_unicast_scatter_write_fields<UpdateMask, PACKET_SIZE>(packet_header, header);
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
template <UnicastInlineWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    populate_unicast_inline_fields<UpdateMask>(packet_header, noc_unicast_inline_write_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastInlineWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_unicast_inline_write_with_state<UpdateMask>(
            &client_interfaces[i], packet_header, noc_unicast_inline_write_command_header);
    });
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_INLINE_WRITE;
        packet_header->payload_size_bytes = 0;
        populate_unicast_inline_fields<UpdateMask>(packet_header, header);
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

template <UnicastFusedAtomicIncUpdateMask UpdateMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    populate_unicast_fused_atomic_inc_fields<UpdateMask, PACKET_SIZE>(
        packet_header, noc_fused_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, uint32_t PACKET_SIZE = 0>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<UpdateMask, PACKET_SIZE>(
            &client_interfaces[i], packet_header, src_addr, noc_fused_unicast_atomic_inc_command_header);
    });
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, uint32_t PACKET_SIZE, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range, CommandHeaderT header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
        populate_unicast_fused_atomic_inc_fields<UpdateMask, PACKET_SIZE>(packet_header, header);
    });
}

}  // namespace tt::tt_fabric::linear::experimental
