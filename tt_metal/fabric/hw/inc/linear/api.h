// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

namespace tt::tt_fabric::linear::experimental {

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

FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    auto noc_address_components = get_noc_address_components(noc_unicast_command_header.noc_address);
    auto noc_addr = safe_get_noc_addr(
        noc_address_components.first.x,
        noc_address_components.first.y,
        noc_address_components.second,
        edm_to_local_chip_noc);
    packet_header->command_fields.unicast_write.noc_address = noc_addr;

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_unicast_write_with_state(
            &client_interfaces[i], packet_header, src_addr, noc_unicast_command_header);
    });
}

template <uint32_t PACKET_SIZE>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(uint8_t route_id, uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
        packet_header->payload_size_bytes = PACKET_SIZE;
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

FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    auto noc_address_components = get_noc_address_components(noc_unicast_atomic_inc_command_header.noc_address);
    auto noc_addr = safe_get_noc_addr(
        noc_address_components.first.x,
        noc_address_components.first.y,
        noc_address_components.second,
        edm_to_local_chip_noc);
    packet_header->command_fields.unicast_seminc.noc_address = noc_addr;
    packet_header->command_fields.unicast_seminc.wrap = noc_unicast_atomic_inc_command_header.wrap;
    packet_header->command_fields.unicast_seminc.val = noc_unicast_atomic_inc_command_header.val;
    packet_header->command_fields.unicast_seminc.flush = noc_unicast_atomic_inc_command_header.flush;

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_unicast_atomic_inc_with_state(
            &client_interfaces[i], packet_header, noc_unicast_atomic_inc_command_header);
    });
}

FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(uint8_t route_id, uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
        packet_header->payload_size_bytes = 0;  // No payload for atomic inc
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

FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS; i++) {
        auto noc_address_components = get_noc_address_components(noc_unicast_scatter_command_header.noc_address[i]);
        auto noc_addr = safe_get_noc_addr(
            noc_address_components.first.x,
            noc_address_components.first.y,
            noc_address_components.second,
            edm_to_local_chip_noc);
        packet_header->command_fields.unicast_scatter_write.noc_address[i] = noc_addr;
        if (i < NOC_SCATTER_WRITE_MAX_CHUNKS - 1) {
            packet_header->command_fields.unicast_scatter_write.chunk_size[i] =
                noc_unicast_scatter_command_header.chunk_size[i];
        }
    }

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_scatter_write_with_state(
            &client_interfaces[i], packet_header, src_addr, noc_unicast_scatter_command_header);
    });
}

template <uint32_t PACKET_SIZE>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(uint8_t route_id, uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
        packet_header->payload_size_bytes = PACKET_SIZE;
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

FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    auto noc_address_components = get_noc_address_components(noc_unicast_inline_write_command_header.noc_address);
    auto noc_addr = safe_get_noc_addr(
        noc_address_components.first.x,
        noc_address_components.first.y,
        noc_address_components.second,
        edm_to_local_chip_noc);
    packet_header->command_fields.unicast_inline_write.noc_address = noc_addr;
    packet_header->command_fields.unicast_inline_write.value = noc_unicast_inline_write_command_header.value;

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_unicast_inline_write_with_state(
            &client_interfaces[i], packet_header, noc_unicast_inline_write_command_header);
    });
}

FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(uint8_t route_id, uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_INLINE_WRITE;
        packet_header->payload_size_bytes = 0;
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

FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    auto noc_address_components = get_noc_address_components(noc_fused_unicast_atomic_inc_command_header.noc_address);
    auto noc_addr = safe_get_noc_addr(
        noc_address_components.first.x,
        noc_address_components.first.y,
        noc_address_components.second,
        edm_to_local_chip_noc);
    packet_header->command_fields.unicast_seminc_fused.noc_address = noc_addr;

    auto semaphore_noc_address_components =
        get_noc_address_components(noc_fused_unicast_atomic_inc_command_header.semaphore_noc_address);
    auto semaphore_noc_addr = safe_get_noc_addr(
        semaphore_noc_address_components.first.x,
        semaphore_noc_address_components.first.y,
        semaphore_noc_address_components.second,
        edm_to_local_chip_noc);
    packet_header->command_fields.unicast_seminc_fused.semaphore_noc_address = semaphore_noc_addr;
    packet_header->command_fields.unicast_seminc_fused.wrap = noc_fused_unicast_atomic_inc_command_header.wrap;
    packet_header->command_fields.unicast_seminc_fused.val = noc_fused_unicast_atomic_inc_command_header.val;
    packet_header->command_fields.unicast_seminc_fused.flush = noc_fused_unicast_atomic_inc_command_header.flush;

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
            &client_interfaces[i], packet_header, src_addr, noc_fused_unicast_atomic_inc_command_header);
    });
}

template <uint32_t PACKET_SIZE>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(uint8_t route_id, uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_unicast(num_hops[i]);
        packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
        packet_header->payload_size_bytes = PACKET_SIZE;
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

FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    auto noc_address_components = get_noc_address_components(noc_unicast_command_header.noc_address);
    auto noc_addr = safe_get_noc_addr(
        noc_address_components.first.x,
        noc_address_components.first.y,
        noc_address_components.second,
        edm_to_local_chip_noc);
    packet_header->command_fields.unicast_write.noc_address = noc_addr;

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_unicast_write_with_state(
            &client_interfaces[i], packet_header, src_addr, noc_unicast_command_header);
    });
}

template <uint32_t PACKET_SIZE>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
        packet_header->payload_size_bytes = PACKET_SIZE;
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

FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    auto noc_address_components = get_noc_address_components(noc_unicast_atomic_inc_command_header.noc_address);
    auto noc_addr = safe_get_noc_addr(
        noc_address_components.first.x,
        noc_address_components.first.y,
        noc_address_components.second,
        edm_to_local_chip_noc);
    packet_header->command_fields.unicast_seminc.noc_address = noc_addr;
    packet_header->command_fields.unicast_seminc.wrap = noc_unicast_atomic_inc_command_header.wrap;
    packet_header->command_fields.unicast_seminc.val = noc_unicast_atomic_inc_command_header.val;
    packet_header->command_fields.unicast_seminc.flush = noc_unicast_atomic_inc_command_header.flush;

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_unicast_atomic_inc_with_state(
            &client_interfaces[i], packet_header, noc_unicast_atomic_inc_command_header);
    });
}

FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
        packet_header->payload_size_bytes = 0;  // No payload for atomic inc
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

FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS; i++) {
        auto noc_address_components = get_noc_address_components(noc_unicast_scatter_command_header.noc_address[i]);
        auto noc_addr = safe_get_noc_addr(
            noc_address_components.first.x,
            noc_address_components.first.y,
            noc_address_components.second,
            edm_to_local_chip_noc);
        packet_header->command_fields.unicast_scatter_write.noc_address[i] = noc_addr;
        if (i < NOC_SCATTER_WRITE_MAX_CHUNKS - 1) {
            packet_header->command_fields.unicast_scatter_write.chunk_size[i] =
                noc_unicast_scatter_command_header.chunk_size[i];
        }
    }

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_scatter_write_with_state(
            &client_interfaces[i], packet_header, src_addr, noc_unicast_scatter_command_header);
    });
}

template <uint32_t PACKET_SIZE>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
        packet_header->payload_size_bytes = PACKET_SIZE;
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

FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    auto noc_address_components = get_noc_address_components(noc_unicast_inline_write_command_header.noc_address);
    auto noc_addr = safe_get_noc_addr(
        noc_address_components.first.x,
        noc_address_components.first.y,
        noc_address_components.second,
        edm_to_local_chip_noc);
    packet_header->command_fields.unicast_inline_write.noc_address = noc_addr;
    packet_header->command_fields.unicast_inline_write.value = noc_unicast_inline_write_command_header.value;

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_unicast_inline_write_with_state(
            &client_interfaces[i], packet_header, noc_unicast_inline_write_command_header);
    });
}

FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_INLINE_WRITE;
        packet_header->payload_size_bytes = 0;
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

FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    auto noc_address_components = get_noc_address_components(noc_fused_unicast_atomic_inc_command_header.noc_address);
    auto noc_addr = safe_get_noc_addr(
        noc_address_components.first.x,
        noc_address_components.first.y,
        noc_address_components.second,
        edm_to_local_chip_noc);
    packet_header->command_fields.unicast_seminc_fused.noc_address = noc_addr;

    auto semaphore_noc_address_components =
        get_noc_address_components(noc_fused_unicast_atomic_inc_command_header.semaphore_noc_address);
    auto semaphore_noc_addr = safe_get_noc_addr(
        semaphore_noc_address_components.first.x,
        semaphore_noc_address_components.first.y,
        semaphore_noc_address_components.second,
        edm_to_local_chip_noc);
    packet_header->command_fields.unicast_seminc_fused.semaphore_noc_address = semaphore_noc_addr;
    packet_header->command_fields.unicast_seminc_fused.wrap = noc_fused_unicast_atomic_inc_command_header.wrap;
    packet_header->command_fields.unicast_seminc_fused.val = noc_fused_unicast_atomic_inc_command_header.val;
    packet_header->command_fields.unicast_seminc_fused.flush = noc_fused_unicast_atomic_inc_command_header.flush;

    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interfaces,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
            &client_interfaces[i], packet_header, src_addr, noc_fused_unicast_atomic_inc_command_header);
    });
}

template <uint32_t PACKET_SIZE>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    uint8_t route_id, uint8_t* start_distance, uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance[i], range[i]});
        packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
        packet_header->payload_size_bytes = PACKET_SIZE;
    });
}

}  // namespace tt::tt_fabric::linear::experimental
