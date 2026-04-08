// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef API_TYPE_Linear
#include "tt_metal/fabric/hw/inc/linear/api.h"
using namespace tt::tt_fabric::linear::experimental;
#elif defined(API_TYPE_Mesh)
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;
#else
#error "API_TYPE_Linear or API_TYPE_Mesh must be defined"
#endif
#include "fabric/fabric_edm_packet_header.hpp"
#include "tests/tt_metal/tt_fabric/common/test_host_kernel_common.hpp"
using tt::tt_fabric::fabric_router_tests::FabricPacketType;
using tt::tt_fabric::fabric_router_tests::NocPacketType;

template <uint8_t num_send_dir>
union HopInfo {
    struct {
        uint8_t num_hops[num_send_dir];
    } ucast;
#ifdef API_TYPE_Linear
    struct {
        uint8_t start_distance[num_send_dir];
        uint8_t range[num_send_dir];
    } mcast;
    struct {
        uint16_t hop_mask[num_send_dir];  // Sparse multicast hop bitmask
    } sparse_mcast;
#elif defined(API_TYPE_Mesh)
    struct {
        // For Mesh API: each connection has 4 directional ranges (e/w/n/s)
        uint8_t e[num_send_dir];
        uint8_t w[num_send_dir];
        uint8_t n[num_send_dir];
        uint8_t s[num_send_dir];
    } mcast;
#endif
};

template <FabricPacketType fabric_packet_type, uint8_t num_send_dir>
HopInfo<num_send_dir> get_hop_info_from_args(size_t& rt_arg_idx) {
    HopInfo<num_send_dir> hop_info;
    if constexpr (fabric_packet_type == FabricPacketType::CHIP_SPARSE_MULTICAST) {
#ifdef API_TYPE_Linear
        for (uint32_t i = 0; i < num_send_dir; i++) {
            hop_info.sparse_mcast.hop_mask[i] = static_cast<uint16_t>(get_arg_val<uint32_t>(rt_arg_idx++));
        }
#elif defined(API_TYPE_Mesh)
        ASSERT(false);  // Sparse multicast is not currently supported for Mesh
#endif
    } else if constexpr (fabric_packet_type == FabricPacketType::CHIP_MULTICAST) {
#ifdef API_TYPE_Linear
        for (uint32_t i = 0; i < num_send_dir; i++) {
            hop_info.mcast.start_distance[i] = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
            hop_info.mcast.range[i] = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
        }
#elif defined(API_TYPE_Mesh)
        // For Mesh API: read 4 directional ranges for each connection
        for (uint32_t i = 0; i < num_send_dir; i++) {
            hop_info.mcast.e[i] = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
            hop_info.mcast.w[i] = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
            hop_info.mcast.n[i] = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
            hop_info.mcast.s[i] = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
        }
#endif
    } else {
        for (uint32_t i = 0; i < num_send_dir; i++) {
            hop_info.ucast.num_hops[i] = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
        }
    }
    return hop_info;
}

// Set-state helper: field selection via template; packet size is runtime argument when applicable
template <uint8_t num_send_dir, FabricPacketType fabric_packet_type, NocPacketType noc_packet_type>
void set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    HopInfo<num_send_dir>& hop_info,
    uint16_t packet_size) {
#ifdef API_TYPE_Linear
    static_assert(
        fabric_packet_type != FabricPacketType::CHIP_SPARSE_MULTICAST,
        "Stateful sparse multicast has not been tested yet");
    if constexpr (fabric_packet_type == FabricPacketType::CHIP_MULTICAST) {
        switch (noc_packet_type) {
            case NocPacketType::NOC_UNICAST_WRITE: {
                fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                    connection_manager,
                    route_id,
                    hop_info.mcast.start_distance,
                    hop_info.mcast.range,
                    nullptr,
                    packet_size);
            } break;
            case NocPacketType::NOC_UNICAST_INLINE_WRITE: {
                tt::tt_fabric::NocUnicastInlineWriteCommandHeader hdr;
                hdr.value = 0xDEADBEEF;
                fabric_multicast_noc_unicast_inline_write_set_state<UnicastInlineWriteUpdateMask::Value>(
                    connection_manager, route_id, hop_info.mcast.start_distance, hop_info.mcast.range, hdr);
            } break;
            case NocPacketType::NOC_UNICAST_SCATTER_WRITE: {
                const auto shdr =
                    tt::tt_fabric::NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(packet_size / 2)});
                fabric_multicast_noc_scatter_write_set_state<
                    UnicastScatterWriteUpdateMask::PayloadSize | UnicastScatterWriteUpdateMask::ChunkSizes>(
                    connection_manager,
                    route_id,
                    hop_info.mcast.start_distance,
                    hop_info.mcast.range,
                    shdr,
                    packet_size);
            } break;
            case NocPacketType::NOC_UNICAST_ATOMIC_INC: {
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader ah(
                    0,    // dummy noc_address
                    1,    // val
                    true  // flush
                );
                fabric_multicast_noc_unicast_atomic_inc_set_state<
                    UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                    connection_manager, route_id, hop_info.mcast.start_distance, hop_info.mcast.range, ah);
            } break;
            case NocPacketType::NOC_FUSED_UNICAST_ATOMIC_INC: {
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader fh(
                    0,    // dummy noc_address
                    0,    // dummy semaphore_noc_address
                    1,    // val
                    true  // flush
                );
                fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<
                    UnicastFusedAtomicIncUpdateMask::PayloadSize | UnicastFusedAtomicIncUpdateMask::Val |
                    UnicastFusedAtomicIncUpdateMask::Flush>(
                    connection_manager, route_id, hop_info.mcast.start_distance, hop_info.mcast.range, fh, packet_size);
            } break;
            case NocPacketType::NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC: {
                tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader fh(
                    {0, 0},                                    // dummy noc_addresses
                    0,                                         // dummy semaphore_noc_address
                    {static_cast<uint16_t>(packet_size / 2)},  // dummy chunk_size
                    1,                                         // val
                    true                                       // flush
                );
                fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state<
                    UnicastFusedScatterWriteAtomicIncUpdateMask::PayloadSize |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::WriteChunkSizes |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::Val |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::Flush>(
                    connection_manager, route_id, hop_info.mcast.start_distance, hop_info.mcast.range, fh, packet_size);
            } break;
            default: {
                ASSERT(false);
            } break;
        }
    } else {
        switch (noc_packet_type) {
            case NocPacketType::NOC_UNICAST_WRITE: {
                fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                    connection_manager, route_id, hop_info.ucast.num_hops, nullptr, packet_size);
            } break;
            case NocPacketType::NOC_UNICAST_INLINE_WRITE: {
                tt::tt_fabric::NocUnicastInlineWriteCommandHeader hdr;
                hdr.value = 0xDEADBEEF;
                fabric_unicast_noc_unicast_inline_write_set_state<UnicastInlineWriteUpdateMask::Value>(
                    connection_manager, route_id, hop_info.ucast.num_hops, hdr);
            } break;
            case NocPacketType::NOC_UNICAST_SCATTER_WRITE: {
                const auto shdr =
                    tt::tt_fabric::NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(packet_size / 2)});
                fabric_unicast_noc_scatter_write_set_state<
                    UnicastScatterWriteUpdateMask::PayloadSize | UnicastScatterWriteUpdateMask::ChunkSizes>(
                    connection_manager, route_id, hop_info.ucast.num_hops, shdr, packet_size);
            } break;
            case NocPacketType::NOC_UNICAST_ATOMIC_INC: {
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader ah(
                    0,    // dummy noc_address
                    1,    // val
                    true  // flush
                );
                fabric_unicast_noc_unicast_atomic_inc_set_state<
                    UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                    connection_manager, route_id, hop_info.ucast.num_hops, ah);
            } break;
            case NocPacketType::NOC_FUSED_UNICAST_ATOMIC_INC: {
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader fh(
                    0,    // dummy noc_address
                    0,    // dummy semaphore_noc_address
                    1,    // val
                    true  // flush
                );
                fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
                    UnicastFusedAtomicIncUpdateMask::PayloadSize | UnicastFusedAtomicIncUpdateMask::Val |
                    UnicastFusedAtomicIncUpdateMask::Flush>(
                    connection_manager, route_id, hop_info.ucast.num_hops, fh, packet_size);
            } break;
            case NocPacketType::NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC: {
                tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader fh(
                    {0, 0},                                    // dummy noc_addresses
                    0,                                         // dummy semaphore_noc_address
                    {static_cast<uint16_t>(packet_size / 2)},  // dummy chunk_size
                    1,                                         // val
                    true                                       // flush
                );
                fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state<
                    UnicastFusedScatterWriteAtomicIncUpdateMask::PayloadSize |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::WriteChunkSizes |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::Val |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::Flush>(
                    connection_manager, route_id, hop_info.ucast.num_hops, fh, packet_size);
            } break;
            default: {
                ASSERT(false);
            } break;
        }
    }
#elif defined(API_TYPE_Mesh)
    static_assert(
        fabric_packet_type != FabricPacketType::CHIP_SPARSE_MULTICAST,
        "Sparse multicast is not currently supported for Mesh");
    if constexpr (fabric_packet_type == FabricPacketType::CHIP_MULTICAST) {
        // Build MeshMcastRange array from hop_info
        MeshMcastRange ranges[num_send_dir];
        for (uint32_t i = 0; i < num_send_dir; i++) {
            ranges[i].e = hop_info.mcast.e[i];
            ranges[i].w = hop_info.mcast.w[i];
            ranges[i].n = hop_info.mcast.n[i];
            ranges[i].s = hop_info.mcast.s[i];
        }

        switch (noc_packet_type) {
            case NocPacketType::NOC_UNICAST_WRITE: {
                fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                    connection_manager, route_id, ranges, nullptr, packet_size);
            } break;
            case NocPacketType::NOC_UNICAST_INLINE_WRITE: {
                tt::tt_fabric::NocUnicastInlineWriteCommandHeader hdr;
                hdr.value = 0xDEADBEEF;
                fabric_multicast_noc_unicast_inline_write_set_state<UnicastInlineWriteUpdateMask::Value>(
                    connection_manager, route_id, ranges, hdr);
            } break;
            case NocPacketType::NOC_UNICAST_SCATTER_WRITE: {
                const auto shdr =
                    tt::tt_fabric::NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(packet_size / 2)});
                fabric_multicast_noc_scatter_write_set_state<
                    UnicastScatterWriteUpdateMask::PayloadSize | UnicastScatterWriteUpdateMask::ChunkSizes>(
                    connection_manager, route_id, ranges, shdr, packet_size);
            } break;
            case NocPacketType::NOC_UNICAST_ATOMIC_INC: {
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader ah(
                    0,    // dummy noc_address
                    1,    // val
                    true  // flush
                );
                fabric_multicast_noc_unicast_atomic_inc_set_state<
                    UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                    connection_manager, route_id, ranges, ah);
            } break;
            case NocPacketType::NOC_FUSED_UNICAST_ATOMIC_INC: {
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader fh(
                    0,    // dummy noc_address
                    0,    // dummy semaphore_noc_address
                    1,    // val
                    true  // flush
                );
                fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<
                    UnicastFusedAtomicIncUpdateMask::PayloadSize | UnicastFusedAtomicIncUpdateMask::Val |
                    UnicastFusedAtomicIncUpdateMask::Flush>(connection_manager, route_id, ranges, fh, packet_size);
            } break;
            case NocPacketType::NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC: {
                tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader fh(
                    {0, 0},                                    // dummy noc_addresses
                    0,                                         // dummy semaphore_noc_address
                    {static_cast<uint16_t>(packet_size / 2)},  // dummy chunk_size
                    1,                                         // val
                    true                                       // flush
                );
                fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state<
                    UnicastFusedScatterWriteAtomicIncUpdateMask::PayloadSize |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::WriteChunkSizes |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::Val |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::Flush>(
                    connection_manager, route_id, ranges, fh, packet_size);
            } break;
            default: {
                ASSERT(false);
            } break;
        }
    } else {
        switch (noc_packet_type) {
            case NocPacketType::NOC_UNICAST_WRITE: {
                fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                    connection_manager, route_id, nullptr, packet_size);
            } break;
            case NocPacketType::NOC_UNICAST_INLINE_WRITE: {
                tt::tt_fabric::NocUnicastInlineWriteCommandHeader hdr;
                hdr.value = 0xDEADBEEF;
                fabric_unicast_noc_unicast_inline_write_set_state<UnicastInlineWriteUpdateMask::Value>(
                    connection_manager, route_id, hdr);
            } break;
            case NocPacketType::NOC_UNICAST_SCATTER_WRITE: {
                const auto shdr =
                    tt::tt_fabric::NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(packet_size / 2)});
                fabric_unicast_noc_scatter_write_set_state<
                    UnicastScatterWriteUpdateMask::PayloadSize | UnicastScatterWriteUpdateMask::ChunkSizes>(
                    connection_manager, route_id, shdr, packet_size);
            } break;
            case NocPacketType::NOC_UNICAST_ATOMIC_INC: {
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader ah(
                    0,    // dummy noc_address
                    1,    // val
                    true  // flush
                );
                fabric_unicast_noc_unicast_atomic_inc_set_state<
                    UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                    connection_manager, route_id, ah);
            } break;
            case NocPacketType::NOC_FUSED_UNICAST_ATOMIC_INC: {
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader fh(
                    0,    // dummy noc_address
                    0,    // dummy semaphore_noc_address
                    1,    // val
                    true  // flush
                );
                fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
                    UnicastFusedAtomicIncUpdateMask::PayloadSize | UnicastFusedAtomicIncUpdateMask::Val |
                    UnicastFusedAtomicIncUpdateMask::Flush>(connection_manager, route_id, fh, packet_size);
            } break;
            case NocPacketType::NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC: {
                tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader fh(
                    {0, 0},                                    // dummy noc_addresses
                    0,                                         // dummy semaphore_noc_address
                    {static_cast<uint16_t>(packet_size / 2)},  // dummy chunk_size
                    1,                                         // val
                    true                                       // flush
                );
                fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state<
                    UnicastFusedScatterWriteAtomicIncUpdateMask::PayloadSize |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::WriteChunkSizes |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::Val |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::Flush>(connection_manager, route_id, fh, packet_size);
            } break;
            default: {
                ASSERT(false);
            } break;
        }
    }
#else
#error "API_TYPE_Linear or API_TYPE_Mesh must be defined"
#endif
}
