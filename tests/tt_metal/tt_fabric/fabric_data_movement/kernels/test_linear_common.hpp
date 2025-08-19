// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
using namespace tt::tt_fabric::linear::experimental;

template <uint8_t num_send_dir>
union HopInfo {
    struct {
        uint8_t num_hops[num_send_dir];
    } ucast;
    struct {
        uint8_t start_distance[num_send_dir];
        uint8_t range[num_send_dir];
    } mcast;
};

template <bool is_chip_multicast, uint8_t num_send_dir>
HopInfo<num_send_dir> get_hop_info_from_args(size_t& rt_arg_idx) {
    HopInfo<num_send_dir> hop_info;
    if constexpr (is_chip_multicast) {
        for (uint32_t i = 0; i < num_send_dir; i++) {
            hop_info.mcast.start_distance[i] = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
            hop_info.mcast.range[i] = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
        }
    } else {
        for (uint32_t i = 0; i < num_send_dir; i++) {
            hop_info.ucast.num_hops[i] = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
        }
    }
    return hop_info;
}

// Set-state helper: field selection via template; packet size is runtime argument when applicable
template <uint8_t num_send_dir, bool is_chip_multicast, tt::tt_fabric::NocSendType noc_send_type>
void set_state(uint8_t route_id, HopInfo<num_send_dir>& hop_info, uint16_t packet_size) {
    if constexpr (is_chip_multicast) {
        switch (noc_send_type) {
            case NOC_UNICAST_WRITE: {
                fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                    route_id, hop_info.mcast.start_distance, hop_info.mcast.range, packet_size, nullptr);
            } break;
            case NOC_UNICAST_INLINE_WRITE: {
                fabric_multicast_noc_unicast_inline_write_set_state<UnicastInlineWriteUpdateMask::Value>(
                    route_id,
                    hop_info.mcast.start_distance,
                    hop_info.mcast.range,
                    tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                        0,  // ignore
                        0xDEADBEEF});
            } break;
            case NOC_UNICAST_SCATTER_WRITE: {
                fabric_multicast_noc_scatter_write_set_state<
                    UnicastScatterWriteUpdateMask::PayloadSize | UnicastScatterWriteUpdateMask::ChunkSizes>(
                    route_id,
                    hop_info.mcast.start_distance,
                    hop_info.mcast.range,
                    packet_size,
                    tt::tt_fabric::NocUnicastScatterCommandHeader{
                        {0, 0},  // ignore
                        static_cast<uint16_t>(packet_size / 2)});
            } break;
            case NOC_UNICAST_ATOMIC_INC: {
                fabric_multicast_noc_unicast_atomic_inc_set_state<
                    UnicastAtomicIncUpdateMask::Wrap | UnicastAtomicIncUpdateMask::Val |
                    UnicastAtomicIncUpdateMask::Flush>(
                    route_id,
                    hop_info.mcast.start_distance,
                    hop_info.mcast.range,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                        0,  // ignore
                        1,
                        std::numeric_limits<uint16_t>::max(),
                        true});
            } break;
            case NOC_FUSED_UNICAST_ATOMIC_INC: {
                fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<
                    UnicastFusedAtomicIncUpdateMask::PayloadSize | UnicastFusedAtomicIncUpdateMask::Wrap |
                    UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush>(
                    route_id,
                    hop_info.mcast.start_distance,
                    hop_info.mcast.range,
                    packet_size,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        0,  // ignore
                        0,  // ignore
                        1,
                        std::numeric_limits<uint16_t>::max(),
                        true});
            } break;
            default: {
                ASSERT(false);
            } break;
        }
    } else {
        switch (noc_send_type) {
            case NOC_UNICAST_WRITE: {
                fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                    route_id, hop_info.ucast.num_hops, packet_size, nullptr);
            } break;
            case NOC_UNICAST_INLINE_WRITE: {
                fabric_unicast_noc_unicast_inline_write_set_state<UnicastInlineWriteUpdateMask::Value>(
                    route_id,
                    hop_info.ucast.num_hops,
                    tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                        0,  // ignore
                        0xDEADBEEF});
            } break;
            case NOC_UNICAST_SCATTER_WRITE: {
                fabric_unicast_noc_scatter_write_set_state<
                    UnicastScatterWriteUpdateMask::PayloadSize | UnicastScatterWriteUpdateMask::ChunkSizes>(
                    route_id,
                    hop_info.ucast.num_hops,
                    packet_size,
                    tt::tt_fabric::NocUnicastScatterCommandHeader{
                        {0, 0},  // ignore
                        static_cast<uint16_t>(packet_size / 2)});
            } break;
            case NOC_UNICAST_ATOMIC_INC: {
                fabric_unicast_noc_unicast_atomic_inc_set_state<
                    UnicastAtomicIncUpdateMask::Wrap | UnicastAtomicIncUpdateMask::Val |
                    UnicastAtomicIncUpdateMask::Flush>(
                    route_id,
                    hop_info.ucast.num_hops,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                        0,  // ignore
                        1,
                        std::numeric_limits<uint16_t>::max(),
                        true});
            } break;
            case NOC_FUSED_UNICAST_ATOMIC_INC: {
                fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
                    UnicastFusedAtomicIncUpdateMask::PayloadSize | UnicastFusedAtomicIncUpdateMask::Wrap |
                    UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush>(
                    route_id,
                    hop_info.ucast.num_hops,
                    packet_size,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        0,  // ignore
                        0,  // ignore
                        1,
                        std::numeric_limits<uint16_t>::max(),
                        true});
            } break;
            default: {
                ASSERT(false);
            } break;
        }
    }
}
