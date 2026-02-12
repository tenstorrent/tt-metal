// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef API_TYPE_Linear
#include "tt_metal/fabric/hw/inc/linear/api.h"
using namespace tt::tt_fabric::linear::experimental;
#elif defined(API_TYPE_Mesh)
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;
#else
#error "API_TYPE_Linear or API_TYPE_Mesh must be defined"
#endif
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "test_linear_common.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tests/tt_metal/tt_fabric/common/test_host_kernel_common.hpp"
using tt::tt_fabric::fabric_router_tests::FabricPacketType;
using tt::tt_fabric::fabric_router_tests::NocPacketType;

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);
constexpr uint32_t notification_mailbox_address = get_compile_time_arg_val(2);
uint32_t target_address = get_compile_time_arg_val(3);
constexpr NocPacketType noc_packet_type = static_cast<NocPacketType>(get_compile_time_arg_val(4));
static_assert(
    noc_packet_type < static_cast<uint32_t>(NocPacketType::NOC_PACKET_TYPE_COUNT),
    "Compile-time arg 4 (noc_packet_type) must be 0 (NOC_UNICAST_WRITE), 1 (NOC_UNICAST_INLINE_WRITE), 2 "
    "(NOC_UNICAST_ATOMIC_INC), 3 (NOC_FUSED_UNICAST_ATOMIC_INC), 4 (NOC_UNICAST_SCATTER_WRITE), 5 "
    "(NOC_MULTICAST_WRITE), 6 (NOC_MULTICAST_ATOMIC_INC), 7 (NOC_UNICAST_READ), 8 "
    "(NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC)");
constexpr uint32_t num_send_dir = get_compile_time_arg_val(5);
constexpr bool with_state = get_compile_time_arg_val(6) == 1;
constexpr uint32_t raw_fabric_packet_type = get_compile_time_arg_val(7);
static_assert(
    raw_fabric_packet_type < static_cast<uint32_t>(FabricPacketType::FABRIC_PACKET_TYPE_COUNT),
    "Compile-time arg 7 (fabric_packet_type) must be 0 (CHIP_UNICAST), 1 (CHIP_MULTICAST), or 2 "
    "(CHIP_SPARSE_MULTICAST)");
constexpr FabricPacketType fabric_packet_type = static_cast<FabricPacketType>(raw_fabric_packet_type);

void kernel_main() {
    size_t rt_arg_idx = 0;
    uint32_t source_l1_buffer_address = get_arg_val<uint32_t>(rt_arg_idx++);
    uint16_t packet_payload_size_bytes = static_cast<uint16_t>(get_arg_val<uint32_t>(rt_arg_idx++));
    uint32_t num_packets = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t noc_x_start = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t noc_y_start = get_arg_val<uint32_t>(rt_arg_idx++);
    auto hop_info = get_hop_info_from_args<fabric_packet_type, num_send_dir>(rt_arg_idx);

#ifdef API_TYPE_Mesh
    // Build MeshMcastRange array from hop_info once
    MeshMcastRange ranges[num_send_dir];
    if constexpr (fabric_packet_type == FabricPacketType::CHIP_MULTICAST) {
        for (uint32_t i = 0; i < num_send_dir; i++) {
            ranges[i].e = hop_info.mcast.e[i];
            ranges[i].w = hop_info.mcast.w[i];
            ranges[i].n = hop_info.mcast.n[i];
            ranges[i].s = hop_info.mcast.s[i];
        }
    }
#endif

    auto route_id = PacketHeaderPool::allocate_header_n(num_send_dir);
    tt::tt_fabric::RoutingPlaneConnectionManager connections;
    open_connections(connections, num_send_dir, rt_arg_idx);

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();

    if constexpr (with_state) {
        set_state<num_send_dir, fabric_packet_type, noc_packet_type>(
            connections, route_id, hop_info, static_cast<uint16_t>(packet_payload_size_bytes));
    }

    for (uint32_t i = 0; i < num_packets; i++) {
#ifdef API_TYPE_Linear
        static_assert(
            fabric_packet_type != FabricPacketType::CHIP_SPARSE_MULTICAST,
            "Sparse multicast has not been tested yet with atomic inc");
        if constexpr (fabric_packet_type == FabricPacketType::CHIP_MULTICAST) {
            switch (noc_packet_type) {
                case NocPacketType::NOC_UNICAST_ATOMIC_INC: {
                    if constexpr (with_state) {
                        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                            connections,
                            route_id,
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address), 0, false));
                    } else {
                        fabric_multicast_noc_unicast_atomic_inc(
                            connections,
                            route_id,
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address), 1, true},
                            hop_info.mcast.start_distance,
                            hop_info.mcast.range);
                    }
                } break;
                case NocPacketType::NOC_FUSED_UNICAST_ATOMIC_INC: {
                    if constexpr (with_state) {
                        fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<
                            UnicastFusedAtomicIncUpdateMask::WriteDstAddr |
                            UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader(
                                get_noc_addr(noc_x_start, noc_y_start, target_address),
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                                0,
                                false));
                    } else {
                        fabric_multicast_noc_fused_unicast_with_atomic_inc(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address),
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                                1,
                                true},
                            hop_info.mcast.start_distance,
                            hop_info.mcast.range);
                    }
                } break;
                case NocPacketType::NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC: {
                    if constexpr (with_state) {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader scatter_hdr(
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                            {first_chunk_size},
                            0,
                            false);
                        fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state<
                            UnicastFusedScatterWriteAtomicIncUpdateMask::WriteDstAddrs |
                            UnicastFusedScatterWriteAtomicIncUpdateMask::SemaphoreDstAddr>(
                            connections, route_id, source_l1_buffer_address, scatter_hdr);
                    } else {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader scatter_hdr(
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                            {first_chunk_size},
                            1,
                            true);
                        fabric_multicast_noc_fused_scatter_write_atomic_inc(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            scatter_hdr,
                            hop_info.mcast.start_distance,
                            hop_info.mcast.range);
                    }
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
        } else {
            switch (noc_packet_type) {
                case NocPacketType::NOC_UNICAST_ATOMIC_INC: {
                    if constexpr (with_state) {
                        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                            connections,
                            route_id,
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address), 0, false));
                    } else {
                        fabric_unicast_noc_unicast_atomic_inc(
                            connections,
                            route_id,
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address), 1, true},
                            hop_info.ucast.num_hops);
                    }
                } break;
                case NocPacketType::NOC_FUSED_UNICAST_ATOMIC_INC: {
                    if constexpr (with_state) {
                        fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
                            UnicastFusedAtomicIncUpdateMask::WriteDstAddr |
                            UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader(
                                get_noc_addr(noc_x_start, noc_y_start, target_address),
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                                0,
                                false));
                    } else {
                        fabric_unicast_noc_fused_unicast_with_atomic_inc(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address),
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                                1,
                                true},
                            hop_info.ucast.num_hops);
                    }
                } break;
                case NocPacketType::NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC: {
                    if constexpr (with_state) {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader scatter_hdr(
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                            {first_chunk_size},
                            0,
                            false);
                        fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state<
                            UnicastFusedScatterWriteAtomicIncUpdateMask::WriteDstAddrs |
                            UnicastFusedScatterWriteAtomicIncUpdateMask::SemaphoreDstAddr>(
                            connections, route_id, source_l1_buffer_address, scatter_hdr);
                    } else {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader scatter_hdr(
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                            {first_chunk_size},
                            1,
                            true);
                        fabric_unicast_noc_fused_scatter_write_atomic_inc(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            scatter_hdr,
                            hop_info.ucast.num_hops);
                    }
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
        }
#elif defined(API_TYPE_Mesh)
        static_assert(
            fabric_packet_type != FabricPacketType::CHIP_SPARSE_MULTICAST,
            "Sparse multicast has not been tested yet with mesh topology");
        if constexpr (fabric_packet_type == FabricPacketType::CHIP_MULTICAST) {
            switch (noc_packet_type) {
                case NocPacketType::NOC_UNICAST_ATOMIC_INC: {
                    if constexpr (with_state) {
                        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                            connections,
                            route_id,
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address), 0, false));
                    } else {
                        fabric_multicast_noc_unicast_atomic_inc(
                            connections,
                            route_id,
                            ranges,
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address), 1, true});
                    }
                } break;
                case NocPacketType::NOC_FUSED_UNICAST_ATOMIC_INC: {
                    if constexpr (with_state) {
                        fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<
                            UnicastFusedAtomicIncUpdateMask::WriteDstAddr |
                            UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader(
                                get_noc_addr(noc_x_start, noc_y_start, target_address),
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                                0,
                                false));
                    } else {
                        fabric_multicast_noc_fused_unicast_with_atomic_inc(
                            connections,
                            route_id,
                            ranges,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address),
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                                1,
                                true});
                    }
                } break;
                case NocPacketType::NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC: {
                    if constexpr (with_state) {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader scatter_hdr(
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                            {first_chunk_size},
                            0,
                            false);
                        fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state<
                            UnicastFusedScatterWriteAtomicIncUpdateMask::WriteDstAddrs |
                            UnicastFusedScatterWriteAtomicIncUpdateMask::SemaphoreDstAddr>(
                            connections, route_id, source_l1_buffer_address, scatter_hdr);
                    } else {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader scatter_hdr(
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                            {first_chunk_size},
                            1,
                            true);
                        fabric_multicast_noc_fused_scatter_write_atomic_inc(
                            connections,
                            route_id,
                            ranges,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            scatter_hdr);
                    }
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
        } else {
            switch (noc_packet_type) {
                case NocPacketType::NOC_UNICAST_ATOMIC_INC: {
                    if constexpr (with_state) {
                        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                            connections,
                            route_id,
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address), 0, false));
                    } else {
                        fabric_unicast_noc_unicast_atomic_inc(
                            connections,
                            route_id,
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address), 1, true});
                    }
                } break;
                case NocPacketType::NOC_FUSED_UNICAST_ATOMIC_INC: {
                    if constexpr (with_state) {
                        fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
                            UnicastFusedAtomicIncUpdateMask::WriteDstAddr |
                            UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader(
                                get_noc_addr(noc_x_start, noc_y_start, target_address),
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                                0,
                                false));
                    } else {
                        fabric_unicast_noc_fused_unicast_with_atomic_inc(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address),
                                get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                                1,
                                true});
                    }
                } break;
                case NocPacketType::NOC_FUSED_UNICAST_SCATTER_WRITE_ATOMIC_INC: {
                    if constexpr (with_state) {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader scatter_hdr(
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                            {first_chunk_size},
                            0,
                            false);
                        fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state<
                            UnicastFusedScatterWriteAtomicIncUpdateMask::WriteDstAddrs |
                            UnicastFusedScatterWriteAtomicIncUpdateMask::SemaphoreDstAddr>(
                            connections, route_id, source_l1_buffer_address, scatter_hdr);
                    } else {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader scatter_hdr(
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                            {first_chunk_size},
                            1,
                            true);
                        fabric_unicast_noc_fused_scatter_write_atomic_inc(
                            connections, route_id, source_l1_buffer_address, packet_payload_size_bytes, scatter_hdr);
                    }
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
        }
#else
#error "API_TYPE_Linear or API_TYPE_Mesh must be defined"
#endif
        noc_async_writes_flushed();
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    close_connections(connections);

    noc_async_write_barrier();

    uint64_t total_operations = num_packets;

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)total_operations;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = total_operations >> 32;
}
