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
        time_seed = prng_next(time_seed);
        tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address);
        fill_packet_data(start_addr, packet_payload_size_bytes / 16, time_seed);

#ifdef API_TYPE_Linear
        if constexpr (fabric_packet_type == FabricPacketType::CHIP_SPARSE_MULTICAST) {
            // Currently sparse multicast has only been tested for NoC Unicast Writes
            switch (noc_packet_type) {
                case NOC_UNICAST_WRITE: {
                    fabric_sparse_multicast_noc_unicast_write(
                        connections,
                        route_id,
                        source_l1_buffer_address,
                        packet_payload_size_bytes,
                        tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(noc_x_start, noc_y_start, target_address)},
                        hop_info.sparse_mcast.hop_mask);
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
        } else if constexpr (fabric_packet_type == FabricPacketType::CHIP_MULTICAST) {
            switch (noc_packet_type) {
                case NOC_UNICAST_WRITE: {
                    if constexpr (with_state) {
                        fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            tt::tt_fabric::NocUnicastCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address)});
                    } else {
                        fabric_multicast_noc_unicast_write(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            tt::tt_fabric::NocUnicastCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address)},
                            hop_info.mcast.start_distance,
                            hop_info.mcast.range);
                    }
                } break;
                case NOC_UNICAST_INLINE_WRITE: {
                    if constexpr (with_state) {
                        tt::tt_fabric::NocUnicastInlineWriteCommandHeader ih{};
                        ih.noc_address = get_noc_addr(noc_x_start, noc_y_start, target_address);
                        fabric_multicast_noc_unicast_inline_write_with_state<UnicastInlineWriteUpdateMask::DstAddr>(
                            connections, route_id, ih);
                    } else {
                        fabric_multicast_noc_unicast_inline_write(
                            connections,
                            route_id,
                            tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address), 0xDEADBEEF},
                            hop_info.mcast.start_distance,
                            hop_info.mcast.range);
                    }
                } break;
                case NOC_UNICAST_SCATTER_WRITE: {
                    if constexpr (with_state) {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        const auto sh = tt::tt_fabric::NocUnicastScatterCommandHeader(
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            {first_chunk_size});
                        fabric_multicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                            connections, route_id, source_l1_buffer_address, sh);
                    } else {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        fabric_multicast_noc_scatter_write(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            tt::tt_fabric::NocUnicastScatterCommandHeader(
                                {get_noc_addr(noc_x_start, noc_y_start, target_address),
                                 get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                                {first_chunk_size}),
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
                case NOC_UNICAST_WRITE: {
                    if constexpr (with_state) {
                        fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            tt::tt_fabric::NocUnicastCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address)},
                            packet_payload_size_bytes);
                    } else {
                        fabric_unicast_noc_unicast_write(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            tt::tt_fabric::NocUnicastCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address)},
                            hop_info.ucast.num_hops);
                    }
                } break;
                case NOC_UNICAST_INLINE_WRITE: {
                    if constexpr (with_state) {
                        tt::tt_fabric::NocUnicastInlineWriteCommandHeader ih{};
                        ih.noc_address = get_noc_addr(noc_x_start, noc_y_start, target_address);
                        fabric_unicast_noc_unicast_inline_write_with_state<UnicastInlineWriteUpdateMask::DstAddr>(
                            connections, route_id, ih);
                    } else {
                        fabric_unicast_noc_unicast_inline_write(
                            connections,
                            route_id,
                            tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address), 0xDEADBEEF},
                            hop_info.ucast.num_hops);
                    }
                } break;
                case NOC_UNICAST_SCATTER_WRITE: {
                    if constexpr (with_state) {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        const auto sh = tt::tt_fabric::NocUnicastScatterCommandHeader(
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            {first_chunk_size});
                        fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                            connections, route_id, source_l1_buffer_address, sh);
                    } else {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        fabric_unicast_noc_scatter_write(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            tt::tt_fabric::NocUnicastScatterCommandHeader(
                                {get_noc_addr(noc_x_start, noc_y_start, target_address),
                                 get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                                {first_chunk_size}),
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
                case NOC_UNICAST_WRITE: {
                    if constexpr (with_state) {
                        fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            tt::tt_fabric::NocUnicastCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address)},
                            packet_payload_size_bytes);
                    } else {
                        fabric_multicast_noc_unicast_write(
                            connections,
                            route_id,
                            ranges,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            tt::tt_fabric::NocUnicastCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address)});
                    }
                } break;
                case NOC_UNICAST_INLINE_WRITE: {
                    if constexpr (with_state) {
                        fabric_multicast_noc_unicast_inline_write_with_state<UnicastInlineWriteUpdateMask::DstAddr>(
                            connections,
                            route_id,
                            tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address), 0xDEADBEEF});
                    } else {
                        fabric_multicast_noc_unicast_inline_write(
                            connections,
                            route_id,
                            ranges,
                            tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address), 0xDEADBEEF});
                    }
                } break;
                case NOC_UNICAST_SCATTER_WRITE: {
                    if constexpr (with_state) {
                        fabric_multicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            tt::tt_fabric::NocUnicastScatterCommandHeader(
                                {get_noc_addr(noc_x_start, noc_y_start, target_address),
                                 get_noc_addr(
                                     noc_x_start, noc_y_start, target_address + packet_payload_size_bytes / 2)}));
                    } else {
                        fabric_multicast_noc_scatter_write(
                            connections,
                            route_id,
                            ranges,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            tt::tt_fabric::NocUnicastScatterCommandHeader(
                                {get_noc_addr(noc_x_start, noc_y_start, target_address),
                                 get_noc_addr(
                                     noc_x_start, noc_y_start, target_address + packet_payload_size_bytes / 2)},
                                {static_cast<uint16_t>(packet_payload_size_bytes / 2)}));
                    }
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
        } else {
            switch (noc_packet_type) {
                case NOC_UNICAST_WRITE: {
                    if constexpr (with_state) {
                        fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            tt::tt_fabric::NocUnicastCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address)},
                            packet_payload_size_bytes);
                    } else {
                        fabric_unicast_noc_unicast_write(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            tt::tt_fabric::NocUnicastCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address)});
                    }
                } break;
                case NOC_UNICAST_INLINE_WRITE: {
                    if constexpr (with_state) {
                        tt::tt_fabric::NocUnicastInlineWriteCommandHeader ih{};
                        ih.noc_address = get_noc_addr(noc_x_start, noc_y_start, target_address);
                        fabric_unicast_noc_unicast_inline_write_with_state<UnicastInlineWriteUpdateMask::DstAddr>(
                            connections, route_id, ih);
                    } else {
                        fabric_unicast_noc_unicast_inline_write(
                            connections,
                            route_id,
                            tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                                get_noc_addr(noc_x_start, noc_y_start, target_address), 0xDEADBEEF});
                    }
                } break;
                case NOC_UNICAST_SCATTER_WRITE: {
                    if constexpr (with_state) {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        const auto sh = tt::tt_fabric::NocUnicastScatterCommandHeader(
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            {first_chunk_size});
                        fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                            connections, route_id, source_l1_buffer_address, sh);
                    } else {
                        uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                        fabric_unicast_noc_scatter_write(
                            connections,
                            route_id,
                            source_l1_buffer_address,
                            packet_payload_size_bytes,
                            tt::tt_fabric::NocUnicastScatterCommandHeader(
                                {get_noc_addr(noc_x_start, noc_y_start, target_address),
                                 get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                                {first_chunk_size}));
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
        target_address += packet_payload_size_bytes;
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    close_connections(connections);

    noc_async_write_barrier();

    uint64_t bytes_sent = packet_payload_size_bytes * num_packets;

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)bytes_sent;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = bytes_sent >> 32;
}
