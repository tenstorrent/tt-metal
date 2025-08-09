// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"

using namespace tt::tt_fabric::linear::experimental;
constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);
constexpr uint32_t notification_mailbox_address = get_compile_time_arg_val(2);
uint32_t target_address = get_compile_time_arg_val(3);
constexpr NocSendType noc_send_type = static_cast<NocSendType>(get_compile_time_arg_val(4));
constexpr uint32_t num_send_dir = get_compile_time_arg_val(5);
constexpr bool is_chip_multicast = get_compile_time_arg_val(6) == 1;

void kernel_main() {
    size_t rt_arg_idx = 0;
    uint32_t source_l1_buffer_address = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t num_packets = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t noc_x_start = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t noc_y_start = get_arg_val<uint32_t>(rt_arg_idx++);
    union {
        struct {
            uint8_t num_hops[num_send_dir];
        } ucast;
        struct {
            uint8_t start_distance[num_send_dir];
            uint8_t range[num_send_dir];
        } mcast;
    } hop_info;
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

    auto route_id = PacketHeaderPool::allocate_header_n(num_send_dir);
    tt::tt_fabric::WorkerToFabricEdmSender connections[num_send_dir] = {};
    open_connections(connections, rt_arg_idx);

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();

    for (uint32_t i = 0; i < num_packets; i++) {
        time_seed = prng_next(time_seed);
        tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address);
        fill_packet_data(start_addr, packet_payload_size_bytes / 16, time_seed);

        if constexpr (is_chip_multicast) {
            switch (noc_send_type) {
                case NOC_UNICAST_WRITE: {
                    fabric_multicast_noc_unicast_write(
                        connections,
                        route_id,
                        source_l1_buffer_address,
                        packet_payload_size_bytes,
                        tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(noc_x_start, noc_y_start, target_address)},
                        hop_info.mcast.start_distance,
                        hop_info.mcast.range);
                } break;
                case NOC_UNICAST_INLINE_WRITE: {
                    fabric_multicast_noc_unicast_inline_write(
                        connections,
                        route_id,
                        tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                            get_noc_addr(noc_x_start, noc_y_start, target_address), 0xDEADBEEF},
                        hop_info.mcast.start_distance,
                        hop_info.mcast.range);
                } break;
                case NOC_UNICAST_SCATTER_WRITE: {
                    uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                    fabric_multicast_noc_scatter_write(
                        connections,
                        route_id,
                        source_l1_buffer_address,
                        packet_payload_size_bytes,
                        tt::tt_fabric::NocUnicastScatterCommandHeader{
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            first_chunk_size},
                        hop_info.mcast.start_distance,
                        hop_info.mcast.range);
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
        } else {
            switch (noc_send_type) {
                case NOC_UNICAST_WRITE: {
                    fabric_unicast_noc_unicast_write(
                        connections,
                        route_id,
                        source_l1_buffer_address,
                        packet_payload_size_bytes,
                        tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(noc_x_start, noc_y_start, target_address)},
                        hop_info.ucast.num_hops);
                } break;
                case NOC_UNICAST_INLINE_WRITE: {
                    fabric_unicast_noc_unicast_inline_write(
                        connections,
                        route_id,
                        tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                            get_noc_addr(noc_x_start, noc_y_start, target_address), 0xDEADBEEF},
                        hop_info.ucast.num_hops);
                } break;
                case NOC_UNICAST_SCATTER_WRITE: {
                    uint16_t first_chunk_size = packet_payload_size_bytes / 2;
                    fabric_unicast_noc_scatter_write(
                        connections,
                        route_id,
                        source_l1_buffer_address,
                        packet_payload_size_bytes,
                        tt::tt_fabric::NocUnicastScatterCommandHeader{
                            {get_noc_addr(noc_x_start, noc_y_start, target_address),
                             get_noc_addr(noc_x_start, noc_y_start, target_address + first_chunk_size)},
                            first_chunk_size},
                        hop_info.ucast.num_hops);
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
        }
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
