// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender for the per-page sparse-multicast test: one payload is delivered to a set of
// non-contiguous colinear chips (selected by a hop bitmask), each at its OWN destination
// address. address[i] targets the i-th writing chip in ascending hop order, matching the
// order in which the router advances write_idx. Single direction.

#include "tt_metal/fabric/hw/inc/linear/api.h"
using namespace tt::tt_fabric::linear::experimental;
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "test_linear_common.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tests/tt_metal/tt_fabric/common/test_host_kernel_common.hpp"

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);
constexpr uint32_t target_base_address = get_compile_time_arg_val(2);
constexpr uint32_t num_dests = get_compile_time_arg_val(3);

void kernel_main() {
    size_t rt_arg_idx = 0;
    uint32_t source_l1_buffer_address = get_arg_val<uint32_t>(rt_arg_idx++);
    uint16_t packet_payload_size_bytes = static_cast<uint16_t>(get_arg_val<uint32_t>(rt_arg_idx++));
    uint32_t num_packets = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t recv_noc_x = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t recv_noc_y = get_arg_val<uint32_t>(rt_arg_idx++);
    uint16_t hop_mask = static_cast<uint16_t>(get_arg_val<uint32_t>(rt_arg_idx++));

    // Single direction: one first-hop connection, one packet header.
    auto route_id = PacketHeaderPool::allocate_header_n(1);
    tt::tt_fabric::RoutingPlaneConnectionManager connections;
    open_connections(connections, 1, rt_arg_idx);

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();

    for (uint32_t p = 0; p < num_packets; p++) {
        time_seed = prng_next(time_seed);
        tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address);
        fill_packet_data(start_addr, packet_payload_size_bytes / 16, time_seed);

        // Distinct destination address per writing chip; slot i (ascending hop order) lands at
        // target_base + i * payload. num_packets is assumed 1 (see host) so streaming does not overlap.
        // One page per chip: counts are all 1 and num_chips == num_dests.
        tt::tt_fabric::NocSparseMulticastWriteCommandHeader cmd;
        for (uint8_t i = 0; i < num_dests; i++) {
            cmd.noc_address[i] =
                get_noc_addr(recv_noc_x, recv_noc_y, target_base_address + i * packet_payload_size_bytes);
            cmd.counts[i] = 1;
        }
        cmd.num_dests = num_dests;
        cmd.num_chips = num_dests;

        PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* hdr, uint8_t) {
            fabric_sparse_multicast_noc_scatter_write(
                &connections.get(0).sender, hdr, source_l1_buffer_address, packet_payload_size_bytes, cmd, hop_mask);
        });
        noc_async_writes_flushed();
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    // send_payload_flush_non_blocking confirms the writes left the NIU but not that they landed in the
    // EDM's L1; closing with a send mid-flight would let the EDM advance its slot bookkeeping before the
    // bytes arrive. Barrier before close so payload/header writes (and credit atomics) have completed.
    noc_async_full_barrier();
    close_connections(connections);

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
}
