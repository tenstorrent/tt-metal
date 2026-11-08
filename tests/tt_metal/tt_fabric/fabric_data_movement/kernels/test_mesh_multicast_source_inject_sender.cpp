// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "test_mesh_multicast_source_inject_common.hpp"

using namespace tt::tt_fabric::mesh::experimental;
using namespace tt::tt_fabric::fabric_router_tests::source_inject;

constexpr uint32_t test_results_address = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);

namespace {

void fill_source_payload(uint32_t source_address, uint32_t payload_size, ValidationPhase phase, uint32_t packet_index) {
    auto* source = reinterpret_cast<tt_l1_ptr uint32_t*>(source_address);
    const uint32_t payload_words = payload_size / sizeof(uint32_t);
    for (uint32_t word = 0; word < payload_words; ++word) {
        source[word] = payload_word(phase, packet_index, word);
    }
}

}  // namespace

void kernel_main() {
    size_t rt_arg_idx = 0;
    const uint32_t data_base = get_arg_val<uint32_t>(rt_arg_idx++);
    const uint32_t payload_size = get_arg_val<uint32_t>(rt_arg_idx++);
    const uint32_t receiver_noc_x = get_arg_val<uint32_t>(rt_arg_idx++);
    const uint32_t receiver_noc_y = get_arg_val<uint32_t>(rt_arg_idx++);
    const uint8_t east = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
    const uint8_t west = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
    const uint8_t north = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
    const uint8_t south = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_arg_idx++));
    const uint32_t num_connections = get_arg_val<uint32_t>(rt_arg_idx++);

    auto* test_results = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(test_results_address);
    for (uint32_t i = 0; i < test_results_size_bytes / sizeof(uint32_t); ++i) {
        test_results[i] = 0;
    }
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    ASSERT(payload_size >= 2 * PAYLOAD_ALIGNMENT);
    ASSERT((payload_size % PAYLOAD_ALIGNMENT) == 0);
    ASSERT(first_scatter_chunk_size(payload_size) != 0);
    ASSERT(second_scatter_chunk_size(payload_size) != 0);

    PacketHeaderPool::reset();
    // Every root output selected for this one logical branch intentionally reuses one encoded header;
    // each API rewrites only its NOC command. On express meshes those outputs can be cardinal plus Z.
    const uint8_t route_id = PacketHeaderPool::allocate_header_n(1);

    tt::tt_fabric::RoutingPlaneConnectionManager connections;
    open_connections(connections, num_connections, rt_arg_idx);

    const MeshMcastRange branch{east, west, north, south};

    // Every atomic-bearing packet has its own destination word. An aggregate count can hide one
    // missing delivery behind one duplicate; individual words make those cases observable as 0 and 2.
    for (uint32_t packet = 0; packet < ATOMIC_PACKET_COUNT; ++packet) {
        fabric_multicast_source_inject_noc_unicast_atomic_inc(
            connections,
            route_id,
            branch,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                get_noc_addr(receiver_noc_x, receiver_noc_y, counter_address(data_base, payload_size, packet)),
                1,
                true});

        // Source injection copies the shared header to each selected EDM non-blockingly. Manager
        // credits do not mean those NOC reads of worker L1 have completed, so flush before rewriting it.
        noc_async_writes_flushed();
    }

    for (uint32_t packet = 0; packet < PAYLOAD_PACKET_COUNT; ++packet) {
        fill_source_payload(data_base, payload_size, ValidationPhase::FUSED_WRITE, packet);
        // flush=true orders this packet's payload before its matching atomic. It does not order
        // different packets or API phases, which is why each packet has a distinct counter.
        fabric_multicast_source_inject_noc_fused_unicast_with_atomic_inc(
            connections,
            route_id,
            branch,
            data_base,
            payload_size,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                get_noc_addr(receiver_noc_x, receiver_noc_y, fused_write_address(data_base, payload_size, packet)),
                get_noc_addr(
                    receiver_noc_x,
                    receiver_noc_y,
                    counter_address(data_base, payload_size, ATOMIC_PACKET_COUNT + packet)),
                1,
                true});
        noc_async_writes_flushed();
    }

    const uint16_t first_chunk = static_cast<uint16_t>(first_scatter_chunk_size(payload_size));
    for (uint32_t packet = 0; packet < PAYLOAD_PACKET_COUNT; ++packet) {
        fill_source_payload(data_base, payload_size, ValidationPhase::FUSED_SCATTER, packet);
        fabric_multicast_source_inject_noc_fused_scatter_write_atomic_inc(
            connections,
            route_id,
            branch,
            data_base,
            payload_size,
            tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                {get_noc_addr(receiver_noc_x, receiver_noc_y, scatter_first_address(data_base, payload_size, packet)),
                 get_noc_addr(receiver_noc_x, receiver_noc_y, scatter_second_address(data_base, payload_size, packet))},
                get_noc_addr(
                    receiver_noc_x,
                    receiver_noc_y,
                    counter_address(data_base, payload_size, ATOMIC_PACKET_COUNT + PAYLOAD_PACKET_COUNT + packet)),
                {first_chunk},
                1,
                true});
        noc_async_writes_flushed();
    }

    for (uint32_t packet = 0; packet < PAYLOAD_PACKET_COUNT; ++packet) {
        fill_source_payload(data_base, payload_size, ValidationPhase::PLAIN_WRITE, packet);
        // Full-payload checking catches loss, corruption, and wrong-chip delivery. An identical
        // duplicate plain write is inherently idempotent and has no second side effect to count.
        fabric_multicast_source_inject_noc_unicast_write(
            connections,
            route_id,
            branch,
            data_base,
            payload_size,
            tt::tt_fabric::NocUnicastCommandHeader{
                get_noc_addr(receiver_noc_x, receiver_noc_y, plain_write_address(data_base, payload_size, packet))});
        noc_async_writes_flushed();
    }

    // The per-packet flushes make the reused worker buffers safe to mutate, but teardown must also
    // wait for payload/header writes and credit atomics to land in every selected EDM.
    noc_async_full_barrier();
    close_connections(connections);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
}
