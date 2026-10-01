// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Intra-mesh fabric stress op for ForwardChainStress (2D fabric). ONE worker core per
// chip opens a WorkerToFabricEdmSender on EVERY link toward EVERY intra-mesh neighbor and
// exchanges atomic-inc packets with them. The host builder allocates 2 WORKER semaphores
// per connection, and the max degree is 4 dirs * 2 links = 8 connections, so this lands at
// exactly the 16-semaphore-per-core cap (the earlier overflow was a double-count bug in
// the connection enumeration, not the link count). Two jobs from one op:
//
//   * LEASE TEST — whatever link the D2D sender service forwards on is necessarily among
//     the ones opened here (interior chips route their D2D forward through an intra-mesh
//     first hop), so holding that EDM channel while the D2D sender is leased off exercises
//     the wait/release lease (a broken lease => the open collides on that channel).
//   * ROUTING / DROP TEST — each chip atomic-incs every neighbor's fabric-test
//     GlobalSemaphore, then noc_semaphore_waits on its OWN until this iter's expected
//     count lands. A dropped inc => the wait never completes => this launch's Finish hangs.
//     Intra-mesh ONLY: all of a rank's chips run this in the SAME launch, so the exchange
//     rendezvous locally with no cross-rank (cascade-coupled) dependency => deadlock-free.
//
// Structured as three loops (per the all_gather.hpp adapter pattern): SETUP builds/opens
// each connection and pre-fills a fixed-ring packet header (route + atomic-inc command are
// constant per connection; flush bit OFF); SEND round-robins one non-blocking inc per
// connection per round; CLOSE tears them down. Connections are held open across the send
// + the wait, widening the lease-contention window. The sem is reset each iter (the
// per-launch Finish is a rank-wide barrier, so no neighbor sends the next iter's incs
// before every chip has reset).
//
// RT layout (only the single fabric core gets it):
//   [0] num_connections
//   [1] local_sem_addr        (this chip's fabric-test GlobalSemaphore L1 address)
//   [2] dest_noc_x            (phys x of the fabric core; the sem sits there on EVERY chip)
//   [3] dest_noc_y
//   [4] incs_per_conn
//   [5] expected_per_iter     (= incoming_conns * incs_per_conn; spin target, then reset)
//   then, back to back per connection:
//     [ append_fabric_connection_rt_args block ] [dst_dev_id] [dst_mesh_id]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"  // tt::tt_fabric::WorkerToFabricEdmSender
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

// Blackhole has up to 2 links × 4 directions = 8 connections per chip.
constexpr uint32_t kMaxConnections = 8;

void kernel_main() {
    size_t rt_idx = 0;
    const uint32_t num_connections = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t local_sem_addr = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t dest_noc_x = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t dest_noc_y = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t incs_per_conn = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t expected_per_iter = get_arg_val<uint32_t>(rt_idx++);

    // The neighbor's fabric-test GlobalSemaphore lives at the same (core, L1 addr) on every
    // chip (mesh-wide GlobalSemaphore), so one dest NoC address serves all of them;
    // fabric_set_unicast_route routes each packet to the right chip.
    const uint64_t dest_noc_addr = get_noc_addr(dest_noc_x, dest_noc_y, local_sem_addr);

    tt::tt_fabric::WorkerToFabricEdmSender connections[kMaxConnections];
    volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[kMaxConnections];

    PacketHeaderPool::reset();

    // (1) SETUP — build + open each connection; allocate + pre-fill its header. The route
    //     and the atomic-inc command are constant per connection (same neighbor, inc=1, no
    //     flush), so we fill them once here and only re-send in the loop below.
    for (uint32_t c = 0; c < num_connections; ++c) {
        connections[c] = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_idx);
        const uint32_t dst_dev_id = get_arg_val<uint32_t>(rt_idx++);
        const uint32_t dst_mesh_id = get_arg_val<uint32_t>(rt_idx++);
        connections[c].open_start();
        headers[c] = PacketHeaderPool::allocate_header();
        tt::tt_fabric::fabric_set_unicast_route(
            headers[c], static_cast<uint16_t>(dst_dev_id), static_cast<uint16_t>(dst_mesh_id));
        headers[c]->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{dest_noc_addr, 1, /*flush=*/false});
        connections[c].open_finish();
    }

    // (2) SEND — round-robin over connections, incs_per_conn rounds, non-blocking, no flush.
    for (uint32_t round = 0; round < incs_per_conn; ++round) {
        for (uint32_t c = 0; c < num_connections; ++c) {
            connections[c].wait_for_empty_write_slot();
            connections[c].send_payload_non_blocking_from_address(
                reinterpret_cast<uint32_t>(headers[c]), sizeof(PACKET_HEADER_TYPE));
            // no need to flush since we are not overwriting any header
        }
    }

    // block until this iter's incs from every neighbor have landed, then
    // reset for the next iter.
    volatile tt_l1_ptr uint32_t* local_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_sem_addr);
    noc_semaphore_wait(local_sem, expected_per_iter);
    noc_semaphore_set(local_sem, 0);

    // (3) CLOSE all connections.
    for (uint32_t c = 0; c < num_connections; ++c) {
        connections[c].close();
    }
}
