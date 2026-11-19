// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Row exchange, exchange core (BRISC): the chip's one fabric endpoint. Waits until every worker staged its pieces in
// CB_SEND, then sends every peer its packets (CPP consecutive chunk pieces of PS bytes each, one fabric unicast of
// payload + fused atomic increment of the peer's arrival semaphore, landing in the peer's persistent receive staging
// at this chip's source slot), waits for the peers' packets to land here, releases the workers (GO) and resets the
// counters. MODE 0 / 1 (gather / reduce over the mesh rows): one phase, NP packets per peer. MODE 2 (all-reduce over
// the mesh columns): two phases -- the pieces of the chunks a peer owns go to that peer (owner p: chunks
// [p * CPO, (p + 1) * CPO)), the owners' workers sum them, then every owner sends its summed chunks to every peer.
//
// compile-time: [MODE, CB_SEND, SEM_STAGED, SEM_GO, NWORKERS, NCHUNKS, PS, CPP, NP, NSRC, <TensorAccessorArgs recv>]
//   (NP: packets per peer per phase, = ceil(CPO / CPP) in MODE 2; NSRC: participants = mesh rows / columns)
// runtime: [my_idx, xchg_noc, recv_addr, arrived_sem_addr, worker nocs x NWORKERS, n_peers,
//           (idx, dev_id, mesh_id, group) x n_peers, n_groups, <fabric connection args> x n_groups]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t CB_SEND = get_compile_time_arg_val(1);
    constexpr uint32_t SEM_STAGED = get_compile_time_arg_val(2);
    constexpr uint32_t SEM_GO = get_compile_time_arg_val(3);
    constexpr uint32_t NWORKERS = get_compile_time_arg_val(4);
    constexpr uint32_t NCHUNKS = get_compile_time_arg_val(5);
    constexpr uint32_t PS = get_compile_time_arg_val(6);
    constexpr uint32_t CPP = get_compile_time_arg_val(7);
    constexpr uint32_t NP = get_compile_time_arg_val(8);
    constexpr uint32_t NSRC = get_compile_time_arg_val(9);
    constexpr auto recv_args = TensorAccessorArgs<10>();
    constexpr uint32_t PKT = CPP * PS;
    constexpr uint32_t SLOT = NCHUNKS * PS;
    constexpr uint32_t CPO = (NCHUNKS + NSRC - 1) / NSRC;  // chunks per owner (MODE 2)
    constexpr uint32_t MAX_PEERS = 8, MAX_GROUPS = 4;
    static_assert(NSRC <= MAX_PEERS, "peer table");

    size_t a = 0;
    const uint32_t my_idx = get_arg_val<uint32_t>(a++);
    const uint32_t xchg_noc = get_arg_val<uint32_t>(a++);
    const uint32_t recv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t arrived_addr = get_arg_val<uint32_t>(a++);
    uint32_t wnoc[NWORKERS];
    for (uint32_t i = 0; i < NWORKERS; ++i) {
        wnoc[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t n_peers = get_arg_val<uint32_t>(a++);
    uint32_t prow[MAX_PEERS], pdev[MAX_PEERS], pmesh[MAX_PEERS], pgrp[MAX_PEERS];
    for (uint32_t j = 0; j < n_peers; ++j) {
        prow[j] = get_arg_val<uint32_t>(a++);
        pdev[j] = get_arg_val<uint32_t>(a++);
        pmesh[j] = get_arg_val<uint32_t>(a++);
        pgrp[j] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t n_groups = get_arg_val<uint32_t>(a++);
    tt::tt_fabric::WorkerToFabricEdmSender senders[MAX_GROUPS];
    for (uint32_t g = 0; g < n_groups; ++g) {
        senders[g] = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    }

    const auto s_recv = TensorAccessor(recv_args, recv_addr, PKT);
    const uint32_t send_base = get_write_ptr(CB_SEND);
    volatile tt_l1_ptr uint32_t* staged = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_STAGED));
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrived_addr);
    const uint32_t go_addr = get_semaphore(SEM_GO);
    const uint32_t xx = xchg_noc >> 16, xy = xchg_noc & 0xffff;
    // the peers' copies of the arrival semaphore and the receive staging sit at this chip's addresses (one mesh
    // allocation each); the header re-encodes both for the router's NoC
    const uint64_t sem_noc = get_noc_addr(xx, xy, arrived_addr);

    // MODE 2 helpers: the chunks owner p holds and its packets per phase
    auto owned = [&](uint32_t p) -> uint32_t {
        return (p * CPO >= NCHUNKS) ? 0u : ((NCHUNKS - p * CPO < CPO) ? NCHUNKS - p * CPO : CPO);
    };
    auto npk = [&](uint32_t p) -> uint32_t { return (owned(p) + CPP - 1) / CPP; };

    // 1. the workers' pieces are in the send staging
    noc_semaphore_wait_min(staged, NWORKERS);

    // 2. the packets
    for (uint32_t g = 0; g < n_groups; ++g) {
        senders[g].open();
    }
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header(1);
    auto send = [&](uint32_t j, uint32_t src, uint32_t bytes, uint32_t page) {
        const uint64_t dst = s_recv.get_noc_addr(page, 0, tt::tt_fabric::edm_to_local_chip_noc);
        fabric_set_unicast_route(hdr, static_cast<uint16_t>(pdev[j]), static_cast<uint16_t>(pmesh[j]));
        hdr->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem_noc, 1, true}, bytes);
        auto& s = senders[pgrp[j]];
        s.wait_for_empty_write_slot();
        s.send_payload_without_header_non_blocking_from_address(src, bytes);
        s.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
    };
    if constexpr (MODE < 2) {
        for (uint32_t j = 0; j < n_peers; ++j) {
            const uint32_t src_slot = (MODE == 0) ? send_base : send_base + prow[j] * SLOT;
            for (uint32_t p = 0; p < NP; ++p) {
                const uint32_t left = NCHUNKS - p * CPP;
                const uint32_t bytes = (left < CPP ? left : CPP) * PS;
                send(j, src_slot + p * PKT, bytes, my_idx * NP + p);
            }
        }
        noc_async_writes_flushed();
        for (uint32_t g = 0; g < n_groups; ++g) {
            senders[g].close();
        }
        // 3. the peers' packets landed: release the workers
        noc_semaphore_wait_min(arrived, n_peers * NP);
        for (uint32_t i = 0; i < NWORKERS; ++i) {
            noc_semaphore_inc(get_noc_addr(wnoc[i] >> 16, wnoc[i] & 0xffff, go_addr), 1);
        }
    } else {
        // phase 1: every peer gets this chip's pieces of the chunks it owns; page (phase, source, packet)
        uint32_t a2 = 0;
        for (uint32_t j = 0; j < n_peers; ++j) {
            const uint32_t p = prow[j];
            a2 += npk(p);
            for (uint32_t pk = 0; pk < npk(p); ++pk) {
                const uint32_t left = owned(p) - pk * CPP;
                const uint32_t bytes = (left < CPP ? left : CPP) * PS;
                send(j, send_base + p * SLOT + (p * CPO + pk * CPP) * PS, bytes, (0 * NSRC + my_idx) * NP + pk);
            }
        }
        const uint32_t a1 = n_peers * npk(my_idx);
        noc_semaphore_wait_min(arrived, a1);
        for (uint32_t i = 0; i < NWORKERS; ++i) {
            noc_semaphore_inc(get_noc_addr(wnoc[i] >> 16, wnoc[i] & 0xffff, go_addr), 1);
        }
        // phase 2: the summed chunks this chip owns go to every peer (one more increment per owner worker; a
        // non-owner never increments twice, so the phase-1 count above was exact)
        noc_semaphore_wait_min(staged, NWORKERS + owned(my_idx));
        for (uint32_t j = 0; j < n_peers; ++j) {
            for (uint32_t pk = 0; pk < npk(my_idx); ++pk) {
                const uint32_t left = owned(my_idx) - pk * CPP;
                const uint32_t bytes = (left < CPP ? left : CPP) * PS;
                send(j, send_base + NSRC * SLOT + (my_idx * CPO + pk * CPP) * PS, bytes, (1 * NSRC + my_idx) * NP + pk);
            }
        }
        noc_async_writes_flushed();
        for (uint32_t g = 0; g < n_groups; ++g) {
            senders[g].close();
        }
        noc_semaphore_wait_min(arrived, a1 + a2);
        for (uint32_t i = 0; i < NWORKERS; ++i) {
            noc_semaphore_inc(get_noc_addr(wnoc[i] >> 16, wnoc[i] & 0xffff, go_addr), 1);
        }
    }
    noc_semaphore_set(staged, 0);
    noc_semaphore_set(arrived, 0);
    noc_async_atomic_barrier();
    noc_async_full_barrier();
}
