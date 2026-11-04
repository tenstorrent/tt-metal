// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Persistent MoE program, exchange core (BRISC): the chip's fabric endpoint for BOTH row exchanges of the layer's
// MoE block inside one program. Phase A (gather): the row_exchange gather protocol (custom_ops/row_exchange,
// xchg_row_exchange.cpp MODE 0); once the workers have written the gathered tiles it multicasts a 1 into the
// expert cores' four PERSIST semaphores (one per expert kernel: the experts start). Phase B (reduce): the reduce
// protocol (MODE 1) after the workers staged their pieces (their readers wait for the down cores' output).
//
// compile-time: [CB_SEND_A, CB_SEND_B, SEM_STAGED, SEM_GO, NW_A, NW_B, NCHUNKS_A, NCHUNKS_B, PS, CPP, NP_A, NP_B,
//                NSRC, PSEM0, <TensorAccessorArgs recvA>, <TensorAccessorArgs recvB>]   (PSEM0..PSEM0+3 multicast)
// runtime: [my_idx, xchg_noc, recvA_addr, arrivedA_addr, recvB_addr, arrivedB_addr, worker nocs x NW_A, n_peers,
//           (idx, dev_id, mesh_id, group) x n_peers, n_groups, n_rects, (x0, y0, x1, y1, num_dests, loopback) x
//           n_rects, <fabric connection args> x n_groups]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

void kernel_main() {
    constexpr uint32_t CB_SEND_A = get_compile_time_arg_val(0);
    constexpr uint32_t CB_SEND_B = get_compile_time_arg_val(1);
    constexpr uint32_t SEM_STAGED = get_compile_time_arg_val(2);
    constexpr uint32_t SEM_GO = get_compile_time_arg_val(3);
    constexpr uint32_t NW_A = get_compile_time_arg_val(4);
    constexpr uint32_t NW_B = get_compile_time_arg_val(5);
    constexpr uint32_t NCHUNKS_A = get_compile_time_arg_val(6);
    constexpr uint32_t NCHUNKS_B = get_compile_time_arg_val(7);
    constexpr uint32_t PS = get_compile_time_arg_val(8);
    constexpr uint32_t CPP = get_compile_time_arg_val(9);
    constexpr uint32_t NP_A = get_compile_time_arg_val(10);
    constexpr uint32_t NP_B = get_compile_time_arg_val(11);
    constexpr uint32_t NSRC = get_compile_time_arg_val(12);
    constexpr uint32_t PSEM0 = get_compile_time_arg_val(13);
    constexpr auto recvA_args = TensorAccessorArgs<14>();
    constexpr auto recvB_args = TensorAccessorArgs<recvA_args.next_compile_time_args_offset()>();
    constexpr uint32_t PKT = CPP * PS;
    constexpr uint32_t SLOT_B = NCHUNKS_B * PS;
    constexpr uint32_t MAX_PEERS = 8, MAX_GROUPS = 4;

    size_t a = 0;
    const uint32_t my_idx = get_arg_val<uint32_t>(a++);
    const uint32_t xchg_noc = get_arg_val<uint32_t>(a++);
    const uint32_t recvA_addr = get_arg_val<uint32_t>(a++);
    const uint32_t arrivedA_addr = get_arg_val<uint32_t>(a++);
    const uint32_t recvB_addr = get_arg_val<uint32_t>(a++);
    const uint32_t arrivedB_addr = get_arg_val<uint32_t>(a++);
    uint32_t wnoc[NW_A];
    for (uint32_t i = 0; i < NW_A; ++i) {
        wnoc[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t n_peers = get_arg_val<uint32_t>(a++);
    uint32_t pidx[MAX_PEERS], pdev[MAX_PEERS], pmesh[MAX_PEERS], pgrp[MAX_PEERS];
    for (uint32_t j = 0; j < n_peers; ++j) {
        pidx[j] = get_arg_val<uint32_t>(a++);
        pdev[j] = get_arg_val<uint32_t>(a++);
        pmesh[j] = get_arg_val<uint32_t>(a++);
        pgrp[j] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t n_groups = get_arg_val<uint32_t>(a++);
    // the grid's multicast rects (Tensix-only, physically contiguous): x0, y0, x1, y1, num_dests, loopback
    const uint32_t n_rects = get_arg_val<uint32_t>(a++);
    uint32_t rect[4][6];
    for (uint32_t r = 0; r < n_rects; ++r) {
        for (uint32_t f = 0; f < 6; ++f) {
            rect[r][f] = get_arg_val<uint32_t>(a++);
        }
    }
    // a multicast rect is walked from its start corner along the NOC's direction: on NOC 1 the far corner is the
    // start (tt-metal's factories swap the corners for NOC_1); the wrong orientation is a wrap-around multicast that
    // reaches the grid's edges only (the 2026-11-04 gate84_B4_persist_g11b triage)
    if (noc_index == 1) {
        for (uint32_t r = 0; r < n_rects; ++r) {
            const uint32_t x0 = rect[r][0], y0 = rect[r][1];
            rect[r][0] = rect[r][2];
            rect[r][1] = rect[r][3];
            rect[r][2] = x0;
            rect[r][3] = y0;
        }
    }
    tt::tt_fabric::WorkerToFabricEdmSender senders[MAX_GROUPS];
    for (uint32_t g = 0; g < n_groups; ++g) {
        senders[g] = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    }

    const auto s_recvA = TensorAccessor(recvA_args, recvA_addr, PKT);
    const auto s_recvB = TensorAccessor(recvB_args, recvB_addr, PKT);
    const uint32_t sendA = get_write_ptr(CB_SEND_A);
    const uint32_t sendB = get_write_ptr(CB_SEND_B);
    volatile tt_l1_ptr uint32_t* staged = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_STAGED));
    volatile tt_l1_ptr uint32_t* arrivedA = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrivedA_addr);
    volatile tt_l1_ptr uint32_t* arrivedB = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrivedB_addr);
    const uint32_t go_addr = get_semaphore(SEM_GO);
    const uint32_t xx = xchg_noc >> 16, xy = xchg_noc & 0xffff;
    const uint64_t semA_noc = get_noc_addr(xx, xy, arrivedA_addr);
    const uint64_t semB_noc = get_noc_addr(xx, xy, arrivedB_addr);

    for (uint32_t g = 0; g < n_groups; ++g) {
        senders[g].open();
    }
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header(1);
    auto send = [&](uint32_t j, uint32_t src, uint32_t bytes, uint64_t dst, uint64_t sem_noc) {
        fabric_set_unicast_route(hdr, static_cast<uint16_t>(pdev[j]), static_cast<uint16_t>(pmesh[j]));
        hdr->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem_noc, 1, true}, bytes);
        auto& s = senders[pgrp[j]];
        s.wait_for_empty_write_slot();
        s.send_payload_without_header_non_blocking_from_address(src, bytes);
        s.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
    };
    auto go = [&](uint32_t nw) {
        for (uint32_t i = 0; i < nw; ++i) {
            noc_semaphore_inc(get_noc_addr(wnoc[i] >> 16, wnoc[i] & 0xffff, go_addr), 1);
        }
    };

    // ---- phase A: the gather (one payload for every peer)
    noc_semaphore_wait_min(staged, NW_A);
    for (uint32_t j = 0; j < n_peers; ++j) {
        for (uint32_t p = 0; p < NP_A; ++p) {
            const uint32_t left = NCHUNKS_A - p * CPP;
            const uint32_t bytes = (left < CPP ? left : CPP) * PS;
            send(
                j,
                sendA + p * PKT,
                bytes,
                s_recvA.get_noc_addr(my_idx * NP_A + p, 0, tt::tt_fabric::edm_to_local_chip_noc),
                semA_noc);
        }
    }
    noc_async_writes_flushed();
    noc_semaphore_wait_min(arrivedA, n_peers * NP_A);
    go(NW_A);
    // the workers wrote the gathered tiles: release the experts (one multicast per expert kernel semaphore)
    noc_semaphore_wait_min(staged, 2 * NW_A);
    noc_semaphore_set(staged, 0);  // before the release: a worker's phase-B increment can only follow it
    noc_async_atomic_barrier();
    for (uint32_t s = 0; s < 4; ++s) {
        volatile tt_l1_ptr uint32_t* one = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(PSEM0 + s));
        noc_semaphore_set(one, 1);
        for (uint32_t r = 0; r < n_rects; ++r) {
            const uint64_t dst =
                get_noc_multicast_addr(rect[r][0], rect[r][1], rect[r][2], rect[r][3], get_semaphore(PSEM0 + s));
            if (rect[r][5]) {
                noc_semaphore_set_multicast_loopback_src(get_semaphore(PSEM0 + s), dst, rect[r][4]);
            } else {
                noc_semaphore_set_multicast(get_semaphore(PSEM0 + s), dst, rect[r][4]);
            }
        }
    }
    noc_async_write_barrier();
    noc_semaphore_set(arrivedA, 0);

    // ---- phase B: the reduce (one payload per peer: its users' rows)
    noc_semaphore_wait_min(staged, NW_B);
    for (uint32_t j = 0; j < n_peers; ++j) {
        const uint32_t src_slot = sendB + pidx[j] * SLOT_B;
        for (uint32_t p = 0; p < NP_B; ++p) {
            const uint32_t left = NCHUNKS_B - p * CPP;
            const uint32_t bytes = (left < CPP ? left : CPP) * PS;
            send(
                j,
                src_slot + p * PKT,
                bytes,
                s_recvB.get_noc_addr(my_idx * NP_B + p, 0, tt::tt_fabric::edm_to_local_chip_noc),
                semB_noc);
        }
    }
    noc_async_writes_flushed();
    for (uint32_t g = 0; g < n_groups; ++g) {
        senders[g].close();
    }
    noc_semaphore_wait_min(arrivedB, n_peers * NP_B);
    go(NW_B);
    noc_semaphore_set(staged, 0);
    noc_semaphore_set(arrivedB, 0);
    noc_async_atomic_barrier();
    noc_async_full_barrier();
}
