// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// One-shot all-reduce over the 8 chips of a mesh row (ring), communication core (BRISC), one core per ring direction.
//   Input  : the per-chip partial [32 x 5120] bf16 TILE tensor in DRAM, only rows 0..VR-1 (VR = 8) of every tile are
//   valid. Phase 1: this chip's 7 slices (640 columns = NT = 20 tiles) go to their owner chips: the forward core serves
//   the owners
//            at distance 1..4 (c+d), the backward core the owners at distance 1..3 (c-d). A slice is split into NP
//            chunks of Q tiles (one per reducer core); a chunk is the 'compact' tile format (8x32 tiny tile = rows 0..7
//            of face 0 and face 1, 512 B per tile), one fabric packet (fused write + semaphore increment) per (owner,
//            reducer).
//   Phase 2: the reduced chunk of every reducer (written into this core's staging buffer by the reducer) is multicast
//   to all other
//            chips of the ring (forward: 4 chips, backward: 3 chips) and lands in the landing core of its tile columns.
// Cross-chip buffers are persistent L1 tensors, cross-chip semaphores are global semaphores, so no barrier is needed:
// the all-to-all dependency chain of consecutive calls guarantees that a receive buffer is free when the next call
// writes it.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    constexpr uint32_t DIR = get_compile_time_arg_val(0);  // 0 forward (c+d), 1 backward (c-d)
    constexpr uint32_t ND = get_compile_time_arg_val(1);   // distances served: 4 forward, 3 backward
    constexpr uint32_t PN = get_compile_time_arg_val(2);   // reducer chunks served by this core
    constexpr uint32_t P0 = get_compile_time_arg_val(3);   // first reducer
    constexpr uint32_t Q = get_compile_time_arg_val(4);    // tiles per chunk
    constexpr uint32_t NT = get_compile_time_arg_val(5);   // tiles per slice
    constexpr uint32_t CB_PKT = get_compile_time_arg_val(6);
    constexpr uint32_t CB_STG = get_compile_time_arg_val(7);
    constexpr uint32_t SEM_STG0 = get_compile_time_arg_val(8);  // local semaphore id of reducer p = SEM_STG0 + p
    constexpr uint32_t NP = get_compile_time_arg_val(9);        // reducers
    constexpr uint32_t ARGBASE = get_compile_time_arg_val(10);  // runtime arg index of the fabric connection args
    constexpr uint32_t STOP = get_compile_time_arg_val(11);     // debug: 1 = stop after phase 1 sends
    constexpr uint32_t IN_L1 = get_compile_time_arg_val(
        12);  // 1: the input is L1 width sharded on 8 cores (core j = slice j), args after the landing array
    constexpr auto in_args = TensorAccessorArgs<13>();

    constexpr uint32_t PIECE = 256;
    constexpr uint32_t TILEB = 2 * PIECE;
    constexpr uint32_t CHUNK = Q * TILEB;
    constexpr uint32_t RING = 8;

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t recv_base = get_common_arg_val<uint32_t>(1);
    const uint32_t gsem = get_common_arg_val<uint32_t>(2);
    const uint32_t land_base = get_common_arg_val<uint32_t>(3);
    const uint32_t c = get_arg_val<uint32_t>(0);  // ring position of this chip
    // reducer core coordinates: args 1 + 2p, 2 + 2p ; landing cores: 1 + 2 NP + 2m, 2 + 2 NP + 2m

    DeviceZoneScopedN("C_ALL");
    const auto in_acc = TensorAccessor(in_args, in_addr, 2048);
    auto src_addr = [&](uint32_t j, uint32_t t, uint32_t off) -> uint64_t {
        if constexpr (IN_L1) {
            const uint32_t b = 1 + 2 * NP + 2 * NP * 8;
            return get_noc_addr(
                get_arg_val<uint32_t>(b + 2 * j), get_arg_val<uint32_t>(b + 2 * j + 1), in_addr + t * 2048 + off);
        } else {
            return in_acc.get_noc_addr(j * NT + t, off);
        }
    };

    size_t arg_idx = ARGBASE;
    auto mgr = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx, 1);
    auto* sender = &mgr.get(0).sender;
    PacketHeaderPool::reset();
    volatile PACKET_HEADER_TYPE* hdr0 = PacketHeaderPool::allocate_header(1);
    volatile PACKET_HEADER_TYPE* hdr1 = PacketHeaderPool::allocate_header(1);

    const uint32_t pkt = get_write_ptr(CB_PKT);
    {
        DeviceZoneScopedN("C_READ");
        for (uint32_t di = 0; di < ND; ++di) {
            const uint32_t d = ND - di;  // farthest first
            const uint32_t j = (DIR == 0) ? ((c + d) & (RING - 1)) : ((c + RING - d) & (RING - 1));
            for (uint32_t pi = 0; pi < PN; ++pi) {
                const uint32_t p = P0 + pi;
                const uint32_t dst = pkt + (di * PN + pi) * CHUNK;
                const uint32_t tile0 = j * NT + p * Q;
                for (uint32_t i = 0; i < Q; ++i) {
                    noc_async_read(src_addr(j, p * Q + i, 0), dst + i * TILEB, PIECE);
                    noc_async_read(src_addr(j, p * Q + i, 512), dst + i * TILEB + PIECE, PIECE);
                }
            }
        }
        noc_async_read_barrier();
    }
    {
        DeviceZoneScopedN("C_OPEN");
        mgr.open_finish();
    }

    uint32_t k = 0;
    {
        DeviceZoneScopedN("C_SEND1");
        for (uint32_t di = 0; di < ND; ++di) {
            const uint32_t d = ND - di;
            for (uint32_t pi = 0; pi < PN; ++pi) {
                const uint32_t p = P0 + pi;
                const uint32_t rx = get_arg_val<uint32_t>(1 + 2 * p);
                const uint32_t ry = get_arg_val<uint32_t>(2 + 2 * p);
                // the owner's receive buffer has one slot per REMOTE chip: this chip (c) takes slot c if c < owner else
                // c - 1
                const uint32_t owner = (DIR == 0) ? ((c + d) & (RING - 1)) : ((c + RING - d) & (RING - 1));
                const uint32_t slot = (c < owner) ? c : (c - 1);
                const uint64_t dst = get_noc_addr(rx, ry, recv_base + slot * CHUNK);
                const uint64_t sem = get_noc_addr(rx, ry, gsem);
                volatile PACKET_HEADER_TYPE* h = (k & 1) ? hdr1 : hdr0;
                fabric_unicast_noc_fused_unicast_with_atomic_inc(
                    sender,
                    h,
                    pkt + (di * PN + pi) * CHUNK,
                    CHUNK,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true},
                    static_cast<uint8_t>(d));
                noc_async_writes_flushed();
                ++k;
            }
        }
    }
    if constexpr (STOP == 0) {
        DeviceZoneScopedN("C_P2");
        // ---- phase 2: multicast the reduced chunks
        const uint32_t stg = get_write_ptr(CB_STG);
        for (uint32_t pi = 0; pi < PN; ++pi) {
            const uint32_t p = P0 + pi;
            volatile tt_l1_ptr uint32_t* ssem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_STG0 + p));
            {
                DeviceZoneScopedN("C_WAIT");
                noc_semaphore_wait_min(ssem, 1);
            }
            noc_semaphore_set(ssem, 0);
            const uint32_t m = c * NP + p;  // landing core of this chunk
            const uint32_t lx = get_arg_val<uint32_t>(1 + 2 * NP + 2 * m);
            const uint32_t ly = get_arg_val<uint32_t>(2 + 2 * NP + 2 * m);
            const uint64_t dst = get_noc_addr(lx, ly, land_base);
            const uint64_t sem = get_noc_addr(lx, ly, gsem);
            volatile PACKET_HEADER_TYPE* h = (k & 1) ? hdr1 : hdr0;
            fabric_multicast_noc_fused_unicast_with_atomic_inc(
                sender,
                h,
                stg + p * CHUNK,
                CHUNK,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true},
                static_cast<uint8_t>(1),
                static_cast<uint8_t>(ND));
            noc_async_writes_flushed();
            ++k;
        }
    }
    {
        DeviceZoneScopedN("C_CLOSE");
        mgr.close();
        noc_async_full_barrier();
    }
    noc_async_full_barrier();
}
