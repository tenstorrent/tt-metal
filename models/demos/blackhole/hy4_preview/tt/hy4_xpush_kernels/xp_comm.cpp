// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B47a XPUSH (flag HY4_B47A_XPUSH): partial exchange of the compact owner-layout split output INSIDE the split SDPA
// program. Runs on extra cores (one per ring direction x lane) of the split program. The split writer leaders write
// their bands into the DRAM output tile row of owner j as before and then increment the global semaphore gs[j] on the
// comm cores serving owner j (S increments per call, one per user). A comm core polls its owners; when gs[j] == S it
// resets it, reads this lane's packets (2 contiguous tiles = 4 KiB, 9 packets per owner: tiles (0,1) .. (14,15),
// (16,17) = ms,l) of owner j from the DRAM output and sends them to the landing tensor of chip j (DRAM, 4 KiB pages,
// page c*9+p of source c) with a fused atomic increment of flag[c] on the flag core of chip j (monotonic, the consumer
// keeps a launch counter). The own slot (d == 0) is a local copy + increment. The program never waits for a peer.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    constexpr uint32_t DIR = get_compile_time_arg_val(0);
    constexpr uint32_t ND = get_compile_time_arg_val(1);  // peers in this direction (4 fwd, 3 bwd)
    constexpr uint32_t LANE = get_compile_time_arg_val(2);
    constexpr uint32_t LANES = get_compile_time_arg_val(3);
    constexpr uint32_t CB_PKT = get_compile_time_arg_val(4);
    constexpr uint32_t ARGBASE = get_compile_time_arg_val(5);
    constexpr uint32_t NH = get_compile_time_arg_val(6);
    constexpr uint32_t S = get_compile_time_arg_val(7);  // users = increments per owner per call
    constexpr uint32_t DBG = get_compile_time_arg_val(8);
    constexpr auto o_args = TensorAccessorArgs<9, 0>();
    constexpr auto l_args =
        TensorAccessorArgs<o_args.next_compile_time_args_offset(), o_args.next_common_runtime_args_offset()>();
    constexpr uint32_t TILE_B = 2048, PKT = 2 * TILE_B, NPK = 9, TPR = 18, RING = 8;
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t land_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(0);                                  // ring position of this chip
    const uint32_t fx = get_arg_val<uint32_t>(1), fy = get_arg_val<uint32_t>(2);  // flag core (physical)
    const uint32_t flag_c =
        get_arg_val<uint32_t>(3 + c);   // flag word of this chip as a SOURCE (same address on every chip)
    const uint32_t gs_base = 3 + RING;  // gs addresses of the owners (absolute ring index j) at args gs_base + j
    // pages of the output tensor: owner j tile row: tiles j*TPR .. j*TPR+17 (TILE layout, 2 KiB pages, TR = 1)
    const auto oa = TensorAccessor(o_args, o_addr, TILE_B);
    const auto la = TensorAccessor(l_args, land_addr, PKT);
    const uint64_t flag_noc = get_noc_addr(fx, fy, flag_c);
    size_t arg_idx = ARGBASE;
    auto mgr = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx, 1);
    auto* sender = &mgr.get(0).sender;
    PacketHeaderPool::reset();
    volatile PACKET_HEADER_TYPE* hdr[NH];
    for (uint32_t i = 0; i < NH; ++i) {
        hdr[i] = PacketHeaderPool::allocate_header(1);
    }
    const uint32_t pkt = get_write_ptr(CB_PKT);
    mgr.open_finish();

    // owners served: DIR 0: d = 1..ND (hops d) and the own slot (d = 0, handled by this direction's lanes too); DIR 1:
    // d = 1..ND (backward)
    constexpr uint32_t NOWN = ND + (DIR == 0 ? 1 : 0);
    uint32_t pending = (1u << NOWN) - 1;  // bit i: owner index i, d = i (DIR 0: d=0 is self) or d = i + 1 (DIR 1)
    uint32_t n = 0;
    while (pending) {
        for (uint32_t i = 0; i < NOWN; ++i) {
            if (!(pending & (1u << i))) {
                continue;
            }
            const uint32_t d = (DIR == 0) ? i : i + 1;
            const uint32_t j = (DIR == 0) ? (c + d) % RING : (c + RING - d) % RING;
            volatile tt_l1_ptr uint32_t* gs =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(gs_base + j));
            if (*gs < S) {
                continue;
            }
            *gs = 0;
            pending &= ~(1u << i);
            // read this lane's packets of owner j
            uint32_t k = 0;
            for (uint32_t p = LANE; p < NPK; p += LANES) {
                const uint32_t t0 = j * TPR + 2 * p;
                noc_async_read(oa.get_noc_addr(t0), pkt + k * PKT, TILE_B);
                noc_async_read(oa.get_noc_addr(t0 + 1), pkt + k * PKT + TILE_B, TILE_B);
                ++k;
            }
            noc_async_read_barrier();
            k = 0;
            if (d == 0) {
                for (uint32_t p = LANE; p < NPK; p += LANES) {
                    noc_async_write(pkt + k * PKT, la.get_noc_addr(c * NPK + p), PKT);
                    ++k;
                }
                noc_async_write_barrier();
                for (uint32_t p = LANE; p < NPK; p += LANES) {
                    noc_semaphore_inc(flag_noc, 1);
                }
                noc_async_atomic_barrier();
            } else if constexpr (DBG == 0) {
                for (uint32_t p = LANE; p < NPK; p += LANES) {
                    const uint64_t dst = la.get_noc_addr(c * NPK + p);
                    volatile PACKET_HEADER_TYPE* h = hdr[n % NH];
                    fabric_unicast_noc_fused_unicast_with_atomic_inc(
                        sender,
                        h,
                        pkt + k * PKT,
                        PKT,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, flag_noc, 1, true},
                        static_cast<uint8_t>(d));
                    ++n;
                    ++k;
                }
                noc_async_writes_flushed();
            }
        }
    }
    mgr.close();
    noc_async_full_barrier();
}
