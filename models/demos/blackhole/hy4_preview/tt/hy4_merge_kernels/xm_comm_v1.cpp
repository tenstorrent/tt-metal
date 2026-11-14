// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Fused exchange + merge (fuse-sparse-merge A2), communication core (BRISC), V1 = full 32x32 tiles (the validated path: 13/13 PCC cases, run_x2.log), one per (ring direction, link). Sends this chip's partial (o, ms, l) of owner j (the split-kernel output x [8 owners][S][18 tiles],
// DRAM tile pages) to the owner chip's landing buffer: forward core serves the owners at distance 1..4 (c+d), the backward core 1..3 (c-d). A (owner, user) block of 18 tiles is cut into runs per
// merge core g (TPC o tiles + the 2 stat tiles); a packet = 2 contiguous tiles (4 KiB), fused with a global-semaphore increment at the owner's merge core. Landing layout on the owner (merge core (s, g)):
// slot(src c) * (TPC + 2) + {o tile i | TPC: ms | TPC+1: l}. Link LK serves the merge cores g = LK, LK + LINKS, ...
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    constexpr uint32_t DIR = get_compile_time_arg_val(0);
    constexpr uint32_t ND = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t TPC = get_compile_time_arg_val(3);
    constexpr uint32_t NSPLIT = get_compile_time_arg_val(4);
    constexpr uint32_t LK = get_compile_time_arg_val(5);
    constexpr uint32_t LINKS = get_compile_time_arg_val(6);
    constexpr uint32_t CB_PKT = get_compile_time_arg_val(7);
    constexpr uint32_t ARGBASE = get_compile_time_arg_val(8);
    constexpr uint32_t DBG = get_compile_time_arg_val(9);  // 1: reader does not wait (merge cores run alone), 2: + no sends (reads/open/close only), 3: comm core idle
    constexpr auto x_args = TensorAccessorArgs<10, 3>();
    constexpr uint32_t TILE_B = 2048, TPR = 18, RING = 8, LT = TPC + 2, PKT = 2 * TILE_B;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t land_base = get_common_arg_val<uint32_t>(1);
    const uint32_t gsem = get_common_arg_val<uint32_t>(2);
    const uint32_t c = get_arg_val<uint32_t>(0);  // ring position of this chip
    // merge core coordinates: args 1 + 2m, 2 + 2m for m = s * NSPLIT + g
    const auto xa = TensorAccessor(x_args, x_addr, TILE_B);
    if constexpr (DBG == 3) {
        return;
    }
    size_t arg_idx = ARGBASE;
    auto mgr = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx, 1);
    auto* sender = &mgr.get(0).sender;
    PacketHeaderPool::reset();
    volatile PACKET_HEADER_TYPE* hdr0 = PacketHeaderPool::allocate_header(1);
    volatile PACKET_HEADER_TYPE* hdr1 = PacketHeaderPool::allocate_header(1);

    const uint32_t pkt = get_write_ptr(CB_PKT);
    uint32_t k = 0;
    for (uint32_t s = 0; s < S; ++s) {
        for (uint32_t di = 0; di < ND; ++di) {
            const uint32_t d = ND - di;  // farthest first
            const uint32_t j = (DIR == 0) ? ((c + d) & (RING - 1)) : ((c + RING - d) & (RING - 1));
            const uint32_t base = (j * S + s) * TPR;
            for (uint32_t g = LK; g < NSPLIT; g += LINKS) {
                for (uint32_t p = 0; p < TPC / 2; ++p) {
                    noc_async_read(xa.get_noc_addr(base + g * TPC + 2 * p), pkt + k * PKT, TILE_B);
                    noc_async_read(xa.get_noc_addr(base + g * TPC + 2 * p + 1), pkt + k * PKT + TILE_B, TILE_B);
                    ++k;
                }
                noc_async_read(xa.get_noc_addr(base + 16), pkt + k * PKT, TILE_B);
                noc_async_read(xa.get_noc_addr(base + 17), pkt + k * PKT + TILE_B, TILE_B);
                ++k;
            }
        }
    }
    noc_async_read_barrier();
    mgr.open_finish();
    if constexpr (DBG == 2) {
        mgr.close();
        noc_async_full_barrier();
        return;
    }
    k = 0;
    uint32_t n = 0;
    for (uint32_t s = 0; s < S; ++s) {
        for (uint32_t di = 0; di < ND; ++di) {
            const uint32_t d = ND - di;
            for (uint32_t g = LK; g < NSPLIT; g += LINKS) {
                const uint32_t m = s * NSPLIT + g;
                const uint32_t mx = get_arg_val<uint32_t>(1 + 2 * m);
                const uint32_t my = get_arg_val<uint32_t>(2 + 2 * m);
                const uint64_t sem = get_noc_addr(mx, my, gsem);
                for (uint32_t p = 0; p < TPC / 2 + 1; ++p) {
                    const uint32_t tile = (p < TPC / 2) ? 2 * p : TPC;  // landing tile index within the slot
                    const uint64_t dst = get_noc_addr(mx, my, land_base + (c * LT + tile) * TILE_B);
                    volatile PACKET_HEADER_TYPE* h = (n & 1) ? hdr1 : hdr0;
                    fabric_unicast_noc_fused_unicast_with_atomic_inc(
                        sender,
                        h,
                        pkt + k * PKT,
                        PKT,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true},
                        static_cast<uint8_t>(d));
                    noc_async_writes_flushed();
                    ++k;
                    ++n;
                }
            }
        }
    }
    mgr.close();
    noc_async_full_barrier();
}
