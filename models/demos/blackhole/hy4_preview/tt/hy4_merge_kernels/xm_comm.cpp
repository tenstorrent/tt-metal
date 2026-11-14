// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Fused exchange + merge (fuse-sparse-merge A2), communication core (BRISC), one per (ring direction, comm lane). Sends this chip's partial (o, ms, l) of owner j (the split-kernel output x
// [8 owners][S][18 tiles], DRAM tile pages) to the owner chip's landing buffer: forward cores serve the owners at distance 1..4 (c+d), backward cores 1..3 (c-d). A (owner, user) block is cut into runs
// per merge core g (TPC o tiles + the 2 stat tiles, contiguous in the landing slot). Lane L serves the merge cores g = L, L + NLANE, ...; lane L uses fabric link L % LINKS.
// COMPACT=0: 32x32 tiles, a packet = 2 tiles (4 KiB), TPC/2 + 1 packets per (owner, g); COMPACT=1: 8x32 tiles (rows 0..7: 2 x 256 B per tile), ONE packet of (TPC + 2) x 512 B per (owner, g).
// Every packet carries a fused global-semaphore increment at the owner's merge core.
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
    constexpr uint32_t LANE = get_compile_time_arg_val(5);
    constexpr uint32_t NLANE = get_compile_time_arg_val(6);
    constexpr uint32_t CB_PKT = get_compile_time_arg_val(7);
    constexpr uint32_t ARGBASE = get_compile_time_arg_val(8);
    constexpr uint32_t DBG = get_compile_time_arg_val(9);  // 1: reader does not wait (merge cores run alone), 2: + no sends (reads/open/close only), 3: comm core idle
    constexpr uint32_t COMPACT = get_compile_time_arg_val(10);
    constexpr auto x_args = TensorAccessorArgs<11, 3>();
    constexpr uint32_t TILE_B = 2048, TPR = 18, RING = 8, LT = TPC + 2;
    constexpr uint32_t TB = COMPACT ? 512 : 2048;                        // L1 tile bytes
    constexpr uint32_t PKT_B = COMPACT ? LT * 512 : 2 * TILE_B;          // payload bytes of one packet
    constexpr uint32_t NPK = COMPACT ? 1 : (TPC / 2 + 1);                // packets per (owner, user, merge core)
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
    uint32_t k = 0;  // packet index (read phase), packets are PKT_B_MAX-strided
    constexpr uint32_t STRIDE = 2 * TILE_B;
    auto rd = [&](uint32_t page, uint32_t dst) {
        if constexpr (COMPACT) {
            noc_async_read(xa.get_noc_addr(page, 0), dst, 256);
            noc_async_read(xa.get_noc_addr(page, 512), dst + 256, 256);
        } else {
            noc_async_read(xa.get_noc_addr(page), dst, TILE_B);
        }
    };
    for (uint32_t s = 0; s < S; ++s) {
        for (uint32_t di = 0; di < ND; ++di) {
            const uint32_t d = ND - di;  // farthest first
            const uint32_t j = (DIR == 0) ? ((c + d) & (RING - 1)) : ((c + RING - d) & (RING - 1));
            const uint32_t base = (j * S + s) * TPR;
            for (uint32_t g = LANE; g < NSPLIT; g += NLANE) {
                if constexpr (COMPACT) {
                    for (uint32_t i = 0; i < TPC; ++i) {
                        rd(base + g * TPC + i, pkt + k * STRIDE + i * 512);
                    }
                    rd(base + 16, pkt + k * STRIDE + TPC * 512);
                    rd(base + 17, pkt + k * STRIDE + (TPC + 1) * 512);
                    ++k;
                } else {
                    for (uint32_t p = 0; p < TPC / 2; ++p) {
                        rd(base + g * TPC + 2 * p, pkt + k * STRIDE);
                        rd(base + g * TPC + 2 * p + 1, pkt + k * STRIDE + TILE_B);
                        ++k;
                    }
                    rd(base + 16, pkt + k * STRIDE);
                    rd(base + 17, pkt + k * STRIDE + TILE_B);
                    ++k;
                }
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
            for (uint32_t g = LANE; g < NSPLIT; g += NLANE) {
                const uint32_t m = s * NSPLIT + g;
                const uint32_t mx = get_arg_val<uint32_t>(1 + 2 * m);
                const uint32_t my = get_arg_val<uint32_t>(2 + 2 * m);
                const uint64_t sem = get_noc_addr(mx, my, gsem);
                for (uint32_t p = 0; p < NPK; ++p) {
                    const uint32_t tile = COMPACT ? 0 : ((p < TPC / 2) ? 2 * p : TPC);  // landing tile index within the slot
                    const uint64_t dst = get_noc_addr(mx, my, land_base + (c * LT + tile) * TB);
                    volatile PACKET_HEADER_TYPE* h = (n & 1) ? hdr1 : hdr0;
                    fabric_unicast_noc_fused_unicast_with_atomic_inc(
                        sender,
                        h,
                        pkt + k * STRIDE,
                        PKT_B,
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
