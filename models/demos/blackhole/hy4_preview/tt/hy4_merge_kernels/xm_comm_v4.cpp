// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B26 exchange communication core (flag HY4_XM_V4 / HY4_XM_HALF / HY4_XM_NH): xm_comm_v1 with (a) NH rotating packet
// headers, the writes are flushed only when all NH headers are in flight (v1 flushed after every packet), (b) HALF=1:
// only the first 1 KiB of every 2 KiB tile is moved (faces 0 and 1 = tile rows 0..15; the 8 valid head rows are
// inside), a run (TPC o tiles + ms + l = TPC+2 half tiles, TPC == 2 -> exactly 4 KiB) is ONE packet landing
// contiguously (1 KiB pitch) in the merge core's landing buffer (the merge compute reads it through a CB with page size
// 1 KiB; rows 16..31 of every tile then alias the next tile = don't-care rows).
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
    constexpr uint32_t DBG = get_compile_time_arg_val(9);
    constexpr uint32_t HALF = get_compile_time_arg_val(10);
    constexpr uint32_t NH = get_compile_time_arg_val(11);
    constexpr uint32_t SLOTSEM = get_compile_time_arg_val(
        12);  // 1: the semaphore of this chip's slot (ring position c) is runtime arg ARGBASE_SEM + ... see below
    constexpr uint32_t SEMARG =
        get_compile_time_arg_val(13);  // runtime arg index of the 8 per-slot semaphore addresses
    constexpr uint32_t DRAML = get_compile_time_arg_val(
        14);  // 1: the landing buffer is DRAM (4 KiB pages: page (m * 8 + c) * 2 + p), TPC == 2, no HALF
    constexpr auto x_args = TensorAccessorArgs<15, 3>();
    constexpr auto l_args =
        TensorAccessorArgs<x_args.next_compile_time_args_offset(), x_args.next_common_runtime_args_offset()>();
    constexpr uint32_t TILE_B = 2048, TPR = 18, RING = 8, LT = TPC + 2;
    constexpr uint32_t UNIT = HALF ? 1024 : 2048;  // bytes moved per tile
    constexpr uint32_t PKT = HALF ? LT * 1024 : 2 * TILE_B;
    constexpr uint32_t NPK = HALF ? 1 : (TPC / 2 + 1);  // packets per (owner, user, merge core)
    static_assert(!HALF || (LT * 1024 == 4096), "HALF needs TPC == 2");
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t land_base = get_common_arg_val<uint32_t>(1);
    const uint32_t gsem = get_common_arg_val<uint32_t>(2);
    const uint32_t c = get_arg_val<uint32_t>(0);  // ring position of this chip
    const uint32_t gsem_use =
        SLOTSEM ? get_arg_val<uint32_t>(SEMARG + c) : gsem;  // semaphore address (same on every core)
    const auto xa = TensorAccessor(x_args, x_addr, TILE_B);
    const auto la = TensorAccessor(l_args, land_base, 4096);
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
    uint32_t k = 0;
    {
        DeviceZoneScopedN("XC_READ");
        for (uint32_t s = 0; s < S; ++s) {
            for (uint32_t di = 0; di < ND; ++di) {
                const uint32_t d = ND - di;  // farthest first
                const uint32_t j = (DIR == 0) ? ((c + d) & (RING - 1)) : ((c + RING - d) & (RING - 1));
                const uint32_t base = (j * S + s) * TPR;
                for (uint32_t g = LK; g < NSPLIT; g += LINKS) {
                    if constexpr (HALF) {
                        for (uint32_t i = 0; i < TPC; ++i) {
                            noc_async_read(xa.get_noc_addr(base + g * TPC + i), pkt + k * PKT + i * UNIT, UNIT);
                        }
                        noc_async_read(xa.get_noc_addr(base + 16), pkt + k * PKT + TPC * UNIT, UNIT);
                        noc_async_read(xa.get_noc_addr(base + 17), pkt + k * PKT + (TPC + 1) * UNIT, UNIT);
                        ++k;
                    } else {
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
        }
        noc_async_read_barrier();
    }
    {
        DeviceZoneScopedN("XC_OPEN");
        mgr.open_finish();
    }
    k = 0;
    uint32_t n = 0;
    {
        DeviceZoneScopedN("XC_SEND");
        for (uint32_t s = 0; s < S; ++s) {
            for (uint32_t di = 0; di < ND; ++di) {
                const uint32_t d = ND - di;
                for (uint32_t g = LK; g < NSPLIT; g += LINKS) {
                    const uint32_t m = s * NSPLIT + g;
                    const uint32_t mx = get_arg_val<uint32_t>(1 + 2 * m);
                    const uint32_t my = get_arg_val<uint32_t>(2 + 2 * m);
                    const uint64_t sem = get_noc_addr(mx, my, gsem_use);
                    for (uint32_t p = 0; p < NPK; ++p) {
                        uint32_t dst_off;
                        if constexpr (HALF) {
                            dst_off = c * LT * 1024;
                        } else {
                            const uint32_t tile = (p < TPC / 2) ? 2 * p : TPC;  // landing tile index within the slot
                            dst_off = (c * LT + tile) * TILE_B;
                        }
                        uint64_t dst;
                        if constexpr (DRAML) {
                            dst = la.get_noc_addr((m * RING + c) * 2 + p);
                        } else {
                            dst = get_noc_addr(mx, my, land_base + dst_off);
                        }
                        volatile PACKET_HEADER_TYPE* h = hdr[n % NH];
                        fabric_unicast_noc_fused_unicast_with_atomic_inc(
                            sender,
                            h,
                            pkt + k * PKT,
                            PKT,
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true},
                            static_cast<uint8_t>(d));
                        ++n;
                        if ((n % NH) == 0) {
                            noc_async_writes_flushed();
                        }
                        ++k;
                    }
                }
            }
        }
        noc_async_writes_flushed();
    }
    {
        DeviceZoneScopedN("XC_CLOSE");
        mgr.close();
        noc_async_full_barrier();
    }
}
