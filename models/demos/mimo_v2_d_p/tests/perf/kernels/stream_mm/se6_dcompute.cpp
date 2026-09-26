// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Spatially pipelined expert: down projection on its own cores (TRISC). Per virtual expert v (a 128-row sub-block of
// expert e = v / S): y[rows, PCD columns] = h_all(v) @ Wd[:, columns], accumulated in DST over all KT_D K-tiles, one
// row tile at a time (PCD <= 8 accumulators). h_all holds both M-groups' h K-tile-major: [G][KT_D][MTG rows], so a
// gate/up core's slice (its NP K-tiles, all MTG rows) is one contiguous run.
// The expert's down weights stay resident in the in1 ring (RING blocks of [KBLK_D x PCD], filled by se6_dw.cpp) for
// its S sub-blocks and are popped after the last one.
//
// CT: 0 MTG (row tiles per M-group), 1 G, 2 KT_D, 3 KBLK_D, 4 PCD, 5 NUM_EXPERTS, 6 S, 7 SLOT_TILES, 8 RING
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#endif

constexpr uint32_t in1_cb = tt::CBIndex::c_1;
constexpr uint32_t h_all_cb = tt::CBIndex::c_2;
constexpr uint32_t out_cb = tt::CBIndex::c_16;

constexpr uint32_t mtg = get_compile_time_arg_val(0);
constexpr uint32_t groups = get_compile_time_arg_val(1);
constexpr uint32_t kt_d = get_compile_time_arg_val(2);
constexpr uint32_t kblk_d = get_compile_time_arg_val(3);
constexpr uint32_t pcd = get_compile_time_arg_val(4);
constexpr uint32_t num_experts = get_compile_time_arg_val(5);
constexpr uint32_t sub = get_compile_time_arg_val(6);
constexpr uint32_t slot = get_compile_time_arg_val(7);
constexpr uint32_t ring = get_compile_time_arg_val(8);
constexpr uint32_t nblk = kt_d / kblk_d;
constexpr uint32_t hk = 8;
constexpr uint32_t group_tiles = kt_d * mtg;
constexpr uint32_t h_all_tiles = groups * group_tiles;
constexpr uint32_t mt = groups * mtg;
static_assert(pcd <= 8 && kt_d % kblk_d == 0 && kt_d % hk == 0 && ring >= nblk);

uint32_t popped = 0;

FORCE_INLINE uint32_t wblock(uint32_t a) {
    cb_wait_front(in1_cb, (a - popped + 1) * slot);
    return static_cast<uint32_t>(static_cast<int32_t>(a % ring) - static_cast<int32_t>(popped % ring)) * slot;
}

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(h_all_cb, in1_cb, out_cb);
    matmul_block_init(h_all_cb, in1_cb, false, pcd, 1, hk);
#ifdef SE_DYN
    // Dynamic counts: CB 6 holds [n_act, num_v, subs of each active expert] (from the down core's data movement); the
    // in1 ring holds only the active experts, so an active expert's index is its place in the stream.
    cb_wait_front(tt::CBIndex::c_6, 1);
    const uint32_t n_act = read_tile_value(tt::CBIndex::c_6, 0, 0);
    const uint32_t num_v = read_tile_value(tt::CBIndex::c_6, 0, 1);
    uint32_t e = 0, s = 0, subs_e = n_act ? read_tile_value(tt::CBIndex::c_6, 0, 2) : 0;
    for (uint32_t v = 0; v < num_v; ++v) {
        const bool last_sub = s + 1 == subs_e;
#else
    for (uint32_t v = 0; v < num_experts * sub; ++v) {
        const uint32_t e = v / sub;
        const bool last_sub = v % sub == sub - 1;
#endif
        {
#ifdef SE_ZONES
            DeviceZoneScopedN("SE_H_WAIT");
#endif
            cb_wait_front(h_all_cb, h_all_tiles);
        }
#ifdef SE_ZONES
        DeviceZoneScopedN("SE_DOWN");
#endif
        cb_reserve_back(out_cb, mt * pcd);
        for (uint32_t r = 0; r < mt; ++r) {
            const uint32_t h0 = (r / mtg) * group_tiles + r % mtg;
            tile_regs_acquire();
            uint32_t w = 0;
            for (uint32_t kk = 0; kk < kt_d; ++kk) {
                if (kk % kblk_d == 0) {
                    w = wblock(e * nblk + kk / kblk_d);
                }
                matmul_block(h_all_cb, in1_cb, h0 + kk * mtg, w + (kk % kblk_d) * pcd, 0, false, pcd, 1, hk);
#ifdef SE_EARLY_POP
                // The expert's last row: each weight block goes as soon as it is used, so the next expert's blocks
                // stream into the ring while this row still runs.
                if (last_sub && r == mt - 1 && kk % kblk_d == kblk_d - 1) {
                    cb_pop_front(in1_cb, slot);
                    ++popped;
                }
#endif
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < pcd; ++i) {
                pack_tile<true>(i, out_cb, r * pcd + i);  // row-major [MT x PCD]
            }
            tile_regs_release();
        }
        cb_pop_front(h_all_cb, h_all_tiles);
        cb_push_back(out_cb, mt * pcd);
#ifndef SE_EARLY_POP
        if (last_sub) {  // the expert's last sub-block: its weights can go
            cb_pop_front(in1_cb, nblk * slot);
            popped += nblk;
        }
#endif
#ifdef SE_DYN
        if (++s == subs_e) {
            s = 0;
            ++e;
            subs_e = e < n_act ? read_tile_value(tt::CBIndex::c_6, 0, 2 + e) : 0;
        }
#endif
    }
#ifdef SE_DYN
    cb_pop_front(tt::CBIndex::c_6, 1);
#endif
}
