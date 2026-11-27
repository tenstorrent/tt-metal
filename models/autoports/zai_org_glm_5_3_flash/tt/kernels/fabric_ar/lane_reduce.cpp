// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 14 fabric all-reduce, lane core TRISC = the reduction (tt/fabric_ar.py). Fixed order on every chip, so all
// replicas come out bitwise identical: round 1 (axis 0) r1 = slot0 + slot1 (+ slot2 + slot3 ...) with fp32 DST
// accumulation, packed to bf16 (sent to the axis-1 peers and kept as the round-2 own slot); round 2 (axis 1)
// out = r1(q=0) + r1(q=1) + ... in the same way -> bf16. Pairs are added with add_tiles; the second and later pairs
// accumulate into DST (acc_to_dest): DST = ((s0 + s1) + (s2 + s3)) + ...
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"

constexpr uint32_t PT = get_compile_time_arg_val(0);
constexpr uint32_t N1 = get_compile_time_arg_val(1);
constexpr uint32_t N2 = get_compile_time_arg_val(2);
constexpr uint32_t CB_L1 = get_compile_time_arg_val(3);
constexpr uint32_t CB_L2 = get_compile_time_arg_val(4);
constexpr uint32_t CB_OWN2 = get_compile_time_arg_val(5);  // round-1 result kept for round 2
constexpr uint32_t CB_SEND = get_compile_time_arg_val(6);  // round-1 result handed to BRISC
constexpr uint32_t CB_RES = get_compile_time_arg_val(7);
constexpr uint32_t CB_OWN = get_compile_time_arg_val(8);  // the producers' packed partial
constexpr uint32_t ALIAS = get_compile_time_arg_val(9);   // 1: CB_OWN2 shares CB_SEND's L1 (pointers only), 0: packed
constexpr uint32_t BATCH = 4;                             // fp32 DST tiles (half sync)

// out_a <- sum over q of cb(q) for the PT tiles; out_b != out_a: a second CB over the same L1 (pointers only)
template <uint32_t N>
inline void reduce_round(const uint32_t (&cbs)[N], uint32_t out_a, uint32_t out_b) {
    for (uint32_t t0 = 0; t0 < PT; t0 += BATCH) {
        const uint32_t n = PT - t0 < BATCH ? PT - t0 : BATCH;
        for (uint32_t q = 0; q < N; q++) {
            cb_wait_front(cbs[q], t0 + n);
        }
        tile_regs_acquire();
        for (uint32_t q = 0; q < N; q += 2) {
            add_init(cbs[q], cbs[q + 1], q > 0);
            for (uint32_t j = 0; j < n; j++) {
                add_tiles(cbs[q], cbs[q + 1], t0 + j, t0 + j, j);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(out_a, n);
        for (uint32_t j = 0; j < n; j++) {
            pack_tile(j, out_a);
        }
        cb_push_back(out_a, n);
        if (out_b != out_a) {
            cb_reserve_back(out_b, n);
            if constexpr (!ALIAS) {
                for (uint32_t j = 0; j < n; j++) {
                    pack_tile(j, out_b);
                }
            }
            // ALIAS: out_b (CB_OWN2) is backed by the same L1 as out_a (CB_SEND: the chip's own-index round-2 slot,
            // fabric_ar.py tensor mode), so the packed tiles are already there: only its pointers advance
            cb_push_back(out_b, n);
        }
        tile_regs_release();
    }
    for (uint32_t q = 0; q < N; q++) {
        cb_pop_front(cbs[q], PT);
    }
}

void kernel_main() {
    const uint32_t p1 = get_arg_val<uint32_t>(0);
    const uint32_t p2 = get_arg_val<uint32_t>(1);
    const uint32_t iters = get_arg_val<uint32_t>(2);
    static_assert(N1 == 1 || N1 % 2 == 0, "round 1 adds slot pairs");
    static_assert(N2 == 1 || N2 % 2 == 0, "round 2 adds slot pairs");
    compute_kernel_hw_startup(CB_OWN, CB_OWN, CB_RES);
    for (uint32_t it = 0; it < iters; it++) {
        if constexpr (N1 > 1) {
            uint32_t cbs[N1];
            for (uint32_t q = 0; q < N1; q++) {
                cbs[q] = q == p1 ? CB_OWN : CB_L1 + q;
            }
            if constexpr (N2 > 1) {
                reduce_round<N1>(cbs, CB_SEND, CB_OWN2);
            } else {
                reduce_round<N1>(cbs, CB_RES, CB_RES);
            }
        }
        if constexpr (N2 > 1) {
            uint32_t cbs[N2];
            for (uint32_t q = 0; q < N2; q++) {
                cbs[q] = q == p2 ? (N1 > 1 ? CB_OWN2 : CB_OWN) : CB_L2 + q;
            }
            reduce_round<N2>(cbs, CB_RES, CB_RES);
        }
    }
}
