// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hub of the Hy4 decode MoE program with the router on the chip (BRISC, NOC1) (fused/moe/program.py).
//
// 1. x (the 8 token rows as [ROWS, 32] BF16 row tiles, in this core's L1) goes to every worker core by multicast;
//    a multicast increment of SEM_X starts the router cores.
// 2. The router cores write sigmoid scores (FP32 [ROWS, 32] tiles, tile c = experts 32 c ..) into this core's score
//    buffer and count themselves in SEM_SCORES (bounded wait).
// 3. Per row: top-K of (score + bias) over the E experts in Q30 fixed point (the bias as Q30 integers from the host;
//    an FP32 score converts exactly), ties to the lower expert id, always exactly K picks; weights
//    score / sum(scores of the picks) * scale, rounded to BF16. With ROUTER_CANDIDATES every router core has already
//    cut its 32 experts to the K best of each row (router_writer.cpp), and the hub merges 8 lists of K per row.
// 4. The union of the rows' picks (ascending expert id), restricted to the chip's own experts
//    [expert_base, expert_base + expert_count) and numbered from expert_base; per-row weights of every union slot
//    (0 where the row did not pick it); then n_plain always-on plain slots (the shared expert) with weight 1.
// 5. The meta page goes to every core by multicast, then SEM_GO releases the expert cores.
// The per-row picks (ids, BF16 weights) are left in the picks buffer for the host: [ROWS][K ids | K weights] uint16.
//
// runtime args: [x_addr, status_addr, scores_addr, bias_addr, picks_addr, candidates_addr (ROUTER_CANDIDATES)]
// compile-time args: [cb_x, cb_meta, x_bytes, mc_x0, mc_y0, mc_x1, mc_y1, mc_dests, router_cores, E, K, scale_q16,
//                     expert_base, expert_count, n_plain, plain_id0]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "moe_dm.hpp"

namespace {
// value v * 2^-frac -> BF16 bits, round to nearest
inline uint16_t to_bf16(uint64_t v, int32_t frac) {
    if (v == 0) {
        return 0;
    }
    int32_t msb = 63 - __builtin_clzll(v);
    const int32_t shift = msb - 7;
    uint32_t m =
        shift > 0 ? static_cast<uint32_t>((v + (1ull << (shift - 1))) >> shift) : static_cast<uint32_t>(v << -shift);
    if (m >= 0x100) {
        m >>= 1;
        msb += 1;
    }
    return static_cast<uint16_t>(((127 + msb - frac) << 7) | (m & 0x7F));
}
}  // namespace

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t status_addr = get_arg_val<uint32_t>(1);
    const uint32_t scores_addr = get_arg_val<uint32_t>(2);
    const uint32_t bias_addr = get_arg_val<uint32_t>(3);
    const uint32_t picks_addr = get_arg_val<uint32_t>(4);
#ifdef ROUTER_CANDIDATES
    const uint32_t cand_addr = get_arg_val<uint32_t>(5);
#endif

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t x_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t mc_x0 = get_compile_time_arg_val(3);
    constexpr uint32_t mc_y0 = get_compile_time_arg_val(4);
    constexpr uint32_t mc_x1 = get_compile_time_arg_val(5);
    constexpr uint32_t mc_y1 = get_compile_time_arg_val(6);
    constexpr uint32_t mc_dests = get_compile_time_arg_val(7);
    constexpr uint32_t router_cores = get_compile_time_arg_val(8);
    constexpr uint32_t E = get_compile_time_arg_val(9);
    constexpr uint32_t K = get_compile_time_arg_val(10);
    constexpr uint32_t scale_q16 = get_compile_time_arg_val(11);
    constexpr uint32_t expert_base = get_compile_time_arg_val(12);
    constexpr uint32_t expert_count = get_compile_time_arg_val(13);
    constexpr uint32_t n_plain = get_compile_time_arg_val(14);
    constexpr uint32_t plain_id0 = get_compile_time_arg_val(15);
    static_assert(E % 32 == 0 && E <= 256 && K <= 8, "router size");

    const uint32_t x_l1 = get_write_ptr(cb_x);
    const uint32_t meta_l1 = get_write_ptr(cb_meta);

    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_addr);
    const uint32_t runs = status[hy4moe::ST_RUNS];
    for (uint32_t i = 0; i < hy4moe::ST_WORDS; ++i) {
        status[i] = 0;
    }
    status[hy4moe::ST_RUNS] = runs + 1;
#ifdef HY4_TIMING
    const uint32_t t0 = hy4moe::now();
#endif

    // 1. x to every core; the router cores start
    // Every NOC wait is bounded: a hub that cannot send says so in the status words and ends; the other cores then
    // give up on their own bounded waits.
    uint32_t failed = 0;
    if (!hy4moe::grid_mcast_write(x_addr, mc_x0, mc_y0, mc_x1, mc_y1, x_l1, x_bytes, mc_dests)) {
        failed = 1;
    } else if (!hy4moe::writes_flushed()) {
        failed = 4;
    } else if (!hy4moe::grid_mcast_inc(mc_x0, mc_y0, mc_x1, mc_y1, get_semaphore(hy4moe::SEM_X), 1, mc_dests)) {
        failed = 3;
    }
    if (failed) {
        status[hy4moe::ST_HUB] = failed;
        return;
    }

#ifdef HY4_TIMING
    status[hy4moe::ST_HUB_TIMING] = hy4moe::now() - t0;
#endif
    // 2. the scores
    if (!hy4moe::wait_ge(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hy4moe::SEM_SCORES)), router_cores)) {
        status[hy4moe::ST_ROUTER] = 1;
    }
#ifdef HY4_TIMING
    status[hy4moe::ST_HUB_TIMING + 1] = hy4moe::now() - t0;
#endif
    invalidate_l1_cache();
    [[maybe_unused]] const uint32_t* scores = reinterpret_cast<const uint32_t*>(scores_addr);
    [[maybe_unused]] const int32_t* bias = reinterpret_cast<const int32_t*>(bias_addr);
    volatile tt_l1_ptr uint16_t* picks = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(picks_addr);

    // 3. per row: top-K by score + bias (Q30), weights
    uint32_t mask[E / 32];
    for (uint32_t i = 0; i < E / 32; ++i) {
        mask[i] = 0;
    }
    uint16_t row_id[ROWS][K];
    uint16_t row_w[ROWS][K];
    for (uint32_t r = 0; r < ROWS; ++r) {
        hy4moe::Candidate best[K];
        for (uint32_t j = 0; j < K; ++j) {
            best[j].key = INT32_MIN;
            best[j].score = 0;
            best[j].id = 0;
        }
#ifdef ROUTER_CANDIDATES
        // every router core left its column's K best of this row (best first, ties to the lower id): merge them
        for (uint32_t c = 0; c < E / 32; ++c) {
            const hy4moe::Candidate* cand = reinterpret_cast<const hy4moe::Candidate*>(cand_addr) + (c * ROWS + r) * K;
            for (uint32_t j = 0; j < K; ++j) {
                hy4moe::insert_candidate<K>(best, cand[j].key, cand[j].score, cand[j].id);
            }
        }
#else
        for (uint32_t c = 0; c < E / 32; ++c) {
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t* w = scores + c * (hy4moe::ROW_TILE_BYTES_F32 / 4) + (face * ROWS + r) * 16;
                for (uint32_t j = 0; j < 16; ++j) {
                    const uint32_t e = c * 32 + face * 16 + j;
                    const int32_t s = hy4moe::q30(w[j]);
                    hy4moe::insert_candidate<K>(best, s + bias[e], s, e);
                }
            }
        }
#endif
        uint32_t id[K];
        int32_t sq[K];
        for (uint32_t j = 0; j < K; ++j) {
            id[j] = best[j].id;
            sq[j] = best[j].score;
        }
        // weights: s / sum(s) * scale. Q24 scores, the quotient by long division in 32 bits (Q23), BF16 out
        uint32_t sum = 0;
        uint32_t v[K];
        for (uint32_t j = 0; j < K; ++j) {
            v[j] = static_cast<uint32_t>(sq[j]) >> 6;
            if (v[j] > 0xFFFFFF) {
                v[j] = 0xFFFFFF;
            }
            sum += v[j];
        }
        for (uint32_t j = 0; j < K; ++j) {
            uint32_t q = 0;
            if (sum > 0) {
                uint32_t n = v[j] << 8;
                q = n / sum;
                uint32_t rem = n - q * sum;
                for (uint32_t step = 0; step < 3; ++step) {
                    rem <<= 5;
                    const uint32_t d = rem / sum;
                    q = (q << 5) | d;
                    rem -= d * sum;
                }
            }
            row_id[r][j] = static_cast<uint16_t>(id[j]);
            row_w[r][j] = to_bf16(static_cast<uint64_t>(q) * scale_q16, 39);
            mask[id[j] >> 5] |= 1u << (id[j] & 31);
            picks[r * 2 * K + j] = row_id[r][j];
            picks[r * 2 * K + K + j] = row_w[r][j];
        }
    }

    // 4. the union (this chip's experts), per-row weights by slot, the plain slots
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1);
    for (uint32_t i = 0; i < hy4moe::META_BYTES / 4; ++i) {
        meta[i] = 0;
    }
    volatile tt_l1_ptr uint16_t* meta16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1);
    volatile tt_l1_ptr uint16_t* w16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1 + hy4moe::META_W);
    uint8_t slot_of[E];
    uint32_t U = 0;
    for (uint32_t e = expert_base; e < expert_base + expert_count && e < E; ++e) {
        if ((mask[e >> 5] >> (e & 31)) & 1) {
            slot_of[e] = static_cast<uint8_t>(U);
            meta16[1 + U] = static_cast<uint16_t>(e - expert_base);
            ++U;
        }
    }
    for (uint32_t r = 0; r < ROWS; ++r) {
        for (uint32_t j = 0; j < K; ++j) {
            const uint32_t e = row_id[r][j];
            if (e >= expert_base && e < expert_base + expert_count) {
                w16[r * hy4moe::MAX_SLOTS + slot_of[e]] = row_w[r][j];
            }
        }
    }
    for (uint32_t i = 0; i < n_plain && U < hy4moe::MAX_SLOTS; ++i) {
        meta16[1 + U] = static_cast<uint16_t>(plain_id0 + i);
        for (uint32_t r = 0; r < ROWS; ++r) {
            w16[r * hy4moe::MAX_SLOTS + U] = 0x3F80;  // BF16 1.0
        }
        ++U;
    }
    meta16[0] = static_cast<uint16_t>(U);
    meta16[hy4moe::META_N_PLAIN / 2] = static_cast<uint16_t>(n_plain);

    // 5. the meta page to every core, then the release
    if (!hy4moe::grid_mcast_write(meta_l1, mc_x0, mc_y0, mc_x1, mc_y1, meta_l1, hy4moe::META_BYTES, mc_dests)) {
        failed = 2;
    } else if (!hy4moe::writes_flushed()) {
        failed = 4;
    } else if (!hy4moe::grid_mcast_inc(mc_x0, mc_y0, mc_x1, mc_y1, get_semaphore(hy4moe::SEM_GO), 1, mc_dests)) {
        failed = 3;
    } else if (!hy4moe::atomics_flushed()) {
        failed = 4;
    }
    status[hy4moe::ST_HUB] = failed;
#ifdef HY4_TIMING
    status[hy4moe::ST_HUB_TIMING + 2] = hy4moe::now() - t0;
#endif
}
