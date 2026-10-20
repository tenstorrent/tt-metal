// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grouped MoE routing + plan in one program (dflash/grouped_moe.py GroupedMoE.route): from the router logits [ROWS, E]
// (BF16 tiles, the router matmul's output) to the dense routing weights W [ROWS, E] (the down bias's) and the plan's
// gather / combine matrices and slot table (as gmoe_plan.cpp). Cores: a RW x RH rectangle, core s = y RW + x.
//   phase 1 (cores s < ROWS / 4, 4 rows each): each row's routed experts exactly as TopKRouter.dense_decode selects
//     them (every logit >= the 4th largest, so a tie at the 4th routes more; at most K_IDS kept: logit order, then
//     index), their softmax weights in integer fixed point (the RISC-V cores have no FPU: exp(d) = 2^(d log2 e), a
//     Q30 polynomial; the division 64-bit), rounded to BF16; the row of W written, the row's (ids, weights) sent to the
//     hub (core 0), which multicasts all rows' to every core once all have arrived (semaphores ARRIVE / GO; one core
//     making every slot's entries serially was slower: 96 vs 66 us);
//   phase 2 (every core = slot s): the plan from the rows' expert masks (counts, slots), slot s's stripes of P1T
//     (one-hot rows) and Pw (the routing weights), core 0 the slot table.
//
// common runtime args: [logits, w, p1t, pw, table, hub_x, hub_y, mc_x0, mc_y0, mc_x1, mc_y1]
// compile-time args: [cb_scr, RW, MAX_SLOTS, NCORES, SEM_ARRIVE, SEM_GO, accessors (logits, w, p1t, pw, table)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "gmoe_common.hpp"

namespace {
constexpr uint32_t K = gmoe::K_IDS;  // experts kept per row
constexpr uint32_t ROW_BYTES = 32;   // per row: K id bytes (0xFF: none), pad to 8, K BF16 weights, the expert mask (28)

// byte offset of element (row, col) in a 32 x 32 tile of 2-byte values (faces of 16 x 16)
inline uint32_t off(uint32_t row, uint32_t col) {
    return ((row / 16) * 2 + col / 16) * 512 + ((row % 16) * 16 + col % 16) * 2;
}
// BF16 bits -> an unsigned key in the same order as the values
inline uint32_t key16(uint32_t b) { return (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000); }
// BF16 bits -> signed Q16 fixed point (|value| < 2^15)
inline int32_t q16(uint32_t b) {
    const int32_t e = static_cast<int32_t>((b >> 7) & 0xFF);
    if (e == 0) {
        return 0;
    }
    const int32_t m = static_cast<int32_t>(0x80 | (b & 0x7F));
    const int32_t sh = e - 118;  // value = m 2^(e - 127 - 7); Q16: << 16
    int32_t v = sh >= 0 ? (sh < 23 ? (m << sh) : 0x7FFFFFFF) : (sh > -31 ? (m >> -sh) : 0);
    return (b & 0x8000) ? -v : v;
}
// 2^(t / 2^16) for t <= 0 as Q30 (0 below 2^-30)
inline uint32_t exp2_q30(int64_t t) {
    const int32_t n = static_cast<int32_t>(t >> 16);  // floor
    if (n < -30) {
        return 0;
    }
    const uint32_t f = static_cast<uint32_t>(t - (static_cast<int64_t>(n) << 16));  // [0, 2^16)
    // 2^x on [0, 1): Taylor to degree 7 in Horner form, Q30 (max relative error ~1e-6)
    const int64_t x = static_cast<int64_t>(f) << 14;  // Q30
    const int64_t c[7] = {744261118, 257941248, 59597083, 10327387, 1431680, 165394, 16377};
    int64_t p = c[6];
    for (int32_t k = 5; k >= 0; --k) {
        p = c[k] + ((p * x) >> 30);
    }
    p = (1LL << 30) + ((p * x) >> 30);  // [2^30, 2^31)
    return static_cast<uint32_t>(p >> (-n));
}
// q / 2^30 (0 < q <= 2^30) -> BF16 bits, round to nearest even
inline uint16_t bf16_q30(uint32_t q) {
    if (q == 0) {
        return 0;
    }
    const int32_t msb = 31 - __builtin_clz(q);
    int32_t e = msb - 30 + 127;
    uint32_t m;
    const int32_t sh = msb - 7;
    if (sh > 0) {
        m = q >> sh;
        const uint32_t rem = q & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (m & 1))) {
            ++m;
        }
        if (m == 256) {
            m = 128;
            ++e;
        }
    } else {
        m = q << (-sh);
    }
    return static_cast<uint16_t>((e << 7) | (m & 0x7F));
}
}  // namespace

void kernel_main() {
    const uint32_t logits_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t p1t_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t pw_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t table_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t hub_x = get_common_arg_val<uint32_t>(5), hub_y = get_common_arg_val<uint32_t>(6);
    const uint32_t mx0 = get_common_arg_val<uint32_t>(7), my0 = get_common_arg_val<uint32_t>(8);
    const uint32_t mx1 = get_common_arg_val<uint32_t>(9), my1 = get_common_arg_val<uint32_t>(10);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t RW = get_compile_time_arg_val(1);
    constexpr uint32_t MAX_SLOTS = get_compile_time_arg_val(2);
    constexpr uint32_t NCORES = get_compile_time_arg_val(3);
    constexpr uint32_t SEM_ARRIVE = get_compile_time_arg_val(4);
    constexpr uint32_t SEM_GO = get_compile_time_arg_val(5);
    constexpr auto l_args = TensorAccessorArgs<6>();
    constexpr auto w_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr auto p1t_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto pw_args = TensorAccessorArgs<p1t_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<pw_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile = 2048;
    constexpr uint32_t E = gmoe::E;
    constexpr uint32_t ROWS = gmoe::ROWS;
    constexpr uint32_t RT = ROWS / 32;
    constexpr uint32_t RPC = 4;  // rows per phase-1 core
    constexpr uint32_t N1 = ROWS / RPC;
    static_assert(N1 <= NCORES && E == 32);

    const uint32_t s = static_cast<uint32_t>(get_absolute_logical_y()) * RW + get_absolute_logical_x();
    const auto s_l = TensorAccessor(l_args, logits_addr, tile);
    const auto s_w = TensorAccessor(w_args, w_addr, tile);
    const auto s_p1t = TensorAccessor(p1t_args, p1t_addr, tile);
    const auto s_pw = TensorAccessor(pw_args, pw_addr, tile);
    const auto s_table = TensorAccessor(table_args, table_addr, 8 * E);

    cb_reserve_back(cb_scr, 1);
    const uint32_t rows_l1 = get_write_ptr(cb_scr);  // every row's (ids, weights): ROWS x ROW_BYTES (hub + multicast)
    const uint32_t mine_l1 = rows_l1 + ROWS * ROW_BYTES;  // this core's RPC rows before sending
    const uint32_t lg_l1 = mine_l1 + RPC * ROW_BYTES;     // the logits tile of this core's rows
    const uint32_t wrow_l1 = lg_l1 + tile;                // this core's W rows: RPC x 64 bytes (two face rows each)
    const uint32_t p1t_l1 = wrow_l1 + RPC * 64;           // stripe: RT tiles [32 slot rows, 32 rows]
    const uint32_t pw_l1 = p1t_l1 + RT * tile;            // stripe: RT tiles [32 rows, 32 slot rows]
    const uint32_t table_l1 = pw_l1 + RT * tile;
    auto u16 = [](uint32_t a) -> volatile tt_l1_ptr uint16_t* {
        return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(a);
    };
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t z = 0; z < 2 * RT * tile; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, p1t_l1 + z, MEM_ZEROS_SIZE);
    }

    // ---- phase 1: this core's RPC rows
    if (s < N1) {
        const uint32_t r0 = s * RPC, t = r0 / 32;
        noc_async_read(s_l.get_noc_addr(t), lg_l1, tile);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t i = 0; i < RPC; ++i) {
            const uint32_t rr = r0 % 32 + i;
            uint32_t b[E], key[E];
            for (uint32_t e = 0; e < E; ++e) {
                b[e] = *u16(lg_l1 + off(rr, e));
                key[e] = key16(b[e]);
            }
            // the 4th largest logit (with multiplicity)
            uint32_t top[gmoe::K_SEL];
            for (uint32_t k = 0; k < gmoe::K_SEL; ++k) {
                top[k] = 0;
            }
            for (uint32_t e = 0; e < E; ++e) {
                uint32_t v = key[e];
                for (uint32_t k = 0; k < gmoe::K_SEL; ++k) {
                    if (v > top[k]) {
                        const uint32_t tmp = top[k];
                        top[k] = v;
                        v = tmp;
                    }
                }
            }
            const uint32_t kth = top[gmoe::K_SEL - 1], kmax = top[0];
            // the selected (logit >= kth), at most K: descending logit, then ascending index
            uint32_t sel[K], n = 0;
            uint32_t taken = 0;  // bit e: picked
            for (; n < K; ++n) {
                uint32_t best = E;
                for (uint32_t e = 0; e < E; ++e) {
                    if (key[e] >= kth && !((taken >> e) & 1) && (best == E || key[e] > key[best])) {
                        best = e;
                    }
                }
                if (best == E) {
                    break;
                }
                sel[n] = best;
                taken |= 1u << best;
            }
            // softmax over the selected, fixed point: e_k = 2^((l_k - max) log2 e) (Q30), w_k = e_k / sum
            uint32_t maxb = 0;
            for (uint32_t e = 0; e < E; ++e) {
                if (key[e] == kmax) {
                    maxb = b[e];
                }
            }
            const int32_t qmax = q16(maxb);
            uint32_t ex[K];
            uint64_t sum = 0;
            for (uint32_t k = 0; k < n; ++k) {
                const int64_t d = static_cast<int64_t>(q16(b[sel[k]])) - qmax;  // <= 0, Q16
                ex[k] = exp2_q30((d * 94548) >> 16);                            // log2 e = 94548 / 2^16
                sum += ex[k];
            }
            uint8_t* ro = reinterpret_cast<uint8_t*>(mine_l1 + i * ROW_BYTES);
            volatile tt_l1_ptr uint16_t* rw =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mine_l1 + i * ROW_BYTES + 8);
            volatile tt_l1_ptr uint32_t* wr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wrow_l1 + i * 64);
            for (uint32_t q = 0; q < 16; ++q) {
                wr[q] = 0;
            }
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mine_l1 + i * ROW_BYTES + 28) = taken;
            for (uint32_t k = 0; k < K; ++k) {
                ro[k] = k < n ? static_cast<uint8_t>(sel[k]) : 0xFF;
                uint16_t wb = 0;
                if (k < n) {
                    wb = bf16_q30(static_cast<uint32_t>((static_cast<uint64_t>(ex[k]) << 30) / sum));
                    const uint32_t e = sel[k];
                    *u16(wrow_l1 + i * 64 + (e / 16) * 32 + (e % 16) * 2) = wb;
                }
                rw[k] = wb;
            }
            // W row rr of tile t: its two face rows
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_write(
                    wrow_l1 + i * 64 + f * 32, s_w.get_noc_addr(t) + ((rr / 16) * 2 + f) * 512 + (rr % 16) * 32, 32);
            }
        }
        // the rows to the hub, then its arrival count
        noc_async_write(mine_l1, get_noc_addr(hub_x, hub_y, rows_l1 + r0 * ROW_BYTES), RPC * ROW_BYTES);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(hub_x, hub_y, get_semaphore(SEM_ARRIVE)), 1);
        noc_async_atomic_barrier();
    }
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GO));
    if (s == 0) {  // the hub: every row to every core, then GO
        volatile tt_l1_ptr uint32_t* arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ARRIVE));
        noc_semaphore_wait(arrive, N1);
        noc_async_write_multicast_loopback_src(
            rows_l1, get_noc_multicast_addr(mx0, my0, mx1, my1, rows_l1), ROWS * ROW_BYTES, NCORES);
        noc_async_write_barrier();
        *go = 1;
        noc_semaphore_set_multicast_loopback_src(
            get_semaphore(SEM_GO), get_noc_multicast_addr(mx0, my0, mx1, my1, get_semaphore(SEM_GO)), NCORES);
        noc_async_write_barrier();
    }
    noc_semaphore_wait(go, 1);
    noc_async_read_barrier();  // (the stripes' zero fill)
    invalidate_l1_cache();

    // ---- phase 2: the plan from the rows' expert masks, slot s's stripes
    uint32_t mask[ROWS];
    for (uint32_t r = 0; r < ROWS; ++r) {
        mask[r] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1 + r * ROW_BYTES + 28);
    }
    uint32_t cnt[E];
    for (uint32_t e = 0; e < E; ++e) {
        cnt[e] = 0;
    }
    for (uint32_t r = 0; r < ROWS; ++r) {
        for (uint32_t m = mask[r]; m; m &= m - 1) {
            cnt[__builtin_ctz(m)]++;
        }
    }
    uint32_t es = E, lo = 0, base = 0;
    volatile tt_l1_ptr uint32_t* tb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_l1);
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t tiles = (cnt[e] + 31) / 32;
        if (tiles && base <= s && s < base + tiles) {
            es = e;
            lo = 32 * (s - base);
        }
        tb[e] = tiles;
        tb[E + e] = base;
        base += tiles;
    }
    if (s == 0) {
        noc_async_write(table_l1, s_table.get_noc_addr(0), 8 * E);
    }
    if (es < E) {
        const uint32_t hi = lo + 32;
        uint32_t rank = 0;
        for (uint32_t r = 0; r < ROWS && rank < hi; ++r) {
            if (!((mask[r] >> es) & 1)) {
                continue;
            }
            if (rank >= lo) {
                const uint32_t i = rank - lo;
                const uint8_t* id = reinterpret_cast<const uint8_t*>(rows_l1 + r * ROW_BYTES);
                uint32_t k = 0;
                while (id[k] != es) {
                    ++k;
                }
                *u16(p1t_l1 + (r / 32) * tile + off(i, r % 32)) = 0x3F80;  // 1.0
                *u16(pw_l1 + (r / 32) * tile + off(r % 32, i)) = *u16(rows_l1 + r * ROW_BYTES + 8 + 2 * k);
            }
            ++rank;
        }
    }
    for (uint32_t t = 0; t < RT; ++t) {
        noc_async_write(p1t_l1 + t * tile, s_p1t.get_noc_addr(s * RT + t), tile);
        noc_async_write(pw_l1 + t * tile, s_pw.get_noc_addr(t * MAX_SLOTS + s), tile);
    }
    noc_async_write_barrier();
}
