// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grouped MoE routing + plan in one program (dflash/grouped_moe.py GroupedMoE.route): from the router logits [ROWS, E]
// (BF16 tiles) to the dense routing weights W [ROWS, E] and the plan's gather / combine matrices and slot table. Cores:
// a RW x RH rectangle, core s = y RW + x.
//   phase 1 (cores s < ROWS / 4, 4 rows each): each row's routed experts exactly as TopKRouter.dense_decode selects
//     them (every logit >= the 4th largest, so a tie at the 4th routes more; at most K_IDS kept: logit order, then
//     index), their softmax weights in integer fixed point (the RISC-V cores have no FPU: exp(d) = 2^(d log2 e), a
//     Q30 polynomial; one 64-bit reciprocal per row), rounded to BF16; the row of W written (into Pw's bias slot), each
//     routed expert counted at the hub (atomic increments of its count semaphores, 3 x 10 bits per word), the row's
//     (ids, weights) sent to the hub (core 0), which multicasts all rows and the counts to every core once all have
//     arrived (semaphores ARRIVE / GO; one core making every slot's entries serially was slower: 96 vs 66 us);
//   phase 2 (every core = slot s): the plan from the counts (counting the 256 rows' masks on every core took ~10 us:
//     RISC-V L1 loads, a software ctz), slot s's stripes of P1T (one-hot rows) and Pw (the routing weights), core 0
//     the slot table.
// The down bias through the combine: the last slot (never a routed one) is the bias slot; phase 1 writes the rows of W
//   into Pw's columns of that slot (its core leaves them), and the cores without phase-1 rows copy the layer's down
//   bias rows into that slot of the shared Y_g, so the combine (Pw Y_g) adds each row's routing-weighted down bias.
//
// Larger passes (512 / 1024 rows; GMOE_RPC rows per phase-1 core, more slots than cores): phase 2 loops over the
//   slots s = core + j NCORES; GMOE_CNT16 (1024 rows: an expert's count may reach 1024, past 10 bits): each phase-1
//   core sends its own 16-bit expert counts with its rows and the hub sums them (instead of the count semaphores).
//
// common runtime args: [logits, p1t, pw, table, hub_x, hub_y, mc_x0, mc_y0, mc_x1, mc_y1, down bias rows, Y_g]
// compile-time args: [cb_scr, RW, MAX_SLOTS, NCORES, SEM_ARRIVE, SEM_GO, accessors (logits, p1t, pw, table, down bias
// rows, Y_g)]; GMOE_KTH: hidden tiles

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "gmoe_common.hpp"
#include "route_row.hpp"

namespace {
constexpr uint32_t K = gmoe::K_IDS;  // experts kept per row
constexpr uint32_t ROW_BYTES = 32;   // per row: K id bytes (0xFF: none), pad to 8, K BF16 weights, the expert mask (28)

// byte offset of element (row, col) in a 32 x 32 tile of 2-byte values (faces of 16 x 16)
inline uint32_t off(uint32_t row, uint32_t col) {
    return ((row / 16) * 2 + col / 16) * 512 + ((row % 16) * 16 + col % 16) * 2;
}
}  // namespace

void kernel_main() {
    const uint32_t logits_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p1t_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t pw_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t table_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t hub_x = get_common_arg_val<uint32_t>(4), hub_y = get_common_arg_val<uint32_t>(5);
    const uint32_t mx0 = get_common_arg_val<uint32_t>(6), my0 = get_common_arg_val<uint32_t>(7);
    const uint32_t mx1 = get_common_arg_val<uint32_t>(8), my1 = get_common_arg_val<uint32_t>(9);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t RW = get_compile_time_arg_val(1);
    constexpr uint32_t MAX_SLOTS = get_compile_time_arg_val(2);
    constexpr uint32_t NCORES = get_compile_time_arg_val(3);
    constexpr uint32_t SEM_ARRIVE = get_compile_time_arg_val(4);
    constexpr uint32_t SEM_GO = get_compile_time_arg_val(5);
    constexpr auto l_args = TensorAccessorArgs<6>();
    constexpr auto p1t_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr auto pw_args = TensorAccessorArgs<p1t_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<pw_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto yg_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr uint32_t KTH = GMOE_KTH;             // hidden tiles
    constexpr uint32_t BIAS_SLOT = MAX_SLOTS - 1;  // never a routed one: at most (ROWS K_IDS + 31 E) / 32 = 71 are
    constexpr uint32_t tile = 2048;
    constexpr uint32_t E = gmoe::E;
    constexpr uint32_t ROWS = gmoe::ROWS;
    constexpr uint32_t RT = ROWS / 32;
#ifndef GMOE_RPC
#define GMOE_RPC 4
#endif
    constexpr uint32_t RPC = GMOE_RPC;  // rows per phase-1 core (within one 32-row tile)
    constexpr uint32_t N1 = ROWS / RPC;
    static_assert(N1 < NCORES && E == 32 && 32 % RPC == 0);
    constexpr uint32_t SEM_CNT0 = SEM_GO + 1, CNT_WORDS = (E + 2) / 3;  // count semaphores (10 bits per expert)
#ifndef GMOE_CNT16
    static_assert(ROWS < 1024);
#endif

    const uint32_t s = static_cast<uint32_t>(get_absolute_logical_y()) * RW + get_absolute_logical_x();
    const auto s_l = TensorAccessor(l_args, logits_addr, tile);
    const auto s_p1t = TensorAccessor(p1t_args, p1t_addr, tile);
    const auto s_pw = TensorAccessor(pw_args, pw_addr, tile);
    const auto s_table = TensorAccessor(table_args, table_addr, 8 * E);

    cb_reserve_back(cb_scr, 1);
    const uint32_t rows_l1 = get_write_ptr(cb_scr);  // every row's (ids, weights): ROWS x ROW_BYTES (hub + multicast)
    // then the expert counts (the hub's count semaphores, CNT_WORDS words of 3 x 10 bits; with the rows in the
    // multicast)
    const uint32_t cnts_l1 = rows_l1 + ROWS * ROW_BYTES;
    const uint32_t mine_l1 = cnts_l1 + 64;                // this core's RPC rows before sending
    const uint32_t lg_l1 = mine_l1 + RPC * ROW_BYTES;     // the logits tile of this core's rows
    const uint32_t wrow_l1 = lg_l1 + tile;                // this core's W rows: RPC x 64 bytes (two face rows each)
    const uint32_t p1t_l1 = wrow_l1 + RPC * 64;           // stripe: RT tiles [32 slot rows, 32 rows]
    const uint32_t pw_l1 = p1t_l1 + RT * tile;            // stripe: RT tiles [32 rows, 32 slot rows]
    const uint32_t table_l1 = pw_l1 + RT * tile;
    // GMOE_CNT16: the phase-1 cores' expert counts at the hub (N1 x 32 16-bit; after the down bias staging)
    const uint32_t parts_l1 = table_l1 + 8 * E + 12 * 1088;
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
#ifdef GMOE_CNT16
        uint32_t lcnt[E];
        for (uint32_t e = 0; e < E; ++e) {
            lcnt[e] = 0;
        }
#endif
        noc_async_read(s_l.get_noc_addr(t), lg_l1, tile);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t i = 0; i < RPC; ++i) {
            const uint32_t rr = r0 % 32 + i;
            uint32_t b[E];
            // the row's 32 logits: two 16-value face rows, read 2 values per 32-bit load
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint32_t* src =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lg_l1 + off(rr, 16 * f));
                for (uint32_t w = 0; w < 8; ++w) {
                    const uint32_t two = src[w];
                    b[16 * f + 2 * w] = two & 0xFFFF;
                    b[16 * f + 2 * w + 1] = two >> 16;
                }
            }
            uint32_t sel[K], n, taken;
            uint16_t wbk[K];
            route_row::route<E, K, gmoe::K_SEL>(b, sel, n, taken, wbk);
            uint8_t* ro = reinterpret_cast<uint8_t*>(mine_l1 + i * ROW_BYTES);
            volatile tt_l1_ptr uint16_t* rw =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mine_l1 + i * ROW_BYTES + 8);
            volatile tt_l1_ptr uint32_t* wr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wrow_l1 + i * 64);
            for (uint32_t q = 0; q < 16; ++q) {
                wr[q] = 0;
            }
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mine_l1 + i * ROW_BYTES + 28) = taken;
#ifdef GMOE_CNT16
            for (uint32_t k = 0; k < n; ++k) {
                ++lcnt[sel[k]];
            }
#else
            // the row's experts counted at the hub (atomic increments of its count semaphores: 10 bits per expert)
            for (uint32_t k = 0; k < n; ++k) {
                noc_semaphore_inc(
                    get_noc_addr(hub_x, hub_y, get_semaphore(SEM_CNT0 + sel[k] / 3)), 1u << (10 * (sel[k] % 3)));
            }
#endif
            for (uint32_t k = 0; k < K; ++k) {
                ro[k] = k < n ? static_cast<uint8_t>(sel[k]) : 0xFF;
                uint16_t wb = 0;
                if (k < n) {
                    wb = wbk[k];
                    const uint32_t e = sel[k];
                    *u16(wrow_l1 + i * 64 + (e / 16) * 32 + (e % 16) * 2) = wb;
                }
                rw[k] = wb;
            }
            // W row rr of tile t: its two face rows into Pw's columns of the bias slot
            const uint64_t w_dst = s_pw.get_noc_addr(t * MAX_SLOTS + BIAS_SLOT);
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_write(wrow_l1 + i * 64 + f * 32, w_dst + ((rr / 16) * 2 + f) * 512 + (rr % 16) * 32, 32);
            }
        }
#ifdef GMOE_CNT16
        // this core's counts to the hub (16 bits each; staged in this core's W-row area, now written out)
        noc_async_write_barrier();
        volatile tt_l1_ptr uint16_t* lc = u16(wrow_l1);
        for (uint32_t e = 0; e < E; ++e) {
            lc[e] = static_cast<uint16_t>(lcnt[e]);
        }
        noc_async_write(wrow_l1, get_noc_addr(hub_x, hub_y, parts_l1 + s * 64), 64);
#endif
        // the rows to the hub, then its arrival count
        noc_async_write(mine_l1, get_noc_addr(hub_x, hub_y, rows_l1 + r0 * ROW_BYTES), RPC * ROW_BYTES);
        noc_async_write_barrier();
        noc_async_atomic_barrier();  // (the count increments landed before the arrival)
        noc_semaphore_inc(get_noc_addr(hub_x, hub_y, get_semaphore(SEM_ARRIVE)), 1);
        noc_async_atomic_barrier();
    }
    // the cores without phase-1 rows: the layer's down bias rows [E, hidden] (BFP8 tiles) into the bias slot of the
    // shared Y_g (in L1: the combine reads it there)
    if (s >= N1) {
        constexpr uint32_t bt = 1088;
        const auto s_b = TensorAccessor(b_args, get_common_arg_val<uint32_t>(10), bt);
        const auto s_yg = TensorAccessor(yg_args, get_common_arg_val<uint32_t>(11), bt);
        const uint32_t bl1 = table_l1 + 8 * E;
        uint32_t n = 0;
        for (uint32_t c = s - N1; c < KTH; c += NCORES - N1) {
            noc_async_read(s_b.get_noc_addr(c), bl1 + n++ * bt, bt);
        }
        noc_async_read_barrier();
        n = 0;
        for (uint32_t c = s - N1; c < KTH; c += NCORES - N1) {
            noc_async_write(bl1 + n++ * bt, s_yg.get_noc_addr(BIAS_SLOT * KTH + c), bt);
        }
    }
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GO));
    if (s == 0) {  // the hub: every row to every core, then GO
        volatile tt_l1_ptr uint32_t* arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ARRIVE));
        noc_semaphore_wait(arrive, N1);
        invalidate_l1_cache();
#ifdef GMOE_CNT16
        {
            uint32_t tot[E];
            for (uint32_t e = 0; e < E; ++e) {
                tot[e] = 0;
            }
            for (uint32_t c = 0; c < N1; ++c) {
                volatile tt_l1_ptr uint32_t* pc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(parts_l1 + c * 64);
                for (uint32_t w = 0; w < E / 2; ++w) {
                    const uint32_t two = pc[w];
                    tot[2 * w] += two & 0xFFFF;
                    tot[2 * w + 1] += two >> 16;
                }
            }
            for (uint32_t e = 0; e < E; ++e) {
                u16(cnts_l1)[e] = static_cast<uint16_t>(tot[e]);
            }
        }
#else
        for (uint32_t w = 0; w < CNT_WORDS; ++w) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnts_l1)[w] =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_CNT0 + w));
        }
#endif
        noc_async_write_multicast_loopback_src(
            rows_l1, get_noc_multicast_addr(mx0, my0, mx1, my1, rows_l1), ROWS * ROW_BYTES + 64, NCORES);
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
    // the expert counts (the hub's, multicast with the rows; counting the 256 rows' masks on every core took ~10 us:
    // RISC-V L1 loads and a software ctz)
    uint32_t cnt[E];
    for (uint32_t e = 0; e < E; ++e) {
#ifdef GMOE_CNT16
        cnt[e] = u16(cnts_l1)[e];
#else
        cnt[e] = (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnts_l1)[e / 3] >> (10 * (e % 3))) & 0x3FF;
#endif
    }
    uint32_t tiles_e[E], base_e[E];
    {
        uint32_t base = 0;
        volatile tt_l1_ptr uint32_t* tb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_l1);
        for (uint32_t e = 0; e < E; ++e) {
            tiles_e[e] = (cnt[e] + 31) / 32;
            base_e[e] = base;
            tb[e] = tiles_e[e];
            tb[E + e] = base;
            base += tiles_e[e];
        }
    }
    if (s == 0) {
        noc_async_write(table_l1, s_table.get_noc_addr(0), 8 * E);
    }
    // this core's slots: s, s + NCORES, ... (one per core while MAX_SLOTS <= NCORES)
    for (uint32_t sl = s; sl < MAX_SLOTS; sl += NCORES) {
        if (sl != s) {  // (the stripes again zero: the previous slot's entries written out)
            noc_async_write_barrier();
            for (uint32_t z = 0; z < 2 * RT * tile; z += MEM_ZEROS_SIZE) {
                noc_async_read(zeros, p1t_l1 + z, MEM_ZEROS_SIZE);
            }
            noc_async_read_barrier();
        }
        uint32_t es = E, lo = 0;
        for (uint32_t e = 0; e < E; ++e) {
            if (tiles_e[e] && base_e[e] <= sl && sl < base_e[e] + tiles_e[e]) {
                es = e;
                lo = 32 * (sl - base_e[e]);
            }
        }
        if (es < E) {
            const uint32_t hi = lo + 32;
            uint32_t rank = 0;
            for (uint32_t r = 0; r < ROWS && rank < hi; ++r) {
                const uint32_t mask_r = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1 + r * ROW_BYTES + 28);
                if (!((mask_r >> es) & 1)) {
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
            noc_async_write(p1t_l1 + t * tile, s_p1t.get_noc_addr(sl * RT + t), tile);
            if (sl == BIAS_SLOT) {
                continue;  // (phase 1 wrote the W rows there)
            }
            noc_async_write(pw_l1 + t * tile, s_pw.get_noc_addr(t * MAX_SLOTS + sl), tile);
        }
    }
    noc_async_write_barrier();
}
