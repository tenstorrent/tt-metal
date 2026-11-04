// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused KDA "pre" — reader (BRISC, NoC0): every tensor read, and in ROWS mode the in-place conv-history shift.
//
// Compile-time args: ROWS, then TensorAccessorArgs of hist (state.C [1,3,B,24576] bf16), qkv ([1,1,B,24576] bf16),
// taps (ROWS: packed taps [4*64*32, 32] bf16, page 64 j + k; TILES: taps_hist [1,3,1,24576] fp32, page 768 j + c),
// tap_cur ([1,1,1,24576] fp32, TILES), lr ([1,1,B,320] bf16: f_a | g_a | b), w_fb, w_gb ([128, 8192] bf16), dt_bias,
// decay rate ([1,1,1,8192] fp32), act0 (active [1,1,B,1] fp32), act2 (lower_bound * active).
// Runtime args: the 11 addresses, core k, rows B, update_hist [, KDA_VERIFY: the selector page address].
// KDA_VERIFY (Stage 17 speculative verify, doc/ttnn/17-mtp-spec-decode/design.md §2.2; ROWS mode, B <= 4): the B rows
// are consecutive tokens of ONE user. Row j's conv window = (src(j-3), src(j-2), src(j-1), Q_j): src(i < 0) = history
// slot 3 + i of slot `sel` (word 0 of a [1, 32] uint32 DRAM page; that slot's 3 rows are read ONCE, before any
// write-back), src(i >= 0) = this launch's pre-conv row i. The usual write-back (H1_j, H2_j, Q_j) -> row j is then
// exactly the candidate history of row j. Compute / writer unchanged.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "kda_pre_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"

using namespace kdap;

namespace {

// row 0 of an fp32 [.., 1, W] tile page -> row 0 of the local tile at l1 (2 face-row reads of 64 B)
template <typename Acc>
FORCE_INLINE void read_row0_fp32(const Acc& acc, uint32_t page, uint32_t l1) {
    noc_async_read(acc.get_noc_addr(page, 0), l1, 64);
    noc_async_read(acc.get_noc_addr(page, 1024), l1 + 1024, 64);
}

// bf16 row pair (2u, 2u+1) of page -> rows (2r, 2r+1) of the local bf16 tile at l1: 2 x 64 B (one per face half;
// the pairs keep the 64-B DRAM alignment of source and destination equal)
template <typename Acc>
FORCE_INLINE void read_pair_bf16(const Acc& acc, uint32_t page, uint32_t u, uint32_t l1, uint32_t r) {
    noc_async_read(acc.get_noc_addr(page, tile_off(2 * u, 0) * 2), l1 + tile_off(2 * r, 0) * 2, 64);
    noc_async_read(acc.get_noc_addr(page, tile_off(2 * u, 16) * 2), l1 + tile_off(2 * r, 16) * 2, 64);
}

// rows (2r, 2r+1) of the local bf16 tile at l1 -> row pair (2u, 2u+1) of page
template <typename Acc>
FORCE_INLINE void write_pair_bf16(const Acc& acc, uint32_t page, uint32_t u, uint32_t l1, uint32_t r) {
    noc_async_write(l1 + tile_off(2 * r, 0) * 2, acc.get_noc_addr(page, tile_off(2 * u, 0) * 2), 64);
    noc_async_write(l1 + tile_off(2 * r, 16) * 2, acc.get_noc_addr(page, tile_off(2 * u, 16) * 2), 64);
}

// 64-B face-half row pair for a mixed pair: row p of the pair from `src_new` if active, else from `src_old`
FORCE_INLINE void assemble_pair(
    uint32_t dst, uint32_t src_new, uint32_t src_old, uint32_t r, uint32_t h, bool a0, bool a1) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    const uint32_t off = tile_off(2 * r, 16 * h) * 2;  // bytes; 2 rows x 32 B
    volatile tt_l1_ptr uint32_t* n = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_new + off);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_old + off);
    for (uint32_t i = 0; i < 8; ++i) {
        d[i] = a0 ? n[i] : o[i];
        d[8 + i] = a1 ? n[8 + i] : o[8 + i];
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t ROWS = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr auto hist_args = TensorAccessorArgs<PH_CT0 + 1>();
    constexpr auto qkv_args = TensorAccessorArgs<hist_args.next_compile_time_args_offset()>();
    constexpr auto taps_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto tc_args = TensorAccessorArgs<taps_args.next_compile_time_args_offset()>();
    constexpr auto lr_args = TensorAccessorArgs<tc_args.next_compile_time_args_offset()>();
    constexpr auto wfb_args = TensorAccessorArgs<lr_args.next_compile_time_args_offset()>();
    constexpr auto wgb_args = TensorAccessorArgs<wfb_args.next_compile_time_args_offset()>();
    constexpr auto dt_args = TensorAccessorArgs<wgb_args.next_compile_time_args_offset()>();
    constexpr auto rate_args = TensorAccessorArgs<dt_args.next_compile_time_args_offset()>();
    constexpr auto act0_args = TensorAccessorArgs<rate_args.next_compile_time_args_offset()>();
    constexpr auto act2_args = TensorAccessorArgs<act0_args.next_compile_time_args_offset()>();

    uint32_t a = 0;
    const auto hist = TensorAccessor(hist_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto qkv = TensorAccessor(qkv_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto taps = TensorAccessor(taps_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto tc = TensorAccessor(tc_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto lr = TensorAccessor(lr_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto wfb = TensorAccessor(wfb_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto wgb = TensorAccessor(wgb_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto dt = TensorAccessor(dt_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto rate = TensorAccessor(rate_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto act0 = TensorAccessor(act0_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const auto act2 = TensorAccessor(act2_args, get_arg_val<uint32_t>(ph_rt0 + a++));
    const uint32_t kk = get_arg_val<uint32_t>(ph_rt0 + a++);
    const uint32_t B = get_arg_val<uint32_t>(ph_rt0 + a++);
    const uint32_t update_hist = get_arg_val<uint32_t>(ph_rt0 + a++);
#if defined(KDA_VERIFY)
    static_assert(RSPLIT == 1 && ROWS && UP == 1, "KDA_VERIFY: ROWS mode, one row group, pair layout (KDA_PACK 2)");
    const uint32_t sel_addr = get_arg_val<uint32_t>(ph_rt0 + a++);
#endif
    if (kk >= N_CORES * RSPLIT) {
        return;  // a core past the shard's heads (megakernel owner list): no kda_pre work
    }
    const uint32_t k = kk % N_CORES, rg = kk / N_CORES;
    const bool gate = rg == 0;          // row group 0: g / gate_out / beta (KDA_RSPLIT; always with one group)
    uint32_t p0 = 0, p1 = (B + 1) / 2;  // this core's user-row pairs (ROWS mode)
    rsplit_range((B + 1) / 2, rg, p0, p1);
    const uint32_t c0 = k * CONV_PER_CORE;
    const uint32_t n0 = k * G_PER_CORE;

    // ---- masks (active rows also gate the history shift)
    cb_reserve_back(CB_ACT, 2);
    const uint32_t act_l1 = get_write_ptr(CB_ACT);
    noc_async_read_page(0, act0, act_l1);
    noc_async_read_page(0, act2, act_l1 + FP32_TILE);
    // ---- gate inputs: f_a | g_a, the f_b / g_b weight blocks of this core's 4 output tiles, dt_bias / rate rows, b
    if (gate) {
        cb_reserve_back(CB_FA, K_TILES);
        cb_reserve_back(CB_GA, K_TILES);
        cb_reserve_back(CB_WFB, K_TILES * G_PER_CORE);
        cb_reserve_back(CB_WGB, K_TILES * G_PER_CORE);
        cb_reserve_back(CB_DT, G_PER_CORE);
        cb_reserve_back(CB_RATE, G_PER_CORE);
    }
#if defined(KDA_PREFETCH)
    // Stage 14 (megakernel): everything that does not depend on this step's projections — the f_b / g_b weight tiles,
    // dt_bias / rate rows, the taps, user-pair 0's conv history — is requested BEFORE this reader waits for the
    // readers' qkv + low-rank hand-off (the owner then skips its own wait); only f_a | g_a | b and the qkv rows after
    // it
    if (gate) {
        for (uint32_t kt = 0; kt < K_TILES; ++kt) {
            for (uint32_t j = 0; j < G_PER_CORE; ++j) {
                noc_async_read_page(
                    kt * W_TILES + n0 + j, wfb, get_write_ptr(CB_WFB) + (kt * G_PER_CORE + j) * BF16_TILE);
                noc_async_read_page(
                    kt * W_TILES + n0 + j, wgb, get_write_ptr(CB_WGB) + (kt * G_PER_CORE + j) * BF16_TILE);
            }
        }
        for (uint32_t j = 0; j < G_PER_CORE; ++j) {
            read_row0_fp32(dt, n0 + j, get_write_ptr(CB_DT) + j * FP32_TILE);
            read_row0_fp32(rate, n0 + j, get_write_ptr(CB_RATE) + j * FP32_TILE);
        }
    }
    if constexpr (ROWS) {
        cb_reserve_back(CB_T, 4);
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_read_page(j * N_CORES + k, taps, get_write_ptr(CB_T) + j * BF16_TILE);
        }
#if !defined(KDA_VERIFY)  // (verify: the selected slot's history is read after the selector, below)
        if (p1 > p0) {    // the first unit of this core's rows (none: no read in flight)
            cb_reserve_back(CB_H0, 1);
            cb_reserve_back(CB_H1, 1);
            cb_reserve_back(CB_H2, 1);
            const uint32_t np0 = p1 - p0 < UP ? p1 - p0 : UP;  // unit 0's pairs (KDA_PACK)
            for (uint32_t r = 0; r < CONV_PER_CORE; ++r) {
                for (uint32_t pp = 0; pp < np0; ++pp) {
                    read_pair_bf16(hist, c0 + r, p0 + pp, get_write_ptr(CB_H0), UP * r + pp);
                    read_pair_bf16(hist, QKV_TILES + c0 + r, p0 + pp, get_write_ptr(CB_H1), UP * r + pp);
                    read_pair_bf16(hist, 2 * QKV_TILES + c0 + r, p0 + pp, get_write_ptr(CB_H2), UP * r + pp);
                }
            }
        }
#endif
    }
    mk::wait_handoff(MK_SEM_QKV, MK_QKV_COUNT);
    if (gate) {
        for (uint32_t kt = 0; kt < K_TILES; ++kt) {
            noc_async_read_page(kt, lr, get_write_ptr(CB_FA) + kt * BF16_TILE);
            noc_async_read_page(K_TILES + kt, lr, get_write_ptr(CB_GA) + kt * BF16_TILE);
        }
    }
#else
    if (gate) {
        for (uint32_t kt = 0; kt < K_TILES; ++kt) {
            noc_async_read_page(kt, lr, get_write_ptr(CB_FA) + kt * BF16_TILE);
            noc_async_read_page(K_TILES + kt, lr, get_write_ptr(CB_GA) + kt * BF16_TILE);
            for (uint32_t j = 0; j < G_PER_CORE; ++j) {
                noc_async_read_page(
                    kt * W_TILES + n0 + j, wfb, get_write_ptr(CB_WFB) + (kt * G_PER_CORE + j) * BF16_TILE);
                noc_async_read_page(
                    kt * W_TILES + n0 + j, wgb, get_write_ptr(CB_WGB) + (kt * G_PER_CORE + j) * BF16_TILE);
            }
        }
        for (uint32_t j = 0; j < G_PER_CORE; ++j) {
            read_row0_fp32(dt, n0 + j, get_write_ptr(CB_DT) + j * FP32_TILE);
            read_row0_fp32(rate, n0 + j, get_write_ptr(CB_RATE) + j * FP32_TILE);
        }
    }  // gate
#endif
    if (gate && k < BETA_TILES) {
        cb_reserve_back(CB_B, 1);
        noc_async_read_page(2 * K_TILES + k, lr, get_write_ptr(CB_B));
    }
    noc_async_read_barrier();
    uint32_t active = 0;  // bit t: user row t active (cur_pos >= 0)
    {
        volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_l1);
        for (uint32_t t = 0; t < B; ++t) {
            if ((m[tile_off(t, 0)] & 0x7FFFFFFFu) != 0) {
                active |= 1u << t;
            }
        }
    }
    if (gate) {  // (row groups > 0 read the masks for the history shift only: compute consumes them in the gate part)
        cb_push_back(CB_ACT, 2);
        cb_push_back(CB_FA, K_TILES);
        cb_push_back(CB_GA, K_TILES);
        cb_push_back(CB_WFB, K_TILES * G_PER_CORE);
        cb_push_back(CB_WGB, K_TILES * G_PER_CORE);
        cb_push_back(CB_DT, G_PER_CORE);
        cb_push_back(CB_RATE, G_PER_CORE);
        if (k < BETA_TILES) {
            cb_push_back(CB_B, 1);
        }
    }

    // ---- conv units
    if constexpr (ROWS) {
        // packed tile per unit u: row KDA_PACK r + p = column tile c0 + r, user row KDA_PACK u + p (kda_pre_common.hpp)
#if !defined(KDA_PREFETCH)  // (prefetched: requested before the hand-off, landed by the first barrier)
        cb_reserve_back(CB_T, 4);
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_read_page(j * N_CORES + k, taps, get_write_ptr(CB_T) + j * BF16_TILE);
        }
        noc_async_read_barrier();
#endif
        cb_push_back(CB_T, 4);
        const uint32_t pairs = p1 - p0;  // this core's pairs p0 .. p1 - 1 (all of them with one row group)
        // mixed-pair history rows (3 targets x 2 face halves x 64 B) are assembled in L1 (CB_SCR, 64-B aligned: the
        // NoC cannot read the RISC's local memory)
        const uint32_t s0 = get_write_ptr(CB_SCR);
#if defined(KDA_VERIFY)
        {
            // pages: H0 / H1 / H2 page u = the windows of rows 2u, 2u + 1; page 1 first holds slot `sel`'s history pair
            // (read before any write-back), consumed before page 1 is assembled (H0, then H1, then H2: page 1's H0 rows
            // read H2's slot rows, which H2 overwrites last); CB_Q page u = the pre-conv rows 2u, 2u + 1
            // (B + 1) / 2 pairs: CB_H* / CB_Q hold max(2, units) pages (page 1 of CB_H* is the slot pair's scratch
            // even at units = 1); Stage 17 t = 8 verify: 4 pairs (KDA_PRE_REV 2)
            const uint32_t units = pairs;
            const uint32_t hpages = units < 2 ? 2 : units;
            const InterleavedAddrGen<true> sg = {.bank_base_address = sel_addr, .page_size = 128};
            noc_async_read(sg.get_noc_addr(0), s0, 64);
            noc_async_read_barrier();
            const uint32_t sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s0)[0];
            cb_reserve_back(CB_H0, hpages);
            cb_reserve_back(CB_H1, hpages);
            cb_reserve_back(CB_H2, hpages);
            cb_reserve_back(CB_Q, units);
            const uint32_t hb[3] = {get_write_ptr(CB_H0), get_write_ptr(CB_H1), get_write_ptr(CB_H2)};
            const uint32_t qb = get_write_ptr(CB_Q);
            for (uint32_t r = 0; r < CONV_PER_CORE; ++r) {
                const uint32_t c = c0 + r;
                for (uint32_t sl = 0; sl < HIST; ++sl) {
                    read_pair_bf16(hist, sl * QKV_TILES + c, sel / 2, hb[sl] + BF16_TILE, r);
                }
                for (uint32_t u = 0; u < units; ++u) {
                    read_pair_bf16(qkv, c, u, qb + u * BF16_TILE, r);
                }
            }
            noc_async_read_barrier();
            // packed row R of a bf16 tile: two 16-value face-half rows (32 B each)
            auto copy_row = [](uint32_t dst, uint32_t dr, uint32_t src, uint32_t sr) {
                for (uint32_t hh = 0; hh < 2; ++hh) {
                    volatile tt_l1_ptr uint32_t* d =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + tile_off(dr, 16 * hh) * 2);
                    volatile tt_l1_ptr uint32_t* q =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + tile_off(sr, 16 * hh) * 2);
                    for (uint32_t w = 0; w < 8; ++w) {
                        d[w] = q[w];
                    }
                }
            };
            for (uint32_t u = 0; u < units; ++u) {
                for (uint32_t sl = 0; sl < HIST; ++sl) {  // H_sl[j] = src(j - 3 + sl)
                    for (uint32_t p = 0; p < 2; ++p) {
                        const int32_t i = static_cast<int32_t>(2 * u + p) - 3 + static_cast<int32_t>(sl);
                        for (uint32_t r = 0; r < CONV_PER_CORE; ++r) {
                            if (i < 0) {
                                copy_row(hb[sl] + u * BF16_TILE, 2 * r + p, hb[3 + i] + BF16_TILE, 2 * r + (sel & 1u));
                            } else {
                                copy_row(hb[sl] + u * BF16_TILE, 2 * r + p, qb + (i / 2) * BF16_TILE, 2 * r + (i & 1));
                            }
                        }
                    }
                }
            }
            // candidate histories: row j <- (H1_j, H2_j, Q_j) (every verify row is a real token)
            if (update_hist) {
                for (uint32_t u = 0; u < units; ++u) {
                    for (uint32_t r = 0; r < CONV_PER_CORE; ++r) {
                        const uint32_t c = c0 + r;
                        write_pair_bf16(hist, c, u, hb[1] + u * BF16_TILE, r);
                        write_pair_bf16(hist, QKV_TILES + c, u, hb[2] + u * BF16_TILE, r);
                        write_pair_bf16(hist, 2 * QKV_TILES + c, u, qb + u * BF16_TILE, r);
                    }
                }
            }
            cb_push_back(CB_H0, units);
            cb_push_back(CB_H1, units);
            cb_push_back(CB_H2, units);
            cb_push_back(CB_Q, units);
            noc_async_write_barrier();
        }
#elif defined(KDA_BATCH)
        // GLM53_KDA_MODE=batch: every pair's reads behind ONE barrier and the history write-back behind one final
        // barrier (instead of a read and a write barrier per row pair); the CBs hold every pair (no page reuse inside
        // the phase); the compute starts only after all pairs landed
        static_assert(UP == 1, "KDA_BATCH: pair layout only (KDA_PACK 2)");
        cb_reserve_back(CB_H0, pairs);
        cb_reserve_back(CB_H1, pairs);
        cb_reserve_back(CB_H2, pairs);
        cb_reserve_back(CB_Q, pairs);
        const uint32_t h0b = get_write_ptr(CB_H0), h1b = get_write_ptr(CB_H1), h2b = get_write_ptr(CB_H2);
        const uint32_t qb = get_write_ptr(CB_Q);
        for (uint32_t i = 0; i < pairs; ++i) {
            const uint32_t u = p0 + i;  // global pair; i = this core's page
            for (uint32_t r = 0; r < CONV_PER_CORE; ++r) {
                const uint32_t c = c0 + r;
                read_pair_bf16(hist, c, u, h0b + i * BF16_TILE, r);
                read_pair_bf16(hist, QKV_TILES + c, u, h1b + i * BF16_TILE, r);
                read_pair_bf16(hist, 2 * QKV_TILES + c, u, h2b + i * BF16_TILE, r);
                read_pair_bf16(qkv, c, u, qb + i * BF16_TILE, r);
            }
        }
        noc_async_read_barrier();
        if (update_hist) {
            for (uint32_t i = 0; i < pairs; ++i) {
                const uint32_t u = p0 + i;
                const bool a0 = (active >> (2 * u)) & 1u;
                const bool a1 =
                    (2 * u + 1 < B) && ((active >> (2 * u + 1)) & 1u);  // row B (odd B) is another slot: never written
                if (!(a0 || a1)) {
                    continue;
                }
                const uint32_t h0 = h0b + i * BF16_TILE, h1 = h1b + i * BF16_TILE, h2 = h2b + i * BF16_TILE;
                const uint32_t q = qb + i * BF16_TILE;
                for (uint32_t r = 0; r < CONV_PER_CORE; ++r) {
                    const uint32_t c = c0 + r;
                    if (a0 && a1) {
                        write_pair_bf16(hist, c, u, h1, r);
                        write_pair_bf16(hist, QKV_TILES + c, u, h2, r);
                        write_pair_bf16(hist, 2 * QKV_TILES + c, u, q, r);
                    } else {
                        const uint32_t srcs_new[3] = {h1, h2, q};
                        const uint32_t srcs_old[3] = {h0, h1, h2};
                        for (uint32_t j = 0; j < 3; ++j) {
                            for (uint32_t hh = 0; hh < 2; ++hh) {
                                const uint32_t dst = s0 + (j * 2 + hh) * 64;
                                assemble_pair(dst, srcs_new[j], srcs_old[j], r, hh, a0, a1);
                                noc_async_write(
                                    dst, hist.get_noc_addr(j * QKV_TILES + c, tile_off(2 * u, 16 * hh) * 2), 64);
                            }
                        }
                        noc_async_write_barrier();  // the scratch is reused for the next column tile
                    }
                }
            }
        }
        // compute only reads these pages (the write-back's sources): push before the final write barrier
        cb_push_back(CB_H0, pairs);
        cb_push_back(CB_H1, pairs);
        cb_push_back(CB_H2, pairs);
        cb_push_back(CB_Q, pairs);
        noc_async_write_barrier();
#else
        // unit i = this core's pairs p0 + UP i .. (KDA_PACK 2: unit = pair); global pair up -> packed row pair rp
        const uint32_t n_units = (pairs + UP - 1) / UP;
        for (uint32_t i = 0; i < n_units; ++i) {
            cb_reserve_back(CB_H0, 1);
            cb_reserve_back(CB_H1, 1);
            cb_reserve_back(CB_H2, 1);
            cb_reserve_back(CB_Q, 1);
            const uint32_t h0 = get_write_ptr(CB_H0), h1 = get_write_ptr(CB_H1), h2 = get_write_ptr(CB_H2);
            const uint32_t q = get_write_ptr(CB_Q);
            const uint32_t np = pairs - UP * i < UP ? pairs - UP * i : UP;
            for (uint32_t r = 0; r < CONV_PER_CORE; ++r) {
                const uint32_t c = c0 + r;
                for (uint32_t pp = 0; pp < np; ++pp) {
                    const uint32_t up = p0 + UP * i + pp, rp = UP * r + pp;
#if defined(KDA_PREFETCH)
                    if (i != 0)  // this core's first unit's history was requested before the hand-off (same pages)
#endif
                    {
                        read_pair_bf16(hist, c, up, h0, rp);
                        read_pair_bf16(hist, QKV_TILES + c, up, h1, rp);
                        read_pair_bf16(hist, 2 * QKV_TILES + c, up, h2, rp);
                    }
                    read_pair_bf16(qkv, c, up, q, rp);
                }
            }
            noc_async_read_barrier();
            bool wrote = false;
            for (uint32_t pp = 0; pp < np; ++pp) {
                const uint32_t up = p0 + UP * i + pp;
                const bool a0 = (active >> (2 * up)) & 1u;
                const bool a1 = (2 * up + 1 < B) &&
                                ((active >> (2 * up + 1)) & 1u);  // row B (odd B) is another slot: never written
                if (!(update_hist && (a0 || a1))) {
                    continue;
                }
                wrote = true;
                // history rows t-3, t-2, t-1 <- t-2, t-1, the new pre-conv row; inactive users keep theirs
                for (uint32_t r = 0; r < CONV_PER_CORE; ++r) {
                    const uint32_t c = c0 + r, rp = UP * r + pp;
                    if (a0 && a1) {
                        write_pair_bf16(hist, c, up, h1, rp);
                        write_pair_bf16(hist, QKV_TILES + c, up, h2, rp);
                        write_pair_bf16(hist, 2 * QKV_TILES + c, up, q, rp);
                    } else {
                        const uint32_t srcs_new[3] = {h1, h2, q};
                        const uint32_t srcs_old[3] = {h0, h1, h2};
                        for (uint32_t j = 0; j < 3; ++j) {
                            for (uint32_t hh = 0; hh < 2; ++hh) {
                                const uint32_t dst = s0 + (j * 2 + hh) * 64;
                                assemble_pair(dst, srcs_new[j], srcs_old[j], rp, hh, a0, a1);
                                noc_async_write(
                                    dst, hist.get_noc_addr(j * QKV_TILES + c, tile_off(2 * up, 16 * hh) * 2), 64);
                            }
                        }
                        noc_async_write_barrier();  // the scratch is reused for the next column tile
                    }
                }
            }
            if (wrote) {
                noc_async_write_barrier();  // the packed tiles are reused for the next unit after the compute pops
            }
            cb_push_back(CB_H0, 1);
            cb_push_back(CB_H1, 1);
            cb_push_back(CB_H2, 1);
            cb_push_back(CB_Q, 1);
        }
#endif  // KDA_BATCH
    } else {
#if defined(KDA_TILES_HIST)
        // 7a KDA_TILES_HIST (GLM53_KDA_TILES_MIN, the megakernel at large B on a TP shard: 3 column tiles per core at
        // spread 4 instead of B / 2 row pairs): the history shift of the active users happens here too. Every user row
        // active: whole-tile writes (history rows j <- j + 1, row 2 <- the pre-conv qkv tile); otherwise per row pair
        // as in ROWS mode (rows (2u, 2u + 1) of the native tile; mixed pairs assembled in CB_SCR)
        const uint32_t s0 = get_write_ptr(CB_SCR);
        const uint32_t pairs = (B + 1) / 2;
        const uint32_t live = B >= 32 ? 0xFFFFFFFFu : ((1u << B) - 1u);
        // whole tiles only when the tile's 32 rows are all this program's (a t = w < 32 program runs on the SHARED
        // 32-row state: rows w..31 are other slots)
        const bool all_active = B >= 32 && (active & live) == live;
#endif
        for (uint32_t r = 0; r < CONV_PER_CORE; ++r) {
            const uint32_t c = c0 + r;
            cb_reserve_back(CB_H0, 1);
            cb_reserve_back(CB_H1, 1);
            cb_reserve_back(CB_H2, 1);
            cb_reserve_back(CB_Q, 1);
            cb_reserve_back(CB_T, 4);
            noc_async_read_page(c, hist, get_write_ptr(CB_H0));
            noc_async_read_page(QKV_TILES + c, hist, get_write_ptr(CB_H1));
            noc_async_read_page(2 * QKV_TILES + c, hist, get_write_ptr(CB_H2));
            noc_async_read_page(c, qkv, get_write_ptr(CB_Q));
            const uint32_t t_l1 = get_write_ptr(CB_T);
            for (uint32_t j = 0; j < 3; ++j) {
                read_row0_fp32(taps, j * QKV_TILES + c, t_l1 + j * FP32_TILE);
            }
            read_row0_fp32(tc, c, t_l1 + 3 * FP32_TILE);
            noc_async_read_barrier();
#if defined(KDA_TILES_HIST)
            if (update_hist) {
                const uint32_t h0 = get_write_ptr(CB_H0), h1 = get_write_ptr(CB_H1), h2 = get_write_ptr(CB_H2);
                const uint32_t q = get_write_ptr(CB_Q);
                if (all_active) {
                    noc_async_write(h1, hist.get_noc_addr(c, 0), BF16_TILE);
                    noc_async_write(h2, hist.get_noc_addr(QKV_TILES + c, 0), BF16_TILE);
                    noc_async_write(q, hist.get_noc_addr(2 * QKV_TILES + c, 0), BF16_TILE);
                } else {
                    for (uint32_t u = 0; u < pairs; ++u) {
                        const bool a0 = (active >> (2 * u)) & 1u;
                        const bool a1 = (2 * u + 1 < B) &&
                                        ((active >> (2 * u + 1)) & 1u);  // row B (odd B) is another slot: never written
                        if (!(a0 || a1)) {
                            continue;
                        }
                        if (a0 && a1) {
                            write_pair_bf16(hist, c, u, h1, u);
                            write_pair_bf16(hist, QKV_TILES + c, u, h2, u);
                            write_pair_bf16(hist, 2 * QKV_TILES + c, u, q, u);
                        } else {
                            const uint32_t srcs_new[3] = {h1, h2, q};
                            const uint32_t srcs_old[3] = {h0, h1, h2};
                            for (uint32_t j = 0; j < 3; ++j) {
                                for (uint32_t hh = 0; hh < 2; ++hh) {
                                    const uint32_t dst = s0 + (j * 2 + hh) * 64;
                                    assemble_pair(dst, srcs_new[j], srcs_old[j], u, hh, a0, a1);
                                    noc_async_write(
                                        dst, hist.get_noc_addr(j * QKV_TILES + c, tile_off(2 * u, 16 * hh) * 2), 64);
                                }
                            }
                            noc_async_write_barrier();  // the scratch is reused for the next pair
                        }
                    }
                }
                noc_async_write_barrier();  // the tiles are reused for the next unit after the compute pops
            }
#endif
            cb_push_back(CB_H0, 1);
            cb_push_back(CB_H1, 1);
            cb_push_back(CB_H2, 1);
            cb_push_back(CB_Q, 1);
            cb_push_back(CB_T, 4);
        }
    }
}
