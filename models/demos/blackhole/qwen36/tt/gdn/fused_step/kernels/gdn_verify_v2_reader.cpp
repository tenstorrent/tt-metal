// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GDN fused verify v2, reader (fused_step/gdn_verify_v2.py). Core (h, j): value head h, value-column tile j, key head
// kh. T tokens; token t is the v2 decode step (gdn_step_v2_reader.cpp) for verify row t.
// 0. once: state_blk_idx (mi = idx[0] / Nv, this head's ring block idx[h]); the full channel tiles of E_prev
//    (_verify_win_buf) and of the new rows (proj) for the core's dense slots (q_t, k_t of kh, v_j; + z_j, a|b),
//    the 4 dense tap tiles, constants, gates, norm weight, the initial state S = ring block idx[h]; zero the dense
//    tiles, the split targets and both gather tiles; then +1 on the semaphore of every core of the key group.
// per token t:
// 1. dense tiles H1, H2, H3, X (even rows only; odd rows stay 0) from rows t, t+1, t+2, t+3 of the virtual window
//    E_new = [E_prev rows mi+1 .. mi+3 ; proj rows 0 .. T-1] (z_j row t into slot NS of X), a_t / b_t
// 2. split the compute core's normalised dense tile (v2 step 4 verbatim)
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t cb_h = 0, cb_x = 1, cb_tap = 2, cb_abraw = 3;
constexpr uint32_t cb_const = 6, cb_n = 10, cb_aq = 11, cb_ak = 12, cb_kcol = 13, cb_v = 14, cb_sz = 15;
constexpr uint32_t cb_a = 16, cb_b = 17, cb_na = 18, cb_dt = 19, cb_S = 22, cb_gather = 31, cb_w = 33;
constexpr uint32_t cb_ep = 34, cb_pj = 35, cb_idx = 37;
constexpr uint32_t NCONST = 6;

constexpr uint32_t BF = 2048, F32 = 4096;

// byte offset of dense slot s (row r = 2s, or 16 + 2(s - 8)), face half f (0: columns 0-15, 1: 16-31)
template <uint32_t face_bytes, uint32_t row_bytes>
constexpr uint32_t slot_off(uint32_t s, uint32_t f) {
    const uint32_t r = s < 8 ? 2 * s : 16 + 2 * (s - 8);
    return ((r >= 16 ? 2 : 0) + f) * face_bytes + (r % 16) * row_bytes;
}

// word offset of row r, face half f of a bf16 tile
FORCE_INLINE uint32_t bf_row_w(uint32_t r, uint32_t f) { return ((r >= 16 ? 2 : 0) + f) * 128 + (r % 16) * 8; }

FORCE_INLINE void zero_fill(uint32_t dst, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t z = 0; z < bytes; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + z, (bytes - z) < MEM_ZEROS_SIZE ? (bytes - z) : MEM_ZEROS_SIZE);
    }
}

// row r of the bf16 tile at src -> dense slot s of the bf16 tile at dst (32 bytes per face half)
FORCE_INLINE void row_to_slot(uint32_t src, uint32_t r, uint32_t dst, uint32_t s) {
    volatile tt_l1_ptr uint32_t* ps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* pd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t f = 0; f < 2; f++) {
        const uint32_t so = bf_row_w(r, f);
        const uint32_t dof = slot_off<512, 32>(s, f) / 4;
        for (uint32_t e = 0; e < 8; e++) {
            pd[dof + e] = ps[so + e];
        }
    }
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t z_off = get_compile_time_arg_val(2);
    constexpr uint32_t ab_tile = get_compile_time_arg_val(3);
    constexpr uint32_t a_col = get_compile_time_arg_val(4);
    constexpr uint32_t Nv = get_compile_time_arg_val(5);
    constexpr uint32_t sem_rd = get_compile_time_arg_val(6);
    constexpr uint32_t G = get_compile_time_arg_val(7);  // cores of the key group
    constexpr uint32_t T = get_compile_time_arg_val(8);
    constexpr auto proj_args = TensorAccessorArgs<9>();
    constexpr auto win_args = TensorAccessorArgs<proj_args.next_compile_time_args_offset()>();
    constexpr auto taps_args = TensorAccessorArgs<win_args.next_compile_time_args_offset()>();
    constexpr auto gconst_args = TensorAccessorArgs<taps_args.next_compile_time_args_offset()>();
    constexpr auto normw_args = TensorAccessorArgs<gconst_args.next_compile_time_args_offset()>();
    constexpr auto consts_args = TensorAccessorArgs<normw_args.next_compile_time_args_offset()>();
    constexpr auto ring_args = TensorAccessorArgs<consts_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    constexpr uint32_t NS = 2 * Kt + 1;  // conv slots: q_t, k_t, v_j (z_j = slot NS, new input only)
    constexpr uint32_t KC1 = 3;          // K - 1 conv history rows

    uint32_t i = 0;
    const uint32_t proj_addr = get_arg_val<uint32_t>(i++);
    const uint32_t win_addr = get_arg_val<uint32_t>(i++);
    const uint32_t taps_addr = get_arg_val<uint32_t>(i++);
    const uint32_t gconst_addr = get_arg_val<uint32_t>(i++);
    const uint32_t normw_addr = get_arg_val<uint32_t>(i++);
    const uint32_t consts_addr = get_arg_val<uint32_t>(i++);
    const uint32_t ring_addr = get_arg_val<uint32_t>(i++);
    const uint32_t idx_addr = get_arg_val<uint32_t>(i++);
    const uint32_t h = get_arg_val<uint32_t>(i++);
    const uint32_t j = get_arg_val<uint32_t>(i++);
    const uint32_t ci = get_arg_val<uint32_t>(i++);  // core index h * Vt + j
    uint32_t ch[NS];
    for (uint32_t s = 0; s < NS; s++) {
        ch[s] = get_arg_val<uint32_t>(i++);
    }
    const uint32_t group_arg = i;  // G (x, y) pairs

    const auto proj = TensorAccessor(proj_args, proj_addr, BF);
    const auto win = TensorAccessor(win_args, win_addr, BF);
    const auto taps = TensorAccessor(taps_args, taps_addr, BF);
    const auto gconst = TensorAccessor(gconst_args, gconst_addr, F32);
    const auto normw = TensorAccessor(normw_args, normw_addr, F32);
    const auto consts = TensorAccessor(consts_args, consts_addr, F32);
    const auto ring = TensorAccessor(ring_args, ring_addr, F32);
    const auto idxa = TensorAccessor(idx_args, idx_addr, 64);

    // ---- 0. one-time loads ----
    uint32_t mi, blk;
    uint32_t ep, pj, ab;
    {
        DeviceZoneScopedN("RV-load");
        cb_reserve_back(cb_idx, 1);
        const uint32_t ib = get_write_ptr(cb_idx);
        noc_async_read(idxa.get_noc_addr(0), ib, 64);
        // E_prev and proj channel tiles (+ z_j, a|b)
        cb_reserve_back(cb_ep, NS);
        cb_reserve_back(cb_pj, NS + 1);
        ep = get_write_ptr(cb_ep);
        pj = get_write_ptr(cb_pj);
        for (uint32_t s = 0; s < NS; s++) {
            noc_async_read_page(ch[s], win, ep + s * BF);
            noc_async_read_page(ch[s], proj, pj + s * BF);
        }
        noc_async_read_page(z_off + h * Vt + j, proj, pj + NS * BF);
        cb_reserve_back(cb_abraw, 1);
        ab = get_write_ptr(cb_abraw);
        noc_async_read_page(ab_tile, proj, ab);
        cb_reserve_back(cb_tap, 4);
        uint32_t dst = get_write_ptr(cb_tap);
        for (uint32_t k = 0; k < 4; k++, dst += BF) {
            noc_async_read_page(ci * 4 + k, taps, dst);
        }
        cb_reserve_back(cb_const, NCONST);
        dst = get_write_ptr(cb_const);
        for (uint32_t t = 0; t < NCONST; t++, dst += F32) {
            noc_async_read_page(t, consts, dst);
        }
        cb_reserve_back(cb_na, 1);
        cb_reserve_back(cb_dt, 1);
        noc_async_read_page(2 * h, gconst, get_write_ptr(cb_na));
        noc_async_read_page(2 * h + 1, gconst, get_write_ptr(cb_dt));
        cb_reserve_back(cb_w, 1);
        noc_async_read_page(j, normw, get_write_ptr(cb_w));
        // zero: dense tiles, a/b, split targets, both gather tiles
        cb_reserve_back(cb_h, 3);
        cb_reserve_back(cb_x, 1);
        zero_fill(get_write_ptr(cb_h), 3 * BF);
        zero_fill(get_write_ptr(cb_x), BF);
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        zero_fill(get_write_ptr(cb_a), F32);
        zero_fill(get_write_ptr(cb_b), F32);
        cb_reserve_back(cb_aq, Kt);
        cb_reserve_back(cb_ak, Kt);
        cb_reserve_back(cb_kcol, Kt);
        cb_reserve_back(cb_v, 1);
        cb_reserve_back(cb_sz, 1);
        zero_fill(get_write_ptr(cb_aq), Kt * F32);
        zero_fill(get_write_ptr(cb_ak), Kt * F32);
        zero_fill(get_write_ptr(cb_kcol), Kt * F32);
        zero_fill(get_write_ptr(cb_v), F32);
        zero_fill(get_write_ptr(cb_sz), F32);
        zero_fill(get_write_ptr(cb_gather), 2 * F32);  // the writer produces cb_gather; same addresses
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ib);
        mi = idx[0] / Nv;
        blk = idx[h];
        cb_push_back(cb_idx, 1);  // -> writer (mi for the E_new window)
        // initial state: ring block blk, value-column block j
        cb_reserve_back(cb_S, Kt);
        dst = get_write_ptr(cb_S);
        for (uint32_t t = 0; t < Kt; t++, dst += F32) {
            noc_async_read_page((blk * Kt + t) * Vt + j, ring, dst);
        }
        noc_async_read_barrier();
        cb_push_back(cb_ep, NS);      // -> writer (E_new of the owned channels); still read here
        cb_push_back(cb_pj, NS + 1);  // idem
        cb_push_back(cb_tap, 4);
        cb_push_back(cb_const, NCONST);
        cb_push_back(cb_na, 1);
        cb_push_back(cb_dt, 1);
        cb_push_back(cb_w, 1);
        cb_push_back(cb_S, Kt);
    }
    // every read of E_prev is done and the gather tiles are zero
    for (uint32_t g = 0; g < G; g++) {
        const uint32_t px = get_arg_val<uint32_t>(group_arg + 2 * g);
        const uint32_t py = get_arg_val<uint32_t>(group_arg + 2 * g + 1);
        noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(sem_rd)), 1);
    }

    volatile tt_l1_ptr uint16_t* raw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ab);
    for (uint32_t t = 0; t < T; t++) {
        // ---- 1. dense conv inputs of token t ----
        {
            DeviceZoneScopedN("RV-dense");
            cb_reserve_back(cb_h, 3);
            cb_reserve_back(cb_x, 1);
            const uint32_t hb = get_write_ptr(cb_h);
            const uint32_t xb = get_write_ptr(cb_x);
            for (uint32_t k = 0; k < 3; k++) {
                const uint32_t r = t + k;  // window row
                const bool from_prev = r < KC1;
                const uint32_t base = from_prev ? ep : pj;
                const uint32_t row = from_prev ? (mi + 1 + r) : (r - KC1);
                for (uint32_t s = 0; s < NS; s++) {
                    row_to_slot(base + s * BF, row, hb + k * BF, s);
                }
            }
            for (uint32_t s = 0; s <= NS; s++) {  // q, k, v and z_j (slot NS) of new row t
                row_to_slot(pj + s * BF, t, xb, s);
            }
            cb_push_back(cb_h, 3);
            cb_push_back(cb_x, 1);
            // a_t, b_t at [0, 0] (the rest of the tile stays 0)
            cb_reserve_back(cb_a, 1);
            cb_reserve_back(cb_b, 1);
            const uint32_t ro = ((t >= 16) ? 512 : 0) + (t % 16) * 16;
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_a))[0] =
                static_cast<uint32_t>(raw[ro + a_col + h]) << 16;
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_b))[0] =
                static_cast<uint32_t>(raw[ro + a_col + Nv + h]) << 16;
            cb_push_back(cb_a, 1);
            cb_push_back(cb_b, 1);
        }
        // ---- 2. split the normalised dense tile (v2 verbatim) ----
        {
            DeviceZoneScopedN("RV-split");
            cb_reserve_back(cb_aq, Kt);
            cb_reserve_back(cb_ak, Kt);
            cb_reserve_back(cb_kcol, Kt);
            cb_reserve_back(cb_v, 1);
            cb_reserve_back(cb_sz, 1);
            cb_wait_front(cb_n, 1);
            volatile tt_l1_ptr uint32_t* n = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_n));
            auto row_copy = [&](uint32_t s, uint32_t dst_addr) {
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
                for (uint32_t f = 0; f < 2; f++) {
                    const uint32_t so = slot_off<1024, 64>(s, f) / 4;
                    for (uint32_t e = 0; e < 16; e++) {
                        d[f * 256 + e] = n[so + e];
                    }
                }
            };
            for (uint32_t kt = 0; kt < Kt; kt++) {
                row_copy(kt, get_write_ptr(cb_aq) + kt * F32);
                row_copy(Kt + kt, get_write_ptr(cb_ak) + kt * F32);
                volatile tt_l1_ptr uint32_t* d =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_kcol) + kt * F32);
                for (uint32_t f = 0; f < 2; f++) {
                    const uint32_t so = slot_off<1024, 64>(Kt + kt, f) / 4;
                    for (uint32_t e = 0; e < 16; e++) {
                        d[f * 512 + e * 16] = n[so + e];
                    }
                }
            }
            row_copy(2 * Kt, get_write_ptr(cb_v));
            row_copy(2 * Kt + 1, get_write_ptr(cb_sz));
            cb_pop_front(cb_n, 1);
            cb_push_back(cb_aq, Kt);
            cb_push_back(cb_ak, Kt);
            cb_push_back(cb_kcol, Kt);
            cb_push_back(cb_v, 1);
            cb_push_back(cb_sz, 1);
        }
    }
}
