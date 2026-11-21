// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// 2D shared-fold ternary prefill linear (linear6.py), fold inputs, folded-tile exchange and output writer (NCRISC).
// Per column block and group of J kbs the block's c_5 slot is laid out in column blocks of CP columns (the last maybe
// narrower), each [k][column]; unit row q = cb J + j (column block cb, kb j) is its 4 x CP adjacent tiles, from the
// units (column, kb j) of the block. The unit rows split into R contiguous shares, one per active core of the core
// column (index ri). For its share this core reads each unit's 4 ternary tiles (1 KB, adjacent in the column's decode
// record) into c_1 and the unit's 32 scales (one 64-B row of the scale rows) into c_3, and writes diag(scales) into c_6
// for the compute's fold (the c_6 ring is zeroed once; only diagonals change). The compute packs the share's folded
// tiles into c_7; this kernel then writes them into the same place of the group's c_5 slot on every core of the column
// (multicast, plus a local copy). Two counting semaphores per core: `free`
// (every core adds 1 at each column core once its slot for the group is reserved) and `full` (every core adds 1 at each
// column core once its share has landed there); a share goes out once free reaches R (g + 1), the slot is pushed once
// full does. The inputs of group g + 1 are prepared before the exchange of group g (the compute folds g + 1 before its
// matmuls of g).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t PPR4K = get_common_arg_val<uint32_t>(1);
    const uint32_t sr_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t Nt_out = get_common_arg_val<uint32_t>(4);
    const uint32_t KB = get_common_arg_val<uint32_t>(5);
    // split output (linear2.py split=): as t2_writer.cpp
    const uint32_t ob_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t NA = get_common_arg_val<uint32_t>(7);
    const uint32_t HA = get_common_arg_val<uint32_t>(8), HB = get_common_arg_val<uint32_t>(9);
    const uint32_t Mt = get_common_arg_val<uint32_t>(10);
    const uint32_t res_addr = get_common_arg_val<uint32_t>(11);
    const uint32_t n_first = get_arg_val<uint32_t>(0), n_count = get_arg_val<uint32_t>(1);
    const uint32_t m_first = get_arg_val<uint32_t>(2), m_count = get_arg_val<uint32_t>(3);
    const uint32_t ri = get_arg_val<uint32_t>(4), R = get_arg_val<uint32_t>(5);
    const uint32_t me_x = get_arg_val<uint32_t>(6), me_y = get_arg_val<uint32_t>(7);
    // then the column's R cores (x, y), then the multicast rectangles over them (count, then x0, y0, x1, y1,
    // destinations each; this core excluded from the counts)
    const uint32_t RECT = 8 + 2 * R;
    const uint32_t nrect = get_arg_val<uint32_t>(RECT);
    constexpr uint32_t NBLK = get_compile_time_arg_val(0);
    constexpr uint32_t J = get_compile_time_arg_val(1);
    constexpr uint32_t UM = get_compile_time_arg_val(2);  // units per core and group, at most
    constexpr uint32_t CP = get_compile_time_arg_val(4);  // output columns per matmul pass (t6_compute.cpp)
    constexpr uint32_t KT = 4 * J;
    constexpr uint32_t SLOT = NBLK * KT;  // c_5 tiles per group
    constexpr auto w_args = TensorAccessorArgs<5>();
    constexpr auto s_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto ob_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<ob_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_w = tt::CBIndex::c_1;
    constexpr uint32_t cb_s = tt::CBIndex::c_3;
    constexpr uint32_t cb_wx = tt::CBIndex::c_5;
    constexpr uint32_t cb_dg = tt::CBIndex::c_6;
    constexpr uint32_t cb_fold = tt::CBIndex::c_7;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_res = tt::CBIndex::c_4;
    constexpr uint32_t GROUP = 2048 + 32 * 4 * 256;
#ifdef GLU_PAIRS
    constexpr uint32_t OUT_DIV = 2;
#else
    constexpr uint32_t OUT_DIV = 1;
#endif
    if (n_count == 0 || m_count == 0) {
        return;
    }
    const auto w = TensorAccessor(w_args, w_addr, 4096);
    const auto sr = TensorAccessor(s_args, sr_addr, 64);
    const uint32_t obytes = get_tile_size(cb_out);
    const uint32_t WTB = get_tile_size(cb_wx);
    const auto out = TensorAccessor(o_args, out_addr, obytes);
    const auto outb = TensorAccessor(ob_args, ob_addr, obytes);
    const auto res = TensorAccessor(r_args, res_addr, obytes);
    const uint32_t NB = Nt_out - NA;
    auto dest = [&](uint32_t row, uint32_t n) -> uint64_t {
        if (n < NA) {
            return out.get_noc_addr(HA ? ((n / HA) * Mt + row) * HA + n % HA : row * NA + n);
        }
        const uint32_t nn = n - NA;
        return outb.get_noc_addr(HB ? ((nn / HB) * Mt + row) * HB + nn % HB : row * NB + nn);
    };
    const uint32_t full = get_semaphore(4);
    const uint32_t free = get_semaphore(5);
    volatile tt_l1_ptr uint32_t* full_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(full);
    volatile tt_l1_ptr uint32_t* free_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(free);
    auto mcast = [&](uint32_t q, uint32_t l1) {  // NOC 1 runs the other way: its rectangle starts at the far corner
        const uint32_t b = RECT + 1 + 5 * q;
        const uint32_t x0 = get_arg_val<uint32_t>(b), y0 = get_arg_val<uint32_t>(b + 1);
        const uint32_t x1 = get_arg_val<uint32_t>(b + 2), y1 = get_arg_val<uint32_t>(b + 3);
        return noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, l1) : get_noc_multicast_addr(x1, y1, x0, y0, l1);
    };
    auto inc_all = [&](uint32_t sem) {
        for (uint32_t i = 0; i < R; ++i) {
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(8 + 2 * i), get_arg_val<uint32_t>(9 + 2 * i), sem), 1);
        }
    };
    {  // zero the diag tiles' ring
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_dg));
        const uint32_t words = get_local_cb_interface(cb_dg).fifo_size >> 2;  // bytes on the data movement cores
        for (uint32_t i = 0; i < words; ++i) {
            z[i] = 0;
        }
    }
    const uint32_t NG = KB / J;
    uint32_t gi = 0;  // groups exchanged so far (all blocks)
    for (uint32_t c0 = 0; c0 < n_count; c0 += NBLK) {
        const uint32_t nb = n_count - c0 < NBLK ? n_count - c0 : NBLK;
        // unit rows q = cb J + j (column block cb of CP columns, kb j): 4 CP adjacent c_5 tiles, [k][column]
        const uint32_t Q = (nb + CP - 1) / CP * J;
        const uint32_t q0 = Q * ri / R, q1 = Q * (ri + 1) / R;
        // the fold inputs of group g: this core's units' ternary tiles, scale rows and diag tiles, in unit-row order
        auto prep = [&](uint32_t g) {
            cb_reserve_back(cb_w, UM * 4);
            cb_reserve_back(cb_s, UM);
            cb_reserve_back(cb_dg, UM);
            const uint32_t lw = get_write_ptr(cb_w), ls = get_write_ptr(cb_s), ld = get_write_ptr(cb_dg);
            uint32_t ul = 0;
            for (uint32_t q = q0; q < q1; ++q) {
                const uint32_t c1 = q / J * CP, kb = g * J + q % J;
                const uint32_t off = (kb / 32) * GROUP + 2048 + (kb % 32) * 1024;  // never straddles a 4-KB page
                for (uint32_t c = c1; c < c1 + CP && c < nb; ++c, ++ul) {
                    const uint32_t n = n_first + c0 + c;
#ifndef T6_NOWREAD
                    noc_async_read(w.get_noc_addr(n * PPR4K + off / 4096, off % 4096), lw + ul * 1024, 1024);
                    noc_async_read(sr.get_noc_addr(n * KB + kb), ls + ul * 64, 64);
#endif
                }
            }
            noc_async_read_barrier();
            for (uint32_t u = 0; u < ul; ++u) {
                volatile tt_l1_ptr uint16_t* S = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ls + u * 64);
                volatile tt_l1_ptr uint16_t* D = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ld + u * 2048);
                for (uint32_t i = 0; i < 16; ++i) {
                    D[i * 17] = S[i];             // face 0: columns 0..15
                    D[768 + i * 17] = S[16 + i];  // face 3: columns 16..31
                }
            }
            cb_push_back(cb_w, UM * 4);
            cb_push_back(cb_dg, UM);
            cb_push_back(cb_s, UM);
            cb_wait_front(cb_s, UM);  // c_3 is this kernel's own ring
            cb_pop_front(cb_s, UM);
        };
        prep(0);
        for (uint32_t g = 0; g < NG; ++g, ++gi) {
            if (g + 1 < NG) {
                prep(g + 1);
            }
            // exchange group g: the slot is free here, then (once free everywhere) this core's share to every core
            cb_reserve_back(cb_wx, SLOT);
            const uint32_t slot = get_write_ptr(cb_wx);
#ifdef T6_NOX
            cb_wait_front(cb_fold, UM * 4);
            cb_pop_front(cb_fold, UM * 4);
            cb_push_back(cb_wx, SLOT);
            continue;
#endif
            inc_all(free);
            cb_wait_front(cb_fold, UM * 4);
            noc_semaphore_wait_min(free_p, R * (gi + 1));
            if (q1 > q0) {
                // unit row q's first tile (t6_compute.cpp QOFF: the last column block may be narrower)
                auto qoff = [&](uint32_t q) {
                    const uint32_t c1 = q / J * CP, w = nb - c1 < CP ? nb - c1 : CP;
                    if (c1 >= nb) {
                        return nb * KT;
                    }
                    return q / J * KT * CP + q % J * 4 * w;
                };
                const uint32_t src = get_read_ptr(cb_fold), dst = slot + qoff(q0) * WTB;
                const uint32_t bytes = (qoff(q1) - qoff(q0)) * WTB;
                for (uint32_t q = 0; q < nrect; ++q) {
                    noc_async_write_multicast(src, mcast(q, dst), bytes, get_arg_val<uint32_t>(RECT + 5 + 5 * q));
                }
                noc_async_write(src, get_noc_addr(me_x, me_y, dst), bytes);
                noc_async_write_barrier();
            }
            inc_all(full);
            cb_pop_front(cb_fold, UM * 4);
            noc_semaphore_wait_min(full_p, R * (gi + 1));
            cb_push_back(cb_wx, SLOT);
        }
        noc_async_atomic_barrier();
        // RESIDUAL: the block's residual tiles into c_4, [row][column] (the compute adds them to the accumulators)
        const uint32_t nout = m_count * (nb / OUT_DIV);
#ifdef RESIDUAL
        for (uint32_t k = 0; k < nout; ++k) {
            const uint32_t r = k / (nb / OUT_DIV), c = k % (nb / OUT_DIV);
            cb_reserve_back(cb_res, 1);
            noc_async_read(
                res.get_noc_addr((m_first + r) * Nt_out + (n_first + c0) / OUT_DIV + c), get_write_ptr(cb_res), obytes);
            noc_async_read_barrier();
            cb_push_back(cb_res, 1);
        }
#endif
        // outputs [row][column]
        for (uint32_t k = 0; k < nout; ++k) {
            const uint32_t r = k / (nb / OUT_DIV), c = k % (nb / OUT_DIV);
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), dest(m_first + r, (n_first + c0) / OUT_DIV + c), obytes);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
