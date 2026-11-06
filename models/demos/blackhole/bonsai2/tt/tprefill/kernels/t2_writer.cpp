// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// 2D exact ternary prefill linear (linear2.py), weight reader and output writer (NCRISC). This core's output tile
// columns n_first .. n_first + n_count - 1 go in blocks of up to NBLK columns. Per block and group of J kbs: each
// column's 4 J ternary tiles of the group (J KB, adjacent in its decode record) into c_1, laid out [column][kb][k], and
// each column's ratio tile (scales.pack_ratio_group) into c_3. After the block: its output tiles, [row][column].
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t PPR4K = get_common_arg_val<uint32_t>(1);
    const uint32_t sb_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t Nt_out = get_common_arg_val<uint32_t>(4);
    const uint32_t KB = get_common_arg_val<uint32_t>(5);
    // split output (linear2.py split=): tile columns n < NA go to out, the rest to out_b; each head-major
    // ([heads, T, HA (HB) tiles]) when HA (HB) > 0, else token-major. Unsplit: NA = Nt_out, HA = 0.
    const uint32_t ob_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t NA = get_common_arg_val<uint32_t>(7);
    const uint32_t HA = get_common_arg_val<uint32_t>(8), HB = get_common_arg_val<uint32_t>(9);
    const uint32_t Mt = get_common_arg_val<uint32_t>(10);
    const uint32_t res_addr = get_common_arg_val<uint32_t>(11);  // RESIDUAL: added to the outputs (same layout)
    const uint32_t n_first = get_arg_val<uint32_t>(0), n_count = get_arg_val<uint32_t>(1);
    const uint32_t m_first = get_arg_val<uint32_t>(2), m_count = get_arg_val<uint32_t>(3);
    // weight multicast down the core column: the column's first core reads each group's weights and ratio tiles and
    // multicasts them to the same CB slots of the column's other active cores (rows with token tiles); handshake as
    // t2_reader.cpp (ready count at the sender, valid flag at the receivers)
    const uint32_t is_wsender = get_arg_val<uint32_t>(4), nrecv = get_arg_val<uint32_t>(5);
    const uint32_t ws_x = get_arg_val<uint32_t>(6), ws_y = get_arg_val<uint32_t>(7);
    // then the receivers' rectangles: count, then per rectangle x0, y0, x1, y1, destinations (a line of cores, split
    // where the virtual coordinates jump)
    const uint32_t nrect = get_arg_val<uint32_t>(8);
    constexpr uint32_t NBLK = get_compile_time_arg_val(0);
    // fixed CB slots per kb (2 per ring), so a block of fewer columns never wraps a ring mid-slot
    constexpr uint32_t WSLOT = NBLK * 4;
    constexpr uint32_t J = get_compile_time_arg_val(1);  // kbs per ratio group
    constexpr auto w_args = TensorAccessorArgs<2>();
    constexpr auto s_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto ob_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<ob_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_w = tt::CBIndex::c_1;
    constexpr uint32_t cb_s = tt::CBIndex::c_3;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_res = tt::CBIndex::c_4;
    constexpr uint32_t GROUP = 2048 + 32 * 4 * 256;
#ifdef GLU_PAIRS
    constexpr uint32_t OUT_DIV = 2;  // one output tile per (gate, up) column pair
#else
    constexpr uint32_t OUT_DIV = 1;
#endif
    if (n_count == 0 || m_count == 0) {
        return;
    }
    const auto w = TensorAccessor(w_args, w_addr, 4096);
    const auto sb = TensorAccessor(s_args, sb_addr, 2048);
    const uint32_t obytes = get_tile_size(cb_out);
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
    const uint32_t wready = get_semaphore(2);
    const uint32_t wvalid = get_semaphore(3);
    volatile tt_l1_ptr uint32_t* wready_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wready);
    volatile tt_l1_ptr uint32_t* wvalid_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wvalid);
    const uint64_t wready_noc = get_noc_addr(ws_x, ws_y, wready);
    auto mcast = [&](uint32_t q, uint32_t l1) {  // NOC 1 runs the other way: its rectangle starts at the far corner
        const uint32_t x0 = get_arg_val<uint32_t>(9 + 5 * q), y0 = get_arg_val<uint32_t>(10 + 5 * q);
        const uint32_t x1 = get_arg_val<uint32_t>(11 + 5 * q), y1 = get_arg_val<uint32_t>(12 + 5 * q);
        return noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, l1) : get_noc_multicast_addr(x1, y1, x0, y0, l1);
    };
    for (uint32_t c0 = 0; c0 < n_count; c0 += NBLK) {
        const uint32_t nb = n_count - c0 < NBLK ? n_count - c0 : NBLK;
        for (uint32_t kb0 = 0; kb0 < KB; kb0 += J) {
            // per column the group's J kbs of ternary tiles are adjacent in its record (groups never straddle a
            // 32-kb record group: 32 % J == 0): one read per column, split where it crosses a 4-KB page (pages
            // are interleaved over the DRAM banks)
            const uint32_t off = (kb0 / 32) * GROUP + 2048 + (kb0 % 32) * 1024;
            const uint32_t p0 = off / 4096, o0 = off % 4096;
            const uint32_t first = 4096 - o0 < J * 1024 ? 4096 - o0 : J * 1024;
            cb_reserve_back(cb_w, J * WSLOT);
            const uint32_t lw = get_write_ptr(cb_w);
            cb_reserve_back(cb_s, NBLK);
            const uint32_t ls = get_write_ptr(cb_s);
            if (is_wsender) {
#ifndef T2_NOWREAD
                for (uint32_t c = 0; c < nb; ++c) {
                    const uint32_t n = n_first + c0 + c;
                    const uint32_t dst = lw + c * J * 1024;
                    noc_async_read(w.get_noc_addr(n * PPR4K + p0, o0), dst, first);
                    if (first < J * 1024) {
                        noc_async_read(w.get_noc_addr(n * PPR4K + p0 + 1, 0), dst + first, J * 1024 - first);
                    }
                    noc_async_read(sb.get_noc_addr(n * (KB / J) + kb0 / J), ls + c * 2048, 2048);
                }
                noc_async_read_barrier();
#endif
                if (nrecv) {
                    noc_semaphore_wait(wready_p, nrecv);
                    noc_semaphore_set(wready_p, 0);
                    *wvalid_p = 1;
                    for (uint32_t q = 0; q < nrect; ++q) {
                        const uint32_t nd = get_arg_val<uint32_t>(13 + 5 * q);
                        noc_async_write_multicast(lw, mcast(q, lw), nb * J * 1024, nd);
                        noc_async_write_multicast(ls, mcast(q, ls), nb * 2048, nd);
                        noc_semaphore_set_multicast(wvalid, mcast(q, wvalid), nd);
                    }
                    noc_async_write_barrier();
                }
            } else {
                noc_semaphore_set(wvalid_p, 0);
                noc_semaphore_inc(wready_noc, 1);
                noc_semaphore_wait(wvalid_p, 1);
            }
            cb_push_back(cb_w, J * WSLOT);
            cb_push_back(cb_s, NBLK);
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
            noc_async_writes_flushed();  // the slot is free once the tile has left L1 (no round trip per tile)
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
