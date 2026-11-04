// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// 2D exact ternary prefill linear (linear2.py), weight reader and output writer (NCRISC). This core's output tile
// columns n_first .. n_first + n_count - 1 go in blocks of up to NBLK columns. Per block and kb: each column's 4
// ternary tiles of the kb (1 KB, adjacent in its decode record) into c_1, laid out [column][k], and the block's
// broadcast-scale tiles (scales.py, one per 8 columns) into c_3. After the block: its output tiles, [row][column].
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t PPR4K = get_common_arg_val<uint32_t>(1);
    const uint32_t sb_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t Nt_out = get_common_arg_val<uint32_t>(4);
    const uint32_t KB = get_common_arg_val<uint32_t>(5);
    const uint32_t n_first = get_arg_val<uint32_t>(0), n_count = get_arg_val<uint32_t>(1);
    const uint32_t m_first = get_arg_val<uint32_t>(2), m_count = get_arg_val<uint32_t>(3);
    constexpr uint32_t NBLK = get_compile_time_arg_val(0);
    // fixed CB slots per kb (2 per ring), so a block of fewer columns never wraps a ring mid-slot
    constexpr uint32_t WSLOT = NBLK * 4, SSLOT = (NBLK + 7) / 8 + 1;
    constexpr auto w_args = TensorAccessorArgs<1>();
    constexpr auto s_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_w = tt::CBIndex::c_1;
    constexpr uint32_t cb_s = tt::CBIndex::c_3;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
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
    for (uint32_t c0 = 0; c0 < n_count; c0 += NBLK) {
        const uint32_t nb = n_count - c0 < NBLK ? n_count - c0 : NBLK;
        const uint32_t u0 = (n_first + c0) / 8;  // the block's broadcast-scale tiles: units u0 .. of its columns
        const uint32_t units = (n_first + c0 + nb - 1) / 8 - u0 + 1;
        for (uint32_t kb = 0; kb < KB; ++kb) {
            const uint32_t off = (kb / 32) * GROUP + 2048 + (kb % 32) * 1024;
            cb_reserve_back(cb_w, WSLOT);
            const uint32_t lw = get_write_ptr(cb_w);
            cb_reserve_back(cb_s, SSLOT);
            const uint32_t ls = get_write_ptr(cb_s);
#ifndef T2_NOWREAD
            for (uint32_t c = 0; c < nb; ++c) {
                const uint32_t n = n_first + c0 + c;
                noc_async_read(w.get_noc_addr(n * PPR4K + off / 4096, off % 4096), lw + c * 1024, 1024);
            }
            for (uint32_t u = 0; u < units; ++u) {
                noc_async_read(sb.get_noc_addr((u0 + u) * KB + kb), ls + u * 2048, 2048);
            }
#endif
            noc_async_read_barrier();
            cb_push_back(cb_w, WSLOT);
            cb_push_back(cb_s, SSLOT);
        }
        for (uint32_t r = 0; r < m_count; ++r) {
            for (uint32_t c = 0; c < nb / OUT_DIV; ++c) {
                cb_wait_front(cb_out, 1);
                noc_async_write(
                    get_read_ptr(cb_out),
                    out.get_noc_addr((m_first + r) * Nt_out + (n_first + c0) / OUT_DIV + c),
                    obytes);
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
        }
    }
}
