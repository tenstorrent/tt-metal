// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// 2D exact ternary prefill linear (linear2.py), activation reader (BRISC, NOC 0). Each core row is a row group: it
// computes token row tiles m_first .. m_first + m_count - 1. Per column block and per kb its sender (the row's first
// core) reads the group's x' tiles (m_count rows x the kb's 4 K tiles, laid out [row][k]) from DRAM and multicasts them
// to the same c_0 slot on the row's other cores (one rectangle per run of adjacent virtual columns). Handshake per
// chunk: each receiver reserves its slot, zeroes its `valid` semaphore and increments the sender's `ready`; the sender
// waits for all of them, multicasts the chunk, then valid = 1.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t Kt = get_common_arg_val<uint32_t>(1);
    const uint32_t m_max = get_common_arg_val<uint32_t>(2);
    const uint32_t is_sender = get_arg_val<uint32_t>(0);
    const uint32_t s_x = get_arg_val<uint32_t>(1), s_y = get_arg_val<uint32_t>(2);
    const uint32_t m_first = get_arg_val<uint32_t>(3), m_count = get_arg_val<uint32_t>(4);
    const uint32_t nblk = get_arg_val<uint32_t>(5);
    const uint32_t ncores = get_arg_val<uint32_t>(6);  // cores of the row
    const uint32_t nrect = get_arg_val<uint32_t>(7);   // then per rectangle x0, y0, x1, y1, destinations
    constexpr uint32_t KPC = get_compile_time_arg_val(0);  // kbs per multicast chunk
    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    const uint32_t TB = get_tile_size(cb_x);  // bf16 or bfp8 x' tiles
    const uint32_t KB = Kt / 4;
    const uint32_t CHT = m_count * 4 * KPC;  // tiles per chunk, laid out [row][k]
    const uint32_t XSLOT = m_max * 4 * KPC;  // fixed CB slot per chunk (every row group's slots line up)

    const auto xs = TensorAccessor(x_args, x_addr, TB);
    const uint32_t ready = get_semaphore(0);
    const uint32_t valid = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* ready_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
    volatile tt_l1_ptr uint32_t* valid_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid);
    const uint64_t ready_noc = get_noc_addr(s_x, s_y, ready);
    for (uint32_t b = 0; b < nblk; ++b) {
        for (uint32_t kb = 0; kb < KB; kb += KPC) {
            cb_reserve_back(cb_x, XSLOT);
            const uint32_t dst = get_write_ptr(cb_x);
            if (is_sender) {
#ifndef T2_NOXREAD
                for (uint32_t r = 0; r < m_count; ++r) {
                    for (uint32_t i = 0; i < 4 * KPC; ++i) {
                        noc_async_read(
                            xs.get_noc_addr((m_first + r) * Kt + kb * 4 + i), dst + (r * 4 * KPC + i) * TB, TB);
                    }
                }
                noc_async_read_barrier();
#endif
#ifndef T2_NOXMC
                if (ncores > 1) {
                    noc_semaphore_wait(ready_p, ncores - 1);
                    noc_semaphore_set(ready_p, 0);
                    *valid_p = 1;
                    for (uint32_t q = 0; q < nrect; ++q) {
                        const uint32_t x0 = get_arg_val<uint32_t>(8 + 5 * q), y0 = get_arg_val<uint32_t>(9 + 5 * q);
                        const uint32_t x1 = get_arg_val<uint32_t>(10 + 5 * q), y1 = get_arg_val<uint32_t>(11 + 5 * q);
                        const uint32_t nd = get_arg_val<uint32_t>(12 + 5 * q);
                        if (nd == 0) {
                            continue;
                        }
                        const uint64_t mc_d = noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, dst)
                                                             : get_noc_multicast_addr(x1, y1, x0, y0, dst);
                        const uint64_t mc_v = noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, valid)
                                                             : get_noc_multicast_addr(x1, y1, x0, y0, valid);
                        noc_async_write_multicast(dst, mc_d, CHT * TB, nd);
                        noc_semaphore_set_multicast(valid, mc_v, nd);
                    }
                    noc_async_write_barrier();
                }
#endif
            } else {
#ifndef T2_NOXMC
                noc_semaphore_set(valid_p, 0);
                noc_semaphore_inc(ready_noc, 1);
                noc_semaphore_wait(valid_p, 1);
#endif
            }
            cb_push_back(cb_x, XSLOT);
        }
    }
    noc_async_atomic_barrier();
}
