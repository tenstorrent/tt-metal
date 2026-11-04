// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Exact ternary prefill linear, reader (BRISC, NOC 0).
// 1. This core's weights (output tile columns n_first .. n_first + n_count - 1), read out of the decode records
//    (decode_linear3.pack_records_inline: per column PPR4K 4-KB pages; per group of 32 kbs a 2-KB fp16 scale tile, then
//    the kbs' 256-B ternary tiles): the ternary tiles into c_1, laid out [batch][k][column], the scale tiles into c_3.
// 2. The rotated activation x' [T, K] (bf16 tiles), row tile by row tile in chunks of CH tiles, into c_0 on every core:
//    the sender core reads a chunk from DRAM and multicasts it to the same c_0 address on all other cores. Handshake
//    per chunk: each receiver reserves its c_0 slot, zeroes its `valid` semaphore and increments the sender's `ready`;
//    the sender waits for all of them, multicasts the chunk, then multicasts valid = 1.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t PPR4K = get_common_arg_val<uint32_t>(2);
    const uint32_t Mt = get_common_arg_val<uint32_t>(3);
    const uint32_t s_x = get_common_arg_val<uint32_t>(4);  // sender, virtual coordinates
    const uint32_t s_y = get_common_arg_val<uint32_t>(5);
    const uint32_t ncores = get_common_arg_val<uint32_t>(6);
    // the cores as rectangles of adjacent virtual columns (a rectangle across the non-Tensix columns hangs the
    // multicast): count, then per rectangle x0, y0, x1, y1, destinations other than the sender
    const uint32_t nrect = get_common_arg_val<uint32_t>(7);
    const uint32_t n_first = get_arg_val<uint32_t>(0);
    const uint32_t n_count = get_arg_val<uint32_t>(1);
    const uint32_t is_sender = get_arg_val<uint32_t>(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t CH = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t NB = get_compile_time_arg_val(3);  // columns per compute batch
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_sc = tt::CBIndex::c_3;
    constexpr uint32_t WB = 256;                    // ternary tile
    constexpr uint32_t GROUP = 2048 + 32 * 4 * WB;  // a record group: scale tile, then 32 kbs
    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_w = tt::CBIndex::c_1;
    constexpr uint32_t TB = 2048;

    if (n_count) {
        const auto w = TensorAccessor(w_args, w_addr, 4096);
        auto at = [&](uint32_t n, uint32_t byte) {  // byte offset into column n's record
            return w.get_noc_addr(n * PPR4K + byte / 4096, byte % 4096);
        };
        cb_reserve_back(cb_w, n_count * Kt);
        cb_reserve_back(cb_sc, n_count * G);
        const uint32_t l1w = get_write_ptr(cb_w), l1s = get_write_ptr(cb_sc);
        // L1 layout per batch of up to NB columns: [k][column], so a matmul_block's ct_dim in1 tiles are adjacent
        uint32_t p = 0;
        for (uint32_t n0 = 0; n0 < n_count; n0 += NB) {
            const uint32_t nb = n_count - n0 < NB ? n_count - n0 : NB;
            for (uint32_t k = 0; k < Kt; ++k) {
                const uint32_t g = k / 128, kk = k % 128;  // 128 ternary tiles (32 kbs) per group
                for (uint32_t c = 0; c < nb; ++c) {
                    noc_async_read(at(n_first + n0 + c, g * GROUP + 2048 + kk * WB), l1w + p * WB, WB);
                    ++p;
                }
            }
            noc_async_read_barrier();
        }
        for (uint32_t n = 0; n < n_count; ++n) {
            for (uint32_t g = 0; g < G; ++g) {
                noc_async_read(at(n_first + n, g * GROUP), l1s + (n * G + g) * 2048, 2048);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, n_count * Kt);
        cb_push_back(cb_sc, n_count * G);
    }

    const auto xs = TensorAccessor(x_args, x_addr, TB);
    const uint32_t ready = get_semaphore(0);
    const uint32_t valid = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* ready_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
    volatile tt_l1_ptr uint32_t* valid_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid);
    const uint64_t ready_noc = get_noc_addr(s_x, s_y, ready);
    // The sender reads chunk idx + 1 from DRAM (transaction id by parity) while it multicasts chunk idx.
    const uint32_t per_m = Kt / CH, total = Mt * per_m;
    auto issue = [&](uint32_t idx, uint32_t dst) {
        const uint32_t m = idx / per_m, c = idx % per_m;
        noc_async_read_set_trid(idx % 2 + 1);
        for (uint32_t t = 0; t < CH; ++t) {
            noc_async_read(xs.get_noc_addr(m * Kt + c * CH + t), dst + t * TB, TB);
        }
    };
    const uint32_t fifo_limit = get_local_cb_interface(cb_x).fifo_limit;
    const uint32_t fifo_start = fifo_limit - get_local_cb_interface(cb_x).fifo_size;
    if (is_sender && total) {
        cb_reserve_back(cb_x, CH);
        issue(0, get_write_ptr(cb_x));
    }
    for (uint32_t idx = 0; idx < total; ++idx) {
        cb_reserve_back(cb_x, CH);
        const uint32_t dst = get_write_ptr(cb_x);
        if (is_sender) {
            if (idx + 1 < total) {
                cb_reserve_back(cb_x, 2 * CH);
                uint32_t nd = dst + CH * TB;
                if (nd >= fifo_limit) {
                    nd = fifo_start;
                }
                issue(idx + 1, nd);
            }
            noc_async_read_barrier_with_trid(idx % 2 + 1);
            if (ncores > 1) {
                noc_semaphore_wait(ready_p, ncores - 1);
                noc_semaphore_set(ready_p, 0);
                *valid_p = 1;
                for (uint32_t r = 0; r < nrect; ++r) {
                    const uint32_t x0 = get_common_arg_val<uint32_t>(8 + 5 * r);
                    const uint32_t y0 = get_common_arg_val<uint32_t>(9 + 5 * r);
                    const uint32_t x1 = get_common_arg_val<uint32_t>(10 + 5 * r);
                    const uint32_t y1 = get_common_arg_val<uint32_t>(11 + 5 * r);
                    const uint32_t nd = get_common_arg_val<uint32_t>(12 + 5 * r);
                    // NOC 1 runs the other way: its multicast rectangle starts at the bottom-right corner
                    const uint64_t mc_d = noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, dst)
                                                         : get_noc_multicast_addr(x1, y1, x0, y0, dst);
                    const uint64_t mc_v = noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, valid)
                                                         : get_noc_multicast_addr(x1, y1, x0, y0, valid);
                    noc_async_write_multicast(dst, mc_d, CH * TB, nd);
                    noc_semaphore_set_multicast(valid, mc_v, nd);
                }
                noc_async_write_barrier();
            }
        } else {
            noc_semaphore_set(valid_p, 0);
            noc_semaphore_inc(ready_noc, 1);
            noc_semaphore_wait(valid_p, 1);
        }
        cb_push_back(cb_x, CH);
    }
    noc_async_read_set_trid(0);
    noc_async_atomic_barrier();
}
