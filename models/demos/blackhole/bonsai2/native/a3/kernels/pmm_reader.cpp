// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Short-prefill ternary matmul (A3 at M>1), reader. y [M, N] = x' [M, K] @ (codes * scales), M <= 128.
// Per K super-block G (32 K-tiles = 8 scale groups): this block of x' (MT x 32 tiles) once per core, then for
// each of this core's weight columns its 32 code tiles (contiguous K-stick in its DRAM bank) and the one
// fp32 scale_rows tile holding the block's 8 scale rows.
// Runtime args: x_addr, w_addr, s_addr, c0, nc.

#include "api/dataflow/dataflow_api.h"
#include "scale_loc.h"
#include "api/debug/dprint.h"

// Bounded semaphore wait (debug safety): never hangs the chip; on timeout the caller reports and continues.
inline bool bounded_wait(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    for (uint32_t i = 0; i < (1u << 24); ++i) {
        invalidate_l1_cache();
        if (*p == v) {
            return true;
        }
    }
    return false;
}

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_arg_val<uint32_t>(2);
    const uint32_t c0 = get_arg_val<uint32_t>(3);
    const uint32_t nc = get_arg_val<uint32_t>(4);
    constexpr uint32_t ng = get_compile_time_arg_val(0);
    constexpr uint32_t MT = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t cpb = get_compile_time_arg_val(3);
    constexpr uint32_t wtb = get_compile_time_arg_val(4);
    constexpr uint32_t s_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t xtb = get_compile_time_arg_val(6);
    constexpr uint32_t cb_x = 0, cb_c = 1, cb_s = 2;
    const InterleavedAddrGen<true> xg{.bank_base_address = x_addr, .page_size = xtb};
    constexpr uint32_t pk = get_compile_time_arg_val(12), tail_t = get_compile_time_arg_val(13);
    constexpr uint32_t s_full = get_compile_time_arg_val(14), rem = get_compile_time_arg_val(15);
    constexpr uint32_t s_tb = get_compile_time_arg_val(16);  // scale tile bytes: 4096 fp32, 2048 fp16
    constexpr uint32_t mc = get_compile_time_arg_val(7);  // 1: x' arrives by multicast from the sender core
    constexpr uint32_t snd_x = get_compile_time_arg_val(8), snd_y = get_compile_time_arg_val(9);
    constexpr uint32_t sem_ready_id = get_compile_time_arg_val(10), sem_valid_id = get_compile_time_arg_val(11);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_valid_id));
    const uint64_t ready_noc = get_noc_addr(snd_x, snd_y, get_semaphore(sem_ready_id));
    auto read_codes = [&](uint32_t G) {
        for (uint32_t c = c0; c < c0 + nc; ++c) {
            const uint32_t bank = c / cpb, j = c % cpb;
            cb_reserve_back(cb_s, 1);
            const ScaleLoc loc = scale_loc<pk, s_tiles, tail_t, s_full, rem>(j, G);
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, s_addr + loc.tile * s_tb), get_write_ptr(cb_s), s_tb);
            cb_reserve_back(cb_c, 32);
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, w_addr + (j * kt + G * 32) * wtb),
                           get_write_ptr(cb_c), 32 * wtb);
            noc_async_read_barrier();
            cb_push_back(cb_s, 1);
            cb_push_back(cb_c, 32);
        }
    };
    for (uint32_t G = 0; G < ng; ++G) {
        cb_reserve_back(cb_x, MT * 32);
        if constexpr (mc) {
            // Signal ready, fetch this block's code/scale tiles while the sender gathers everyone, then wait.
            // cb_c/cb_s hold 2 x nc blocks, so this never blocks on compute (which waits on x first).
            noc_semaphore_set(valid, 0);
            noc_semaphore_inc(ready_noc, 1);
            read_codes(G);
            if (!bounded_wait(valid, 1)) {
                DPRINT("PMM recv valid timeout G={} sender=({},{})\n", G, snd_x, snd_y);
            }
            cb_push_back(cb_x, MT * 32);
        } else {
            uint32_t l1 = get_write_ptr(cb_x);
            for (uint32_t m = 0; m < MT; ++m) {
                for (uint32_t t = 0; t < 32; ++t, l1 += xtb) {
                    noc_async_read(get_noc_addr(m * kt + G * 32 + t, xg), l1, xtb);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, MT * 32);
            read_codes(G);
        }
    }
}
