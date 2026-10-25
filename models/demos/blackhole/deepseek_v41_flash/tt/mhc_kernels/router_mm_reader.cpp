// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DSV4.1 router matmul, reader (NCRISC).  Core (n, s) owns output column tile n and the K slice s of S (KS tiles of 32
// rows of the 5120-wide hidden dim): it reads the x tiles [s*KS, (s+1)*KS) (only the first XB bytes of each bf16 tile
// page: 1 KB = rows 0..15, enough for T <= 16) and the KS weight tiles of its slice (pre-arranged: page j * NC + core,
// so that step j of all cores reads consecutive pages), in blocks of B tiles so that compute starts after the first
// block.

#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

static inline uint32_t fkey(uint32_t b) { return b ^ ((uint32_t)((int32_t)b >> 31) | 0x80000000u); }

void kernel_main() {
    DeviceZoneScopedN("RR_ALL");
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t KS = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t XB = get_compile_time_arg_val(4);
    constexpr uint32_t NC = get_compile_time_arg_val(5);  // cores per K step
    constexpr uint32_t S = NC / 12;
    constexpr uint32_t T = get_compile_time_arg_val(6);
    constexpr uint32_t cb_g = get_compile_time_arg_val(7);
    constexpr uint32_t SEM_FA = get_compile_time_arg_val(8);  // writer -> reader: row gathered (counter)
    constexpr uint32_t SEM_FB = get_compile_time_arg_val(9);  // reader -> writer: second half scanned (counter)
    constexpr uint32_t K = get_compile_time_arg_val(10);
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048;

    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t s = get_arg_val<uint32_t>(1);
    const uint32_t kx0 = s * KS;     // first x tile page of the slice
    const uint32_t kw0 = n * S + s;  // weight core index
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto w_acc = TensorAccessor(w_args, w_addr, TILE);
    experimental::CB cbx(cb_x), cbw(cb_w);
    cbx.reserve_back(KS);
    cbw.reserve_back(KS);
    for (uint32_t b = 0; b < KS / B; ++b) {
        for (uint32_t j = b * B; j < (b + 1) * B; ++j) {
            noc.async_read(
                w_acc, cbw, TILE, {.page_id = j * NC + kw0, .offset_bytes = 0}, {.offset_bytes = (j - b * B) * TILE});
            noc.async_read(
                x_acc, cbx, XB, {.page_id = kx0 + j, .offset_bytes = 0}, {.offset_bytes = (j - b * B) * TILE});
        }
        noc.async_read_barrier();
        cbw.push_back(B);
        cbx.push_back(B);
    }

    // ---- row owner assist: scan the second half (ranks 192..383) of every row this core owns while the writer kernel
    // scans the first half; the K best (key, index) go to L1 for the writer to merge (ascending-index order is kept:
    // lowest index wins ties, this half holds the higher indices)
    uint32_t nrows = 0;
    for (uint32_t t = n + 12 * s; t < T; t += 12 * S) {
        ++nrows;
    }
    if (nrows == 0) {
        return;
    }
    const uint32_t g_rank = get_write_ptr(cb_g);
    volatile tt_l1_ptr uint32_t* rk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_rank);
    volatile tt_l1_ptr uint32_t* cand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_rank + 3072);
    volatile tt_l1_ptr uint32_t* fa = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_FA));
    volatile tt_l1_ptr uint32_t* fb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_FB));
    for (uint32_t r = 0; r < nrows; ++r) {
        noc_semaphore_wait_min(fa, r + 1);
        uint32_t bk[K];
        uint32_t bi[K];
        for (uint32_t j = 0; j < K; ++j) {
            bk[j] = 0;
            bi[j] = 0;
        }
        for (uint32_t c0 = 192; c0 < 384; c0 += 4) {
            const uint32_t k0 = fkey(rk[c0]), k1 = fkey(rk[c0 + 1]), k2 = fkey(rk[c0 + 2]), k3 = fkey(rk[c0 + 3]);
            const uint32_t thr = bk[K - 1];
            if (__builtin_expect(k0 > thr || k1 > thr || k2 > thr || k3 > thr, 0)) {
                const uint32_t kk[4] = {k0, k1, k2, k3};
                for (uint32_t q = 0; q < 4; ++q) {
                    const uint32_t key = kk[q];
                    if (key > bk[K - 1]) {
                        uint32_t p = K - 1;
                        while (p > 0 && key > bk[p - 1]) {
                            bk[p] = bk[p - 1];
                            bi[p] = bi[p - 1];
                            --p;
                        }
                        bk[p] = key;
                        bi[p] = c0 + q;
                    }
                }
            }
        }
        for (uint32_t j = 0; j < K; ++j) {
            cand[j] = bk[j];
            cand[8 + j] = bi[j];
        }
        *fb = r + 1;
    }
}
