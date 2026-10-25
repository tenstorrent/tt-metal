// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DSV4.1 fused router, writer (BRISC).  Three roles on a NT x S core grid (core (n, s)):
//  * K-slice core (s > 0): sends its fp32 partial tile into slot s-1 of the reducer (n, 0) and bumps its SEM_PART.
//  * reducer (s == 0): loads its bias tile, waits for the S-1 partials, lets compute produce the rank / score tiles of
//    column tile n (kept in L1), then bumps SEM_ROW of the owner core of every row t < T.
//  * owner of row t (core (t % NT, (t / NT) % S)): after NT bumps per row it pulls the 4 x 64 B pieces of row t of
//  every
//    reducer's rank / score tile into L1 (rank[384], score[384]), selects the exact fp32 top-K (lowest index wins
//    ties), and writes w_i = bf16(score_i / (sum score + eps) * scale) and the uint16 ids to page t of the RM outputs.
// The selection arithmetic is the one of router_select.cpp.

#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

static inline uint32_t fkey(uint32_t b) { return b ^ ((uint32_t)((int32_t)b >> 31) | 0x80000000u); }
static inline float as_f(uint32_t b) {
    union {
        uint32_t u;
        float f;
    } c;
    c.u = b;
    return c.f;
}
static inline uint32_t as_u(float f) {
    union {
        uint32_t u;
        float f;
    } c;
    c.f = f;
    return c.u;
}
#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))

void kernel_main() {
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_red = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rank = get_compile_time_arg_val(3);
    constexpr uint32_t cb_score = get_compile_time_arg_val(4);
    constexpr uint32_t cb_g = get_compile_time_arg_val(5);
    constexpr uint32_t cb_o = get_compile_time_arg_val(6);
    constexpr uint32_t S = get_compile_time_arg_val(7);
    constexpr uint32_t SEM_PART = get_compile_time_arg_val(8);
    constexpr uint32_t SEM_ROW = get_compile_time_arg_val(9);
    constexpr uint32_t T = get_compile_time_arg_val(10);
    constexpr uint32_t K = get_compile_time_arg_val(11);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(12);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(13);
    constexpr uint32_t DUMP = get_compile_time_arg_val(14);
    constexpr uint32_t OPAGE = get_compile_time_arg_val(15);  // page bytes of the weights / ids outputs
    constexpr uint32_t GX =
        get_compile_time_arg_val(16);  // core grid: core (n, s) = logical (n % GX, (12 / GX) * s + n / GX)
    constexpr uint32_t NT = 12;
    constexpr uint32_t SEM_FA = get_compile_time_arg_val(17);
    constexpr uint32_t SEM_FB = get_compile_time_arg_val(18);
    constexpr auto b_args = TensorAccessorArgs<19>();
    constexpr auto w_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto i_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t s = get_arg_val<uint32_t>(1);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t i_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t r_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(4);
    // NOC coordinates of logical column n (x) and row s (y)
    // NOC coordinates of core (n, s): logical (n % GX, GY * s + n / GX)
    constexpr uint32_t GY = 12 / GX;
#define CX_(n_) get_common_arg_val<uint32_t>(5 + ((n_) % GX))
#define CY_(n_, s_) get_common_arg_val<uint32_t>(5 + GX + (s_) * GY + (n_) / GX)

    DeviceZoneScopedN("RW_ALL");
    Noc noc;
    experimental::CB cbpart(cb_part), cbred(cb_red), cbbias(cb_bias), cbrank(cb_rank), cbscore(cb_score), cbg(cb_g),
        cbo(cb_o);
    const uint32_t red_base = get_write_ptr(cb_red);
    const uint32_t rank_base = get_write_ptr(cb_rank);
    const uint32_t score_base = get_write_ptr(cb_score);
    const uint32_t sem_part = get_semaphore(SEM_PART);
    const uint32_t sem_row = get_semaphore(SEM_ROW);

    if (s != 0) {
        cbpart.wait_front(1);
        noc_async_write(cbpart.get_read_ptr(), get_noc_addr(CX_(n), CY_(n, 0), red_base + (s - 1) * TILE), TILE);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(CX_(n), CY_(n, 0), sem_part), 1);
        noc_async_atomic_barrier();
        cbpart.pop_front(1);
    } else {
        const auto b_acc = TensorAccessor(b_args, b_addr, TILE);
        cbbias.reserve_back(1);
        noc.async_read(b_acc, cbbias, TILE, {.page_id = n, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbbias.push_back(1);
        if constexpr (S > 1) {
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_part);
            noc_semaphore_wait_min(sem, S - 1);
            *sem = 0;
            cbred.reserve_back(S - 1);
            cbred.push_back(S - 1);
        }
        cbscore.wait_front(1);
        cbrank.wait_front(1);
        if constexpr (DUMP) {
            const auto r_acc = TensorAccessor(r_args, r_addr, TILE);
            const auto s_acc = TensorAccessor(s_args, s_addr, TILE);
            noc.async_write(cbrank, r_acc, TILE, {.offset_bytes = 0}, {.page_id = n, .offset_bytes = 0});
            noc.async_write(cbscore, s_acc, TILE, {.offset_bytes = 0}, {.page_id = n, .offset_bytes = 0});
            noc.async_write_barrier();
        }
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t on = t % NT, os = (t / NT) % S;
            noc_semaphore_inc(get_noc_addr(CX_(on), CY_(on, os), sem_row), 1);
        }
        noc_async_atomic_barrier();
    }

    {
        DeviceZoneScopedN("RW_redsent");
    }
    // ---- owner role ----
    uint32_t nrows = 0;
    for (uint32_t t = n + NT * s; t < T; t += NT * S) {
        ++nrows;
    }
    if (nrows == 0) {
        return;
    }
    volatile tt_l1_ptr uint32_t* semr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_row);
    noc_semaphore_wait_min(semr, NT * nrows);
    *semr = 0;
    {
        DeviceZoneScopedN("RW_ownwait");
    }
    const auto w_acc = TensorAccessor(w_args, w_addr, OPAGE);
    const auto i_acc = TensorAccessor(i_args, i_addr, OPAGE);
    cbg.reserve_back(1);
    cbo.reserve_back(1);
    const uint32_t g_rank = cbg.get_write_ptr();
    const uint32_t g_score = g_rank + NT * 128;
    volatile tt_l1_ptr uint32_t* fa = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_FA));
    volatile tt_l1_ptr uint32_t* fb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_FB));
    uint32_t row_idx = 0;
    for (uint32_t t = n + NT * s; t < T; t += NT * S) {
        const uint32_t ro = RO(t & 31);  // T <= 32: one tile row
        for (uint32_t c = 0; c < NT; ++c) {
            const uint32_t x = CX_(c), y = CY_(c, 0);
            noc_async_read(get_noc_addr(x, y, rank_base + ro), g_rank + c * 128, 64);
            noc_async_read(get_noc_addr(x, y, rank_base + ro + 1024), g_rank + c * 128 + 64, 64);
            noc_async_read(get_noc_addr(x, y, score_base + ro), g_score + c * 128, 64);
            noc_async_read(get_noc_addr(x, y, score_base + ro + 1024), g_score + c * 128 + 64, 64);
        }
        noc_async_read_barrier();
        {
            DeviceZoneScopedN("RW_gathered");
        }

        volatile tt_l1_ptr uint32_t* rk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_rank);
        volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_score);
        volatile tt_l1_ptr uint32_t* cand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_rank + 3072);
        *fa = row_idx + 1;  // the reader kernel (other RISC) scans ranks 192..383
        uint32_t bk[K];
        uint32_t bi[K];
        for (uint32_t j = 0; j < K; ++j) {
            bk[j] = 0;
            bi[j] = 0;
        }
        // strictly-greater insertion in ascending index order => lowest index wins ties; groups of 4 are skipped with
        // one comparison against the current K-th key
        for (uint32_t c0 = 0; c0 < 192; c0 += 4) {
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
        noc_semaphore_wait_min(fb, row_idx + 1);
        for (uint32_t q = 0; q < K; ++q) {  // merge the second half's K best (in its list order)
            const uint32_t key = cand[q];
            if (key > bk[K - 1]) {
                uint32_t p = K - 1;
                while (p > 0 && key > bk[p - 1]) {
                    bk[p] = bk[p - 1];
                    bi[p] = bi[p - 1];
                    --p;
                }
                bk[p] = key;
                bi[p] = cand[8 + q];
            }
        }
        ++row_idx;
        {
            DeviceZoneScopedN("RW_scanned");
        }
        float sv[K];
        float den = 0.0f;
        for (uint32_t j = 0; j < K; ++j) {
            sv[j] = as_f(sc[bi[j]]);
            den += sv[j];
        }
        const float inv_den = as_f(EPS_BITS) + den;
        const float scale = as_f(SCALE_BITS);
        volatile tt_l1_ptr uint16_t* ow = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbo.get_write_ptr());
        volatile tt_l1_ptr uint16_t* oi = ow + 32;
        for (uint32_t j = 0; j < K; ++j) {
            const uint32_t b = as_u((sv[j] / inv_den) * scale);
            ow[j] = (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
            oi[j] = (uint16_t)bi[j];
        }
        noc.async_write(cbo, w_acc, K * 2, {.offset_bytes = 0}, {.page_id = t, .offset_bytes = 0});
        noc.async_write(cbo, i_acc, K * 2, {.offset_bytes = 64}, {.page_id = t, .offset_bytes = 0});
        noc.async_write_barrier();
    }
    *fa = 0;
    *fb = 0;
}
