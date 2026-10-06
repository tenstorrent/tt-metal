// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DSV4.1 router selection, one core per token row t: exact fp32 top-K of rank[t, :E] (lowest index wins ties), then
// w_i = bf16( score[t, idx_i] / (sum_j score[t, idx_j] + eps) * scale ). rank/score are fp32 TILE tensors [1,1,T,E];
// outputs are row-major [T,1,1,K] pages: uint16 indices and bf16 weights.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

static inline uint32_t fkey(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
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

void kernel_main() {
    constexpr uint32_t cb_rank = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);  // E / 32
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(5);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(6);
    constexpr auto r_args = TensorAccessorArgs<7>();
    constexpr auto s_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto wi_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto ii_args = TensorAccessorArgs<wi_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t t_first = get_arg_val<uint32_t>(0);
    const uint32_t t_stride = get_arg_val<uint32_t>(1);  // rows per core: t_first, t_first + t_stride, ... (count rows)
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t r_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t i_addr = get_common_arg_val<uint32_t>(3);

    Noc noc;
    const auto r_acc = TensorAccessor(r_args, r_addr, TILE);
    const auto s_acc = TensorAccessor(s_args, s_addr, TILE);
    const auto w_acc = TensorAccessor(wi_args, w_addr, K * 2);
    const auto i_acc = TensorAccessor(ii_args, i_addr, K * 2);
    experimental::CB cbr(cb_rank), cbs(cb_sc), cbo(cb_out);
    cbr.reserve_back(1);
    cbs.reserve_back(1);
    cbo.reserve_back(1);

    for (uint32_t it = 0; it < count; ++it) {
        const uint32_t t = t_first + it * t_stride;
        const uint32_t tr = t & 31, pbase = (t >> 5) * NT;  // row inside its tile row, first tile page of the tile row
        const uint32_t roff = ((tr >> 4) << 11) + ((tr & 15) << 6);
        for (uint32_t n = 0; n < NT; ++n) {
            noc.async_read(r_acc, cbr, 64, {.page_id = pbase + n, .offset_bytes = roff}, {.offset_bytes = n * 128});
            noc.async_read(
                r_acc, cbr, 64, {.page_id = pbase + n, .offset_bytes = roff + 1024}, {.offset_bytes = n * 128 + 64});
        }
        noc.async_read_barrier();

        volatile tt_l1_ptr uint32_t* rk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbr.get_write_ptr());
        uint32_t bk[K];
        uint32_t bi[K];
        for (uint32_t j = 0; j < K; ++j) {
            bk[j] = 0;
            bi[j] = 0;
        }
        // strictly-greater insertion in ascending index order => lowest index wins ties
        for (uint32_t c = 0; c < NT * 32; ++c) {
            const uint32_t key = fkey(rk[c]);
            if (key > bk[K - 1]) {
                uint32_t p = K - 1;
                while (p > 0 && key > bk[p - 1]) {
                    bk[p] = bk[p - 1];
                    bi[p] = bi[p - 1];
                    --p;
                }
                bk[p] = key;
                bi[p] = c;
            }
        }
        // scores at the chosen experts
        for (uint32_t j = 0; j < K; ++j) {
            const uint32_t c = bi[j];
            const uint32_t n = c >> 5, h = (c >> 4) & 1;
            noc.async_read(
                s_acc, cbs, 64, {.page_id = pbase + n, .offset_bytes = roff + (h << 10)}, {.offset_bytes = j * 64});
        }
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_write_ptr());
        float s[K];
        float den = 0.0f;
        for (uint32_t j = 0; j < K; ++j) {
            s[j] = as_f(sc[j * 16 + (bi[j] & 15)]);
            den += s[j];
        }
        const float inv_den = as_f(EPS_BITS) + den;
        const float scale = as_f(SCALE_BITS);
        volatile tt_l1_ptr uint16_t* ow = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbo.get_write_ptr());
        volatile tt_l1_ptr uint16_t* oi = ow + 32;  // +64 B
        for (uint32_t j = 0; j < K; ++j) {
            const uint32_t b = as_u((s[j] / inv_den) * scale);
            ow[j] = (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
            oi[j] = (uint16_t)bi[j];
        }
        noc.async_write(cbo, w_acc, K * 2, {.offset_bytes = 0}, {.page_id = t, .offset_bytes = 0});
        noc.async_write(cbo, i_acc, K * 2, {.offset_bytes = 64}, {.page_id = t, .offset_bytes = 0});
        noc.async_write_barrier();
    }
}
