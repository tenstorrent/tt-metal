// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// In-place row scatter into the paged KV pool (prefill / spec-decode multi-row writes):  pool[ids[i]] = src[i]  for i
// in [0, N).
//   src   bf16 [N, 512] ROW_MAJOR (page = 1 row = 1024 B)
//   ids   uint32 ROW_MAJOR [1, N] (one page): physical pool row of every source row; 0xFFFFFFFF = skip this row
//   pool  bf16 or fp8_e4m3 (POOL_FP8: rows converted here) [1,1,R,512] ROW_MAJOR
// Rows i with i % NCORES == core are handled by core ``core`` (runtime arg 0), 8 rows in flight.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// bf16 -> fp8 e4m3fn (sign, 4-bit exponent bias 7, 3-bit mantissa; round to nearest even, saturating at 448, no inf)
static inline uint32_t bf16_to_e4m3(uint32_t b) {
    const uint32_t sign = (b >> 8) & 0x80;
    const uint32_t e = (b >> 7) & 0xFF, m = b & 0x7F;
    if (e == 0xFF) {
        return sign | ((m != 0) ? 0x7F : 0x7E);  // NaN stays NaN, inf saturates
    }
    if (e == 0) {
        return sign;
    }
    const int32_t ef = (int32_t)e - 127 + 7;
    if (ef >= 1) {
        const uint32_t mant = m >> 4, rem = m & 0xF;
        uint32_t v = ((uint32_t)ef << 3) | mant;
        if (rem > 8 || (rem == 8 && (mant & 1))) {
            v += 1;
        }
        return sign | (v > 0x7E ? 0x7E : v);
    }
    const int32_t sh = 125 - (int32_t)e;  // subnormal: units of 2^-9
    if (sh > 9) {
        return sign;
    }
    const uint32_t M = 128 | m;
    uint32_t v = M >> sh;
    const uint32_t rem = M & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (v & 1))) {
        v += 1;
    }
    return sign | v;
}

// 512 bf16 (uint16 pairs in 256 words) -> 512 e4m3 bytes (128 words), in place at ``w``
static inline void row_to_fp8(volatile tt_l1_ptr uint32_t* w) {
    for (uint32_t i = 0; i < 128; ++i) {
        const uint32_t a = w[2 * i], b = w[2 * i + 1];
        w[i] = bf16_to_e4m3(a & 0xFFFF) | (bf16_to_e4m3(a >> 16) << 8) | (bf16_to_e4m3(b & 0xFFFF) << 16) |
               (bf16_to_e4m3(b >> 16) << 24);
    }
}

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t NCORES = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t POOL_FP8 = get_compile_time_arg_val(3);
    constexpr uint32_t ROW_B = POOL_FP8 ? 512 : 1024;
    constexpr auto src_args = TensorAccessorArgs<4>();
    constexpr auto ids_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto pool_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr uint32_t IDS_B = ((N * 4 + 63) / 64) * 64;
    constexpr uint32_t G = 8;

    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t a_src = get_common_arg_val<uint32_t>(0), a_ids = get_common_arg_val<uint32_t>(1),
                   a_pool = get_common_arg_val<uint32_t>(2);
    const uint32_t base_off =
        get_common_arg_val<uint32_t>(3);  // added to every (non-skipped) row id: one ids tensor serves all rings
    Noc noc;
    const auto src_acc = TensorAccessor(src_args, a_src, 1024);
    const auto ids_acc = TensorAccessor(ids_args, a_ids, N * 4);
    const auto pool_acc = TensorAccessor(pool_args, a_pool, ROW_B);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    noc.async_read(ids_acc, cb, N * 4, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    for (uint32_t i0 = core; i0 < N; i0 += NCORES * G) {
        uint32_t cnt = 0;
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t i = i0 + g * NCORES;
            if (i < N) {
                noc.async_read(
                    src_acc, cb, 1024, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = IDS_B + g * 1024});
                ++cnt;
            }
        }
        noc.async_read_barrier();
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t i = i0 + g * NCORES;
            if (i < N && ids[i] != 0xFFFFFFFFu) {
                if constexpr (POOL_FP8) {
                    row_to_fp8(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + IDS_B + g * 1024));
                }
                noc.async_write(
                    cb,
                    pool_acc,
                    ROW_B,
                    {.offset_bytes = IDS_B + g * 1024},
                    {.page_id = ids[i] + base_off, .offset_bytes = 0});
            }
        }
        noc.async_write_barrier();
    }
}
