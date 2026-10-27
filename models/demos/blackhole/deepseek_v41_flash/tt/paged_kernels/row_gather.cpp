// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Row gather out of the paged KV pool (APC snapshots of an fp8 pool, DSV41_FP8_SERVING):  out[i] = bf16(pool[ids[i]])
// for i in [0, N).
//   ids   uint32 ROW_MAJOR [1, N] (one page): physical pool row of every output row; 0xFFFFFFFF = leave out[i]
//   untouched pool  bf16 or fp8_e4m3 (POOL_FP8: rows converted here, EXACTLY: every e4m3 value, subnormals included, is
//   a bf16 value) [1,1,R,512] ROW_MAJOR out   bf16 [N, 512] ROW_MAJOR (page = 1 row = 1024 B)
// Rows i with i % NCORES == core (runtime arg 0) are handled by core ``core``, 8 rows in flight. (The device typecast
// fp8 -> bf16 flushes subnormals to zero: PAGED_NOTES.md open item 2.)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// fp8 e4m3fn byte -> bf16 bits (sign, 4-bit exponent bias 7, 3-bit mantissa; 0x7F / 0xFF = NaN, no inf)
static inline uint32_t e4m3_to_bf16(uint32_t b) {
    const uint32_t sign = (b & 0x80) << 8;
    const uint32_t e = (b >> 3) & 0xF, m = b & 7;
    if (e == 0xF && m == 7) {
        return sign | 0x7FC0;
    }
    if (e == 0) {
        if (m == 0) {
            return sign;
        }
        // subnormal: m * 2^-9 = 1.f * 2^(p - 9), p = index of the leading one of m
        const uint32_t p = (m >= 4) ? 2 : ((m >= 2) ? 1 : 0);
        const uint32_t frac = (m << (3 - p)) & 7;
        return sign | ((p + 118) << 7) | (frac << 4);
    }
    return sign | ((e + 120) << 7) | (m << 4);
}

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t NCORES = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t POOL_FP8 = get_compile_time_arg_val(3);
    constexpr uint32_t ROW_B = POOL_FP8 ? 512 : 1024;
    constexpr auto ids_args = TensorAccessorArgs<4>();
    constexpr auto pool_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<pool_args.next_compile_time_args_offset()>();
    constexpr uint32_t IDS_B = ((N * 4 + 63) / 64) * 64;
    constexpr uint32_t G = 8;
    constexpr uint32_t RAW_OFF = IDS_B;             // G raw rows (ROW_B each)
    constexpr uint32_t OUT_OFF = IDS_B + G * 1024;  // G bf16 rows

    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t a_ids = get_common_arg_val<uint32_t>(0), a_pool = get_common_arg_val<uint32_t>(1),
                   a_out = get_common_arg_val<uint32_t>(2);
    Noc noc;
    const auto ids_acc = TensorAccessor(ids_args, a_ids, N * 4);
    const auto pool_acc = TensorAccessor(pool_args, a_pool, ROW_B);
    const auto out_acc = TensorAccessor(out_args, a_out, 1024);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    noc.async_read(ids_acc, cb, N * 4, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    for (uint32_t i0 = core; i0 < N; i0 += NCORES * G) {
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t i = i0 + g * NCORES;
            if (i < N && ids[i] != 0xFFFFFFFFu) {
                noc.async_read(
                    pool_acc, cb, ROW_B, {.page_id = ids[i], .offset_bytes = 0}, {.offset_bytes = RAW_OFF + g * ROW_B});
            }
        }
        noc.async_read_barrier();
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t i = i0 + g * NCORES;
            if (i < N && ids[i] != 0xFFFFFFFFu) {
                if constexpr (POOL_FP8) {
                    volatile tt_l1_ptr uint32_t* r =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + RAW_OFF + g * 512);
                    volatile tt_l1_ptr uint32_t* o =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + OUT_OFF + g * 1024);
                    for (uint32_t w = 0; w < 128; ++w) {
                        const uint32_t x = r[w];
                        o[2 * w] = e4m3_to_bf16(x & 0xFF) | (e4m3_to_bf16((x >> 8) & 0xFF) << 16);
                        o[2 * w + 1] = e4m3_to_bf16((x >> 16) & 0xFF) | (e4m3_to_bf16(x >> 24) << 16);
                    }
                    noc.async_write(
                        cb, out_acc, 1024, {.offset_bytes = OUT_OFF + g * 1024}, {.page_id = i, .offset_bytes = 0});
                } else {
                    noc.async_write(
                        cb, out_acc, 1024, {.offset_bytes = RAW_OFF + g * 1024}, {.page_id = i, .offset_bytes = 0});
                }
            }
        }
        noc.async_write_barrier();
    }
}
