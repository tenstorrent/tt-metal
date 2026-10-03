// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Engram table row gather + fp8/e8m0 -> bf16 dequant on one chip's row shard (data-movement cores, no FPU).
//   ids   [U, 64] int32 row-major (page 256 B): ids[u][layer*24 + k] = GLOBAL table row (hash kernel output layout)
//   info  [1, 16] int32 (page 64 B), per chip: {base_l0, nrows_l0, base_l1, nrows_l1}
//   tab_l uint8 [rpc_l, 256] row-major (page 256 B): raw fp8 e4m3 rows of this chip's shard (local row = id - base)
//   sc_l  uint8 [ceil(rpc_l/8), 64] row-major (page 64 B): e8m0 scales, 8 B per row, 8 rows per page
//   out   bf16  [NROWS, 256] row-major (page 512 B); out row j: g = j/(4*NL*K), rem = j%(4*NL*K), layer = rem/(4*K),
//         tl = (rem%(4*K))/K, k = rem%K, user u = g*4 + tl. Zeros for ids outside this chip's range.
// Each kernel instance handles output rows [row0, row0+nrow).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

struct Lut {
    uint16_t v[256];
    constexpr Lut() : v() {
        for (int b = 0; b < 256; ++b) {
            uint32_t sign = (b >> 7) & 1, e = (b >> 3) & 15, m = b & 7;
            uint32_t bits = 0;
            if (e == 15 && m == 7) {
                bits = 0x7FC0;  // e4m3fn NaN
            } else if (e > 0) {
                bits = ((e - 7 + 127) << 7) | (m << 4);
            } else if (m > 0) {
                uint32_t p = (m >= 4) ? 2 : (m >= 2 ? 1 : 0);
                bits = ((p - 9 + 127) << 7) | ((m - (1u << p)) << (7 - p));
            } else {
                bits = 0;
            }
            v[b] = (uint16_t)((sign << 15) | bits);
        }
    }
};
static constexpr Lut LUT{};

void kernel_main() {
    constexpr uint32_t U = get_compile_time_arg_val(0);
    constexpr uint32_t NL = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t PER = get_compile_time_arg_val(3);  // max rows per kernel instance
    constexpr uint32_t cb_id = get_compile_time_arg_val(4);
    constexpr auto ids_args = TensorAccessorArgs<5>();
    constexpr auto info_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto t0_args = TensorAccessorArgs<info_args.next_compile_time_args_offset()>();
    constexpr auto s0_args = TensorAccessorArgs<t0_args.next_compile_time_args_offset()>();
    constexpr auto t1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    constexpr auto s1_args = TensorAccessorArgs<t1_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();
    static_assert(NL == 2);

    constexpr uint32_t B_IDS = 0, B_INFO = B_IDS + U * 256, B_ROW = B_INFO + 64, B_SC = B_ROW + PER * 256,
                       B_OUT = B_SC + PER * 64;

    const uint32_t row0 = get_arg_val<uint32_t>(0);
    const uint32_t nrow = get_arg_val<uint32_t>(1);
    const uint32_t a_ids = get_common_arg_val<uint32_t>(0), a_info = get_common_arg_val<uint32_t>(1),
                   a_t0 = get_common_arg_val<uint32_t>(2), a_s0 = get_common_arg_val<uint32_t>(3),
                   a_t1 = get_common_arg_val<uint32_t>(4), a_s1 = get_common_arg_val<uint32_t>(5),
                   a_o = get_common_arg_val<uint32_t>(6);
    Noc noc;
    const auto ids_acc = TensorAccessor(ids_args, a_ids, 256);
    const auto info_acc = TensorAccessor(info_args, a_info, 64);
    const auto t0_acc = TensorAccessor(t0_args, a_t0, 256);
    const auto t1_acc = TensorAccessor(t1_args, a_t1, 256);
    const auto s0_acc = TensorAccessor(s0_args, a_s0, 64);
    const auto s1_acc = TensorAccessor(s1_args, a_s1, 64);
    const auto o_acc = TensorAccessor(o_args, a_o, 512);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr int32_t* ids = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + B_IDS);
    volatile tt_l1_ptr int32_t* info = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + B_INFO);
    volatile tt_l1_ptr uint8_t* rowb = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base + B_ROW);
    volatile tt_l1_ptr uint8_t* scb = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base + B_SC);
    volatile tt_l1_ptr uint32_t* outw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_OUT);

    noc.async_read(info_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = B_INFO});
    for (uint32_t u = 0; u < U; ++u) {
        noc.async_read(ids_acc, cb, 256, {.page_id = u, .offset_bytes = 0}, {.offset_bytes = B_IDS + u * 256});
    }
    noc.async_read_barrier();

    bool valid[PER];
    uint32_t sc_off[PER];
    for (uint32_t i = 0; i < nrow; ++i) {
        const uint32_t j = row0 + i;
        const uint32_t g = j / (4 * NL * K), rem = j % (4 * NL * K);
        const uint32_t layer = rem / (4 * K), tl = (rem % (4 * K)) / K, k = rem % K;
        const int32_t id = ids[(g * 4 + tl) * 64 + layer * K + k];
        const int32_t local = id - info[layer * 2];
        valid[i] = local >= 0 && local < info[layer * 2 + 1];
        if (valid[i]) {
            const uint32_t lr = (uint32_t)local;
            if (layer == 0) {
                noc.async_read(t0_acc, cb, 256, {.page_id = lr, .offset_bytes = 0}, {.offset_bytes = B_ROW + i * 256});
                noc.async_read(
                    s0_acc, cb, 64, {.page_id = lr >> 3, .offset_bytes = 0}, {.offset_bytes = B_SC + i * 64});
            } else {
                noc.async_read(t1_acc, cb, 256, {.page_id = lr, .offset_bytes = 0}, {.offset_bytes = B_ROW + i * 256});
                noc.async_read(
                    s1_acc, cb, 64, {.page_id = lr >> 3, .offset_bytes = 0}, {.offset_bytes = B_SC + i * 64});
            }
            sc_off[i] = (lr & 7) * 8;
        }
    }
    noc.async_read_barrier();

    for (uint32_t i = 0; i < nrow; ++i) {
        volatile tt_l1_ptr uint32_t* o = outw + i * 128;
        if (!valid[i]) {
            for (uint32_t w = 0; w < 128; ++w) {
                o[w] = 0;
            }
            continue;
        }
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rowb + i * 256);
        for (uint32_t blk = 0; blk < 8; ++blk) {
            const int32_t eoff = ((int32_t)scb[i * 64 + sc_off[i] + blk] - 127) << 7;
            const bool nan_scale = scb[i * 64 + sc_off[i] + blk] == 255;
            for (uint32_t w = 0; w < 8; ++w) {
                const uint32_t word = src[blk * 8 + w];
                uint32_t res[2];
                for (uint32_t h = 0; h < 2; ++h) {
                    uint32_t pk = 0;
                    for (uint32_t q = 0; q < 2; ++q) {
                        const uint32_t b = (word >> (8 * (2 * h + q))) & 0xFF;
                        uint32_t bits = LUT.v[b];
                        if ((b & 0x7F) != 0 && (b & 0x7F) != 0x7F) {
                            bits = (uint32_t)((int32_t)bits + eoff) & 0xFFFF;
                        }
                        if (nan_scale) {
                            bits = 0x7FC0;
                        }
                        pk |= bits << (16 * q);
                    }
                    res[h] = pk;
                }
                o[blk * 16 + w * 2] = res[0];
                o[blk * 16 + w * 2 + 1] = res[1];
            }
        }
    }
    for (uint32_t i = 0; i < nrow; ++i) {
        noc.async_write(cb, o_acc, 512, {.offset_bytes = B_OUT + i * 512}, {.page_id = row0 + i, .offset_bytes = 0});
    }
    noc.async_write_barrier();
}
