// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DeepSeek-V4.1 Engram n-gram hash, bit-exact int64 semantics, one data-movement kernel on one Tensix core.
//   hist  [T, 16] int32 row-major: hist[t][s] = RAW token id s steps back from user t's current token (s = 0 is the
//         current token), -1 when before the sequence start. (Only the first W entries are used.)
//   table [V, 16] int32 row-major: compressed-token map, entry 0 of each row (64 B pages so the NOC read is aligned).
//   consts[1, 128] uint32: [mult lo/hi per (layer, shift)] [primes (layer, shift-1, head)] [offsets] [pad_id]
//   out   [T, 64] int32 row-major: out[t][layer * (W-1)*H + (shift-1)*H + head] (first NL*(W-1)*H entries used)
// MODE 0: native uint64_t (compiler 64-bit mul/mod support routines); MODE 1: explicit 32-bit limb arithmetic.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

static inline void mul32x32(uint32_t a, uint32_t b, uint32_t& lo, uint32_t& hi) {
    uint32_t a0 = a & 0xFFFF, a1 = a >> 16, b0 = b & 0xFFFF, b1 = b >> 16;
    uint32_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint32_t mid = (p00 >> 16) + (p01 & 0xFFFF) + (p10 & 0xFFFF);
    lo = (p00 & 0xFFFF) | (mid << 16);
    hi = p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16);
}

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t T = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr uint32_t NL = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id = get_compile_time_arg_val(5);
    constexpr auto h_args = TensorAccessorArgs<6>();
    constexpr auto t_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t NCOL = (W - 1) * H;
    constexpr uint32_t OFF_P = NL * W * 2, OFF_O = OFF_P + NL * NCOL, OFF_PAD = OFF_O + NL * NCOL;
    // L1 carve-up (bytes) inside the CB: consts 512 | tokens T*W*64 | hist T*64 | out T*256
    constexpr uint32_t B_C = 0, B_TOK = 512, B_H = B_TOK + T * W * 64, B_OUT = B_H + T * 64;

    const uint32_t h_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t t_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(3);

    Noc noc;
    const auto h_acc = TensorAccessor(h_args, h_addr, 64);
    const auto t_acc = TensorAccessor(t_args, t_addr, 64);
    const auto c_acc = TensorAccessor(c_args, c_addr, 512);
    const auto o_acc = TensorAccessor(o_args, o_addr, 256);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr int32_t* hist = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + B_H);
    volatile tt_l1_ptr uint32_t* cst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_C);
    volatile tt_l1_ptr int32_t* tok = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + B_TOK);
    volatile tt_l1_ptr int32_t* out = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + B_OUT);

    noc.async_read(c_acc, cb, 512, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = B_C});
    for (uint32_t t = 0; t < T; ++t) {
        noc.async_read(h_acc, cb, 64, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = B_H + t * 64});
    }
    noc.async_read_barrier();
    // compressed-token lookup (only for valid entries)
    for (uint32_t i = 0; i < T * W; ++i) {
        int32_t id = hist[(i / W) * 16 + (i % W)];
        if (id >= 0) {
            noc.async_read(
                t_acc, cb, 64, {.page_id = (uint32_t)id, .offset_bytes = 0}, {.offset_bytes = B_TOK + i * 64});
        }
    }
    noc.async_read_barrier();

    const uint32_t pad = cst[OFF_PAD];
    for (uint32_t t = 0; t < T; ++t) {
        uint32_t tk[W];
        bool blocked = false;
        for (uint32_t s = 0; s < W; ++s) {
            int32_t id = hist[t * 16 + s];
            blocked = blocked || (id < 0);  // sticky, like the reference's running `blocked`
            tk[s] = blocked ? pad : (uint32_t)tok[(t * W + s) * 16];
        }
        for (uint32_t l = 0; l < NL; ++l) {
            if constexpr (MODE == 0) {
                uint64_t rolling = 0;
                for (uint32_t s = 0; s < W; ++s) {
                    uint64_t m = ((uint64_t)cst[(l * W + s) * 2 + 1] << 32) | cst[(l * W + s) * 2];
                    uint64_t prod = (uint64_t)tk[s] * m;
                    rolling = (s == 0) ? prod : (rolling ^ prod);
                    if (s >= 1) {
                        for (uint32_t h = 0; h < H; ++h) {
                            uint64_t p = cst[OFF_P + l * NCOL + (s - 1) * H + h];
                            out[t * 64 + l * NCOL + (s - 1) * H + h] =
                                (int32_t)(rolling % p) + (int32_t)cst[OFF_O + l * NCOL + (s - 1) * H + h];
                        }
                    }
                }
            } else {
                uint32_t rlo = 0, rhi = 0;
                for (uint32_t s = 0; s < W; ++s) {
                    uint32_t mlo = cst[(l * W + s) * 2], mhi = cst[(l * W + s) * 2 + 1];
                    uint32_t plo, phi;
                    mul32x32(tk[s], mlo, plo, phi);
                    phi += tk[s] * mhi;  // no overflow of the 64-bit product (reference bound)
                    if (s == 0) {
                        rlo = plo;
                        rhi = phi;
                    } else {
                        rlo ^= plo;
                        rhi ^= phi;
                    }
                    if (s >= 1) {
                        for (uint32_t h = 0; h < H; ++h) {
                            uint32_t p = cst[OFF_P + l * NCOL + (s - 1) * H + h];
                            uint32_t r = 0;  // shift-subtract long division, p < 2^31
                            for (int b = 63; b >= 0; --b) {
                                uint32_t bit = (b >= 32) ? ((rhi >> (b - 32)) & 1) : ((rlo >> b) & 1);
                                r = (r << 1) | bit;
                                if (r >= p) {
                                    r -= p;
                                }
                            }
                            out[t * 64 + l * NCOL + (s - 1) * H + h] =
                                (int32_t)r + (int32_t)cst[OFF_O + l * NCOL + (s - 1) * H + h];
                        }
                    }
                }
            }
        }
    }
    for (uint32_t t = 0; t < T; ++t) {
        noc.async_write(cb, o_acc, 256, {.offset_bytes = B_OUT + t * 256}, {.page_id = t, .offset_bytes = 0});
    }
    noc.async_write_barrier();
}
