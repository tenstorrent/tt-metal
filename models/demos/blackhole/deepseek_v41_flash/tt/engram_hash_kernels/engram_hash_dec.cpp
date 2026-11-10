// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DeepSeek-V4.1 Engram n-gram hash for the DECODE verify block (DSV41_DEV_ENGRAM=1), bit-exact int64 semantics.
// One instance = a run of block tokens [t0, t0+cnt) of the GLOBAL block (all TTOT = users * n rows of the mesh, user-major, column j of user u is
// token t = u * N + j); every chip hashes the whole block redundantly (each instance 1 token: ~48 64-bit modulos).
//   ctrl [*, CW] int32 row-major (page CW*4 B; this chip's shard, only page 0 is read; every row of the shard carries the same words):
//        [0] token (own row, unused here) [1] position (unused here)
//        [2 .. 2+TTOT)                 RAW token id of every block row t (flat, user-major)
//        [2+TTOT + u*(W-1) + (s-1)]    COMPRESSED token id (token_map applied) of user u's committed token s steps before the block's column 0
//                                      (s = 1 .. W-1), -1 when blocked (before the sequence start or a DEAD cache entry; sticky as in the reference)
//   table [V, 16] int32 (64 B pages): compressed-token map, entry 0 of each row.
//   consts[1, 128] uint32: [mult lo/hi per (layer, shift)] [primes (layer, shift-1, head)] [offsets] [pad_id]   (same as engram_hash.cpp)
//   out   [TTOT, 64] int32 (page 256 B): out[t][layer * (W-1)*H + (shift-1)*H + head] = GLOBAL table row; only the first NL*(W-1)*H entries are meaningful (the rest is 0).
// Token (u, j) at shift s reads: s <= j: block row t - s (raw -> table lookup); s > j: committed lookback s - j of user u (already compressed).
// MODE 0: native uint64_t (compiler 64-bit mul/mod routines); MODE 1: explicit 32-bit limb arithmetic with shift-subtract modulo.

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
    constexpr uint32_t TTOT = get_compile_time_arg_val(1);
    constexpr uint32_t N = get_compile_time_arg_val(2);
    constexpr uint32_t CW = get_compile_time_arg_val(3);
    constexpr uint32_t W = get_compile_time_arg_val(4);
    constexpr uint32_t NL = get_compile_time_arg_val(5);
    constexpr uint32_t H = get_compile_time_arg_val(6);
    constexpr uint32_t PER = get_compile_time_arg_val(7);  // max tokens per instance
    constexpr uint32_t cb_id = get_compile_time_arg_val(8);
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto t_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t NCOL = (W - 1) * H;
    constexpr uint32_t OFF_P = NL * W * 2, OFF_O = OFF_P + NL * NCOL, OFF_PAD = OFF_O + NL * NCOL;
    constexpr uint32_t CWB = (CW * 4 + 63) / 64 * 64;
    // L1 carve-up (bytes) inside the CB: consts 512 | ctrl page CWB | looked-up tokens PER*W*64 | out PER*256
    constexpr uint32_t B_C = 0, B_X = 512, B_TOK = B_X + CWB, B_OUT = B_TOK + PER * W * 64;

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t t_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t cnt = get_arg_val<uint32_t>(1);
    if (cnt == 0) {
        return;
    }

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, CW * 4);
    const auto t_acc = TensorAccessor(t_args, t_addr, 64);
    const auto c_acc = TensorAccessor(c_args, c_addr, 512);
    const auto o_acc = TensorAccessor(o_args, o_addr, 256);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* cst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_C);
    volatile tt_l1_ptr int32_t* ctl = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + B_X);
    volatile tt_l1_ptr int32_t* tok = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + B_TOK);
    volatile tt_l1_ptr int32_t* out = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + B_OUT);

    noc.async_read(c_acc, cb, 512, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = B_C});
    noc.async_read(x_acc, cb, CW * 4, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = B_X});
    noc.async_read_barrier();
    // compressed-token lookup of the in-block entries
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t t = t0 + i, j = t % N;
        for (uint32_t s = 0; s < W; ++s) {
            if (s <= j) {
                const int32_t id = ctl[2 + t - s];
                noc.async_read(
                    t_acc,
                    cb,
                    64,
                    {.page_id = (uint32_t)id, .offset_bytes = 0},
                    {.offset_bytes = B_TOK + (i * W + s) * 64});
            }
        }
    }
    noc.async_read_barrier();

    const uint32_t pad = cst[OFF_PAD];
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t t = t0 + i, u = t / N, j = t % N;
        uint32_t tk[W];
        bool blocked = false;
        for (uint32_t s = 0; s < W; ++s) {
            int32_t v;
            if (s <= j) {
                v = tok[(i * W + s) * 16];
            } else {
                v = ctl[2 + TTOT + u * (W - 1) + (s - j - 1)];
            }
            blocked = blocked || (v < 0);  // sticky, like the reference's running `blocked`
            tk[s] = blocked ? pad : (uint32_t)v;
        }
        for (uint32_t w = 0; w < 64; ++w) {
            out[i * 64 + w] = 0;
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
                            out[i * 64 + l * NCOL + (s - 1) * H + h] =
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
                            uint32_t r = rhi % p;  // p < 2^31: reduce the high word first, then 32 shift-subtract steps
                            for (int b = 31; b >= 0; --b) {
                                r = (r << 1) | ((rlo >> b) & 1);
                                if (r >= p) {
                                    r -= p;
                                }
                            }
                            out[i * 64 + l * NCOL + (s - 1) * H + h] =
                                (int32_t)r + (int32_t)cst[OFF_O + l * NCOL + (s - 1) * H + h];
                        }
                    }
                }
            }
        }
    }
    for (uint32_t i = 0; i < cnt; ++i) {
        noc.async_write(cb, o_acc, 256, {.offset_bytes = B_OUT + i * 256}, {.page_id = t0 + i, .offset_bytes = 0});
    }
    noc.async_write_barrier();
}
