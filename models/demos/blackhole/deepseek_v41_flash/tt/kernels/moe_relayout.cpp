// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One-off device copy between the two bfp8 routed-expert weight layouts of DeepSeek-V4.1-Flash (same 32x32 tiles,
// different arrangement):
//   unified : per local expert e, gate/up [K=160 tile rows, N=72 tile cols], down [72, 160] (DRAM ND-sharded,
//   TensorAccessor page = row * ncols + col) decode  : moe_compute ring layout, pages ((c * E + e) * G + g) * R + r) *
//   4 + t
//       w0_w1 : R = 161 (K padded to 7), per unit (c, e, g, kt): [gate n0, up n0, gate n1, up n1], n_i = 9 c + 2 g + i
//       (n >= 72 or kt >= 160: zero) w2    : R = 77 (N padded to 7),  per unit (c, e, g, r) : down tiles [row(c, r), 20
//       c + 4 g + t], row = 9 * chunk + r % 9, chunk = (8 - ((r / 9 + 8 - c) % 8)) % 8, r >= 72: zero
// MODE 0: w0_w1, MODE 1: w2. DIR 0: unified -> decode, DIR 1: decode -> unified (pads dropped).
// CT args: MODE, DIR, TILE_BYTES, E, then TensorAccessorArgs (MODE 0: gate, up, decode; MODE 1: down, decode).
// Common RT args: decode_addr, then E addresses of gate (+ E of up for MODE 0).  Per-core RT args: first unit, number
// of units.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

constexpr uint32_t NB = 12;  // units per batch

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t DIR = get_compile_time_arg_val(1);
    constexpr uint32_t TB = get_compile_time_arg_val(2);
    constexpr uint32_t E = get_compile_time_arg_val(3);
    constexpr uint32_t CB = 0;
    constexpr uint32_t NC = 8, G = 5, NCOL = 72, NROW_GU = 160, NCOL_D = 160;
    constexpr uint32_t R = MODE == 0 ? 161 : 77;

    constexpr auto a0 = TensorAccessorArgs<4>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto adec =
        TensorAccessorArgs<(MODE == 0 ? a1.next_compile_time_args_offset() : a0.next_compile_time_args_offset())>();

    const uint32_t dec_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    if (nu == 0) {
        return;
    }
    Noc noc;
    // MODE 0: a0 gate, a1 up, a2 decode.  MODE 1: a0 down, a1 decode.
    const auto dec = TensorAccessor(adec, dec_addr, TB);

    const uint32_t stage = get_write_ptr(CB);
    auto zero_tile = [&](uint32_t l1) {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
        for (uint32_t i = 0; i < TB / 4; ++i) {
            z[i] = 0;
        }
    };

    for (uint32_t b0 = 0; b0 < nu; b0 += NB) {
        const uint32_t nb = (nu - b0 < NB) ? (nu - b0) : NB;
        // pass 1: issue the reads (DIR 0: unified tiles; DIR 1: the decode unit), remember where each tile goes
        for (uint32_t i = 0; i < nb; ++i) {
            const uint32_t u = u0 + b0 + i;
            uint32_t rest = u / R;
            const uint32_t r = u - rest * R;
            const uint32_t g = rest % G;
            rest /= G;
            const uint32_t e = rest % E;
            const uint32_t c = rest / E;
            const uint32_t slot = stage + i * 4 * TB;
            if constexpr (DIR == 1) {
                noc.async_read(dec, CoreLocalMem<uint32_t>(slot), 4 * TB, {.page_id = u * 4}, {});
            } else if constexpr (MODE == 0) {
                const uint32_t kt = r;
#pragma GCC unroll 4
                for (uint32_t t = 0; t < 4; ++t) {
                    const uint32_t nl = 2 * g + (t >> 1);
                    const uint32_t n = 9 * c + nl;
                    const bool valid = kt < NROW_GU && nl < 9;
                    if (valid) {
                        const uint32_t addr = get_common_arg_val<uint32_t>(1 + (t & 1) * E + e);
                        if (t & 1) {
                            noc.async_read(
                                TensorAccessor(a1, addr, TB),
                                CoreLocalMem<uint32_t>(slot + t * TB),
                                TB,
                                {.page_id = kt * NCOL + n},
                                {});
                        } else {
                            noc.async_read(
                                TensorAccessor(a0, addr, TB),
                                CoreLocalMem<uint32_t>(slot + t * TB),
                                TB,
                                {.page_id = kt * NCOL + n},
                                {});
                        }
                    } else {
                        zero_tile(slot + t * TB);
                    }
                }
            } else {
                const uint32_t addr = get_common_arg_val<uint32_t>(1 + e);
                const auto acc = TensorAccessor(a0, addr, TB);
                if (r < NCOL) {
                    const uint32_t p = r / 9;
                    const uint32_t chunk = (8 - ((p + 8 - c) % 8)) % 8;
                    const uint32_t row = chunk * 9 + (r - p * 9);
#pragma GCC unroll 4
                    for (uint32_t t = 0; t < 4; ++t) {
                        noc.async_read(
                            acc,
                            CoreLocalMem<uint32_t>(slot + t * TB),
                            TB,
                            {.page_id = row * NCOL_D + 20 * c + 4 * g + t},
                            {});
                    }
                } else {
                    for (uint32_t t = 0; t < 4; ++t) {
                        zero_tile(slot + t * TB);
                    }
                }
            }
        }
        noc.async_read_barrier();
        // pass 2: writes
        if constexpr (DIR == 0) {
            for (uint32_t i = 0; i < nb; ++i) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(stage + i * 4 * TB), dec, 4 * TB, {}, {.page_id = (u0 + b0 + i) * 4});
            }
        } else {
            for (uint32_t i = 0; i < nb; ++i) {
                const uint32_t u = u0 + b0 + i;
                uint32_t rest = u / R;
                const uint32_t r = u - rest * R;
                const uint32_t g = rest % G;
                rest /= G;
                const uint32_t e = rest % E;
                const uint32_t c = rest / E;
                const uint32_t slot = stage + i * 4 * TB;
                if constexpr (MODE == 0) {
                    const uint32_t kt = r;
                    for (uint32_t t = 0; t < 4; ++t) {
                        const uint32_t nl = 2 * g + (t >> 1);
                        if (kt < NROW_GU && nl < 9) {
                            const uint32_t addr = get_common_arg_val<uint32_t>(1 + (t & 1) * E + e);
                            if (t & 1) {
                                noc.async_write(
                                    CoreLocalMem<uint32_t>(slot + t * TB),
                                    TensorAccessor(a1, addr, TB),
                                    TB,
                                    {},
                                    {.page_id = kt * NCOL + 9 * c + nl});
                            } else {
                                noc.async_write(
                                    CoreLocalMem<uint32_t>(slot + t * TB),
                                    TensorAccessor(a0, addr, TB),
                                    TB,
                                    {},
                                    {.page_id = kt * NCOL + 9 * c + nl});
                            }
                        }
                    }
                } else if (r < NCOL) {
                    const uint32_t addr = get_common_arg_val<uint32_t>(1 + e);
                    const auto acc = TensorAccessor(a0, addr, TB);
                    const uint32_t p = r / 9;
                    const uint32_t chunk = (8 - ((p + 8 - c) % 8)) % 8;
                    const uint32_t row = chunk * 9 + (r - p * 9);
                    for (uint32_t t = 0; t < 4; ++t) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(slot + t * TB),
                            acc,
                            TB,
                            {},
                            {.page_id = row * NCOL_D + 20 * c + 4 * g + t});
                    }
                }
            }
        }
        if constexpr (DIR == 0) {
            noc.async_write_barrier();
        } else {
            noc.async_write_barrier();
        }
    }
}
