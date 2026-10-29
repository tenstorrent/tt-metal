// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident MLP half-layer: shared compile-time schedule of the streamer cores.
//
// Per chip, S streamer cores (bank-adjacent) and one hub core run `layers` residual MLP blocks
// x <- x + sum_chips down_c(silu(h @ G_c) * (h @ U_c)),  h = rmsnorm(x) * gamma,
// with gate|up and down TP-sharded over the chips. Every streamer holds a replica of x and computes
// h itself, streams its column slice of gate|up and down from its DRAM bank through the weight ring,
// exchanges its slice of the intermediate activation with the other streamers directly, and sends its
// down-projection columns to the hub. Chip 0 folds x into its partial, so the new x is the plain sum
// of the chips' partials; the hub all-reduces them over fabric and multicasts the per-chip slots back.
//
// Compile-time args (identical on the three RISCs of a streamer):
//   0 layers, 1 weight_sets, 2 num_chips, 3 num_streamers, 4 hidden tiles Ht, 5 inter tiles It,
//   6 chip index, 7 ring_bytes, 8 rmsnorm epsilon (fp32 bits), 9 1/sqrt(hidden) (fp32 bits),
//   10 sem_slots id, 11 sem_act id, 12 sem_gather id,
//   then per entry e (0 gate|up, 1 down), 5 values at 13 + 5e: Kt, sb, pages, page_size, block_bytes,
//   23 debug mode, 24 ng_max, 25 nd_max (per-core tile counts differ by one between the cores of a bank;
//   every core pushes / pops the maximum so each per-layer CB block wraps exactly at the CB end)
#pragma once

#include <stdint.h>

namespace resident_mlp {

constexpr uint32_t kEntries = 2;
// streamer CBs
constexpr uint32_t cb_slots = 0;   // num_chips x Ht partial slots, written by the hub multicast
constexpr uint32_t cb_w0 = 1;      // weight ring: entry e uses CB cb_w0 + e
constexpr uint32_t cb_x = 3;       // residual replica (Ht)
constexpr uint32_t cb_h = 4;       // rmsnorm(x) * gamma (Ht), in0 of gate|up
constexpr uint32_t cb_gu = 5;      // gate tiles then up tiles of this core's columns
constexpr uint32_t cb_aslice = 6;  // silu(g) * u of this core's columns, sent to every streamer
constexpr uint32_t cb_consumed = 7;
constexpr uint32_t cb_act = 8;     // full intermediate activation (It), written by the streamers, in0 of down
constexpr uint32_t cb_dout = 9;    // down-projection columns of this core
constexpr uint32_t cb_pout = 10;   // partial sent to the hub (chip 0: + x)
constexpr uint32_t cb_gamma = 11;  // weight_sets x Ht rmsnorm weights
// 32x32 views of the same bytes (rmsnorm is elementwise apart from the global sum, and its LLKs take
// at most 8 full tiles, so the 1 x hidden vector is normalized as hidden / 1024 full tiles)
constexpr uint32_t cb_x_full = 12;
constexpr uint32_t cb_h_full = 13;
constexpr uint32_t cb_gamma_full = 14;
constexpr uint32_t kUnit = 64;
constexpr uint32_t kTileBytes = 64;  // 1x32 bf16

constexpr uint32_t layers = get_compile_time_arg_val(0);
constexpr uint32_t weight_sets = get_compile_time_arg_val(1);
constexpr uint32_t num_chips = get_compile_time_arg_val(2);
constexpr uint32_t num_streamers = get_compile_time_arg_val(3);
constexpr uint32_t Ht = get_compile_time_arg_val(4);
constexpr uint32_t It = get_compile_time_arg_val(5);
constexpr uint32_t chip = get_compile_time_arg_val(6);
constexpr uint32_t ring_bytes = get_compile_time_arg_val(7);
constexpr uint32_t eps_bits = get_compile_time_arg_val(8);
constexpr uint32_t inv_sqrt_hidden_bits = get_compile_time_arg_val(9);
constexpr uint32_t sem_slots = get_compile_time_arg_val(10);
constexpr uint32_t sem_act = get_compile_time_arg_val(11);
constexpr uint32_t sem_gather = get_compile_time_arg_val(12);
constexpr uint32_t Hf = Ht / 32;
constexpr uint32_t ng_max = get_compile_time_arg_val(24);
constexpr uint32_t nd_max = get_compile_time_arg_val(25);  // full 32x32 tiles in the hidden vector

template <uint32_t E>
struct Entry {
    static constexpr uint32_t Kt = get_compile_time_arg_val(13 + 5 * E + 0);
    static constexpr uint32_t sb = get_compile_time_arg_val(13 + 5 * E + 1);
    static constexpr uint32_t pages = get_compile_time_arg_val(13 + 5 * E + 2);
    static constexpr uint32_t page_size = get_compile_time_arg_val(13 + 5 * E + 3);
    static constexpr uint32_t block_bytes = get_compile_time_arg_val(13 + 5 * E + 4);
    static constexpr uint32_t nkb = Kt / sb;
    static constexpr uint32_t cb = cb_w0 + E;
};

// Deterministic block placement in the weight ring (see stream_engine_common.hpp).
struct RingCursor {
    uint32_t pos = 0;
    uint32_t units = 0;

    template <uint32_t E>
    uint32_t place() {
        constexpr uint32_t block = Entry<E>::block_bytes;
        if (pos + block > ring_bytes) {
            units += (ring_bytes - pos) / kUnit;
            pos = 0;
        }
        const uint32_t at = pos;
        pos += block;
        units += block / kUnit;
        return at;
    }
};

}  // namespace resident_mlp
