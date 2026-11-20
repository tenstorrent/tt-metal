// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The key core's feeder in program P1 of a full-indexer layer (fused/layer_attn/program.py; PLAN.md section 20).
// The 4 tiles of the RAW index key wk(x) of the 8 token rows ([8, 32] BF16 tiles in the device's column order,
// [rope 64, interleaved | pass 64]: the pool tiles of chips 18..21) land in this core's arena with the Xa
// all-gather. This kernel
//   1. fills the reduce scaler (1 / 128: mean over the key's 128 dims) and reads from DRAM the LayerNorm's weight
//      and bias (6 tiles each: the 4 of the key and the pair-swapped twins of the 2 rope tiles), the step's cos /
//      sin tiles (4) and the Hadamard matrix (16): all before the key tiles are there;
//   2. waits (bounded) for the key tiles and takes their count;
//   3. lays them out as standard [32, 32] tiles for the compute kernel, rows 0..7 the token rows, rows 8..31 zero
//      (a small tile's memory is [columns 0..15 of the 8 rows | columns 16..31]: the first 8 rows of faces 0 and 1
//      of a standard tile): cb_raw tiles 0..3 the key, tiles 4, 5 the two rope tiles with the members of every pair
//      (2 i, 2 i + 1) exchanged (no sign: the twins' sign is in their norm weight and bias), made here with integer
//      operations; cb_sum the 4 key tiles once more (the mean's reduce pops them).
// Every buffer gets its tiles in every outcome (a wait that runs out leaves its code in the status word).
//
// Compile-time args: cb_raw (6 BF16 tiles), cb_sum (4), cb_wb (12: weight 6 | bias 6), cb_cs (4: cos 0, cos 1, sin
// 0, sin 1), cb_h (16), cb_mean (BF16 scaler).
// Runtime args: key tiles to wait for, their landing address, the weight | bias tensor's address, the cos / sin
// tensor's, the Hadamard tensor's.
// Stamps (HY4L_TIMING): 0 start, 1 constants read, 2 key tiles arrived, 3 tiles handed to the compute kernel.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#include "attn_common.hpp"

constexpr uint32_t cb_raw = get_compile_time_arg_val(0);
constexpr uint32_t cb_sum = get_compile_time_arg_val(1);
constexpr uint32_t cb_wb = get_compile_time_arg_val(2);
constexpr uint32_t cb_cs = get_compile_time_arg_val(3);
constexpr uint32_t cb_h = get_compile_time_arg_val(4);
constexpr uint32_t cb_mean = get_compile_time_arg_val(5);

namespace {
constexpr uint32_t TINY = 512;     // one [8, 32] BF16 tile
constexpr uint32_t TILE16 = 2048;  // one [32, 32] BF16 tile
constexpr uint32_t KEY = 4, ROPE = 2, WB = 2 * (KEY + ROPE), HAD = 16;

FORCE_INLINE uint64_t tile_page(uint32_t base, uint32_t page) {  // an interleaved DRAM tensor of BF16 tiles
    return get_noc_addr_from_bank_id<true>(page % NUM_DRAM_BANKS, base + (page / NUM_DRAM_BANKS) * TILE16);
}

inline void zero(uint32_t cb, uint32_t tiles, uint32_t tile_bytes) {
    Noc noc;
    experimental::CB buffer(cb);
    for (uint32_t t = 0; t < tiles; ++t) {
        noc.async_write_zeros(buffer, tile_bytes, {.offset_bytes = t * tile_bytes});
    }
    noc.write_zeros_l1_barrier();
}
}  // namespace

void kernel_main() {
    const uint32_t n_key = get_arg_val<uint32_t>(0);
    const uint32_t landing = get_arg_val<uint32_t>(1);
    const uint32_t wb_addr = get_arg_val<uint32_t>(2);
    const uint32_t cs_addr = get_arg_val<uint32_t>(3);
    const uint32_t h_addr = get_arg_val<uint32_t>(4);

    hy4l::stamp(0);
    // the scaler: the value in row 0 of each of the 4 faces, zero elsewhere
    cb_reserve_back(cb_mean, 1);
    zero(cb_mean, 1, TILE16);
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_mean));
        for (uint32_t face = 0; face < 4; ++face) {
            for (uint32_t i = 0; i < 8; ++i) {
                p[face * (TILE16 / 16) + i] = 0x3C003C00u;  // two BF16 values of 1 / 128
            }
        }
    }
    cb_push_back(cb_mean, 1);

    cb_reserve_back(cb_wb, WB);
    cb_reserve_back(cb_cs, 4);
    cb_reserve_back(cb_h, HAD);
    for (uint32_t t = 0; t < WB; ++t) {
        noc_async_read(tile_page(wb_addr, t), get_write_ptr(cb_wb) + t * TILE16, TILE16);
    }
    for (uint32_t t = 0; t < 4; ++t) {
        noc_async_read(tile_page(cs_addr, t), get_write_ptr(cb_cs) + t * TILE16, TILE16);
    }
    for (uint32_t t = 0; t < HAD; ++t) {
        noc_async_read(tile_page(h_addr, t), get_write_ptr(cb_h) + t * TILE16, TILE16);
    }
    cb_reserve_back(cb_raw, KEY + ROPE);
    cb_reserve_back(cb_sum, KEY);
    zero(cb_raw, KEY + ROPE, TILE16);
    zero(cb_sum, KEY, TILE16);
    noc_async_read_barrier();
    cb_push_back(cb_wb, WB);
    cb_push_back(cb_cs, 4);
    cb_push_back(cb_h, HAD);
    hy4l::stamp(1);

    if (hy4a::wait_word(hy4a::A_POOL, n_key, hy4a::ST_POOL)) {
        hy4a::take_word(hy4a::A_POOL, n_key);
    }
    hy4l::stamp(2);
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(landing);
    const uint32_t raw = get_write_ptr(cb_raw);
    const uint32_t sum = get_write_ptr(cb_sum);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw);
    constexpr uint32_t half = TINY / 8;     // words of one half of a small tile (8 rows x 16 values)
    constexpr uint32_t face = TILE16 / 16;  // words of one face of a standard tile
    for (uint32_t t = 0; t < KEY; ++t) {    // (copies inside this core's L1, done by the NoC)
        for (uint32_t h = 0; h < 2; ++h) {
            const uint64_t from = get_noc_addr(landing + t * TINY + h * (TINY / 2));
            noc_async_read(from, raw + t * TILE16 + h * (TILE16 / 4), TINY / 2);
            noc_async_read(from, sum + t * TILE16 + h * (TILE16 / 4), TINY / 2);
        }
    }
    for (uint32_t t = 0; t < ROPE; ++t) {
        for (uint32_t h = 0; h < 2; ++h) {
            for (uint32_t i = 0; i < half; ++i) {
                const uint32_t pair = src[t * (TINY / 4) + h * half + i];  // x[2 i] | x[2 i + 1] << 16
                dst[(KEY + t) * (TILE16 / 4) + h * face + i] = (pair >> 16) | (pair << 16);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_raw, KEY + ROPE);
    cb_push_back(cb_sum, KEY);
    hy4l::stamp(3);
}
