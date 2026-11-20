// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The append core's feeder in program P1 of the attention half (fused/layer_attn/program.py; PLAN.md section 17).
// The 18 pool tiles of the raw latent row kv_a(x) of the 8 token rows ([8, 32] BF16 tiles: 16 of k_pass, 2 of
// k_rope, one from each of the chips 0..17) land in this core's arena with the Xa all-gather. This kernel
//   1. fills the two reduce scalers (1 / 512 for the mean of squares, 1 for the block maximum) and reads the layer's
//      kv norm weight (16 tiles) and the step's cos / sin tiles (4) from DRAM: all before the pool tiles are there;
//   2. waits (bounded) for the pool tiles and takes their count;
//   3. lays them out as standard [32, 32] tiles for the compute kernel: rows 0..7 are the 8 token rows, rows 8..31
//      zero. A small tile's memory is [columns 0..15 of the 8 rows | columns 16..31], which are the first 8 rows of
//      faces 0 and 1 of a standard tile. Tiles 18 and 19 are the two rope tiles with every pair (2 i, 2 i + 1)
//      turned into (-x[2 i + 1], x[2 i]): the second operand of the rotation, made here with integer operations.
// Every buffer gets its tiles in every outcome (a wait that runs out leaves its code in the status word).
//
// Compile-time args: cb_raw (20 BF16 tiles), cb_w (16), cb_cs (4: cos 0, cos 1, sin 0, sin 1), cb_mean (BF16 scaler),
// cb_one (Float32 scaler).
// Runtime args: pool tiles to wait for, their landing address, the weight tensor's address, the cos / sin tensor's.
// Stamps (HY4L_TIMING): 0 start, 1 weights read, 2 pool tiles arrived, 3 tiles handed to the compute kernel.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#include "attn_common.hpp"

constexpr uint32_t cb_raw = get_compile_time_arg_val(0);
constexpr uint32_t cb_w = get_compile_time_arg_val(1);
constexpr uint32_t cb_cs = get_compile_time_arg_val(2);
constexpr uint32_t cb_mean = get_compile_time_arg_val(3);
constexpr uint32_t cb_one = get_compile_time_arg_val(4);

namespace {
#ifdef HY4A_APPEND_ROWS
// (a pass of HY4A_APPEND_ROWS = 16 token rows, PLAN.md section 19.15: the pool tiles are [16, 32] tiles, whose two
// halves fill faces 0 and 1 of a standard tile; rows 0..15 are the token rows)
constexpr uint32_t TINY = 64 * HY4A_APPEND_ROWS;
static_assert(HY4A_APPEND_ROWS == 16, "a row tile is [8, 32] or [16, 32]");
#else
constexpr uint32_t TINY = 512;     // one [8, 32] BF16 tile
#endif
constexpr uint32_t TILE16 = 2048;  // one [32, 32] BF16 tile
constexpr uint32_t TILE32 = 4096;  // one [32, 32] Float32 tile
constexpr uint32_t PASS = 16, ROPE = 2, POOL = PASS + ROPE;

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
    const uint32_t n_pool = get_arg_val<uint32_t>(0);
    const uint32_t landing = get_arg_val<uint32_t>(1);
    const uint32_t weight_addr = get_arg_val<uint32_t>(2);
    const uint32_t cs_addr = get_arg_val<uint32_t>(3);

    hy4l::stamp(0);
    // the scalers: the value in row 0 of each of the 4 faces, zero elsewhere
    cb_reserve_back(cb_mean, 1);
    zero(cb_mean, 1, TILE16);
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_mean));
        for (uint32_t face = 0; face < 4; ++face) {
            for (uint32_t i = 0; i < 8; ++i) {
                p[face * (TILE16 / 16) + i] = 0x3B003B00u;  // two BF16 values of 1 / 512
            }
        }
    }
    cb_push_back(cb_mean, 1);
    cb_reserve_back(cb_one, 1);
    zero(cb_one, 1, TILE32);
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_one));
        for (uint32_t face = 0; face < 4; ++face) {
            for (uint32_t i = 0; i < 16; ++i) {
                p[face * (TILE32 / 16) + i] = 0x3F800000u;  // 1.0f
            }
        }
    }
    cb_push_back(cb_one, 1);

    cb_reserve_back(cb_w, PASS);
    cb_reserve_back(cb_cs, 4);
    for (uint32_t t = 0; t < PASS; ++t) {
        noc_async_read(tile_page(weight_addr, t), get_write_ptr(cb_w) + t * TILE16, TILE16);
    }
    for (uint32_t t = 0; t < 4; ++t) {
        noc_async_read(tile_page(cs_addr, t), get_write_ptr(cb_cs) + t * TILE16, TILE16);
    }
    cb_reserve_back(cb_raw, POOL + ROPE);
    zero(cb_raw, POOL + ROPE, TILE16);
    noc_async_read_barrier();
    cb_push_back(cb_w, PASS);
    cb_push_back(cb_cs, 4);
    hy4l::stamp(1);

    if (hy4a::wait_word(hy4a::A_POOL, n_pool, hy4a::ST_POOL)) {
        hy4a::take_word(hy4a::A_POOL, n_pool);
    }
    hy4l::stamp(2);
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(landing);
    const uint32_t raw = get_write_ptr(cb_raw);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw);
    constexpr uint32_t half = TINY / 8;     // words of one half of a small tile (8 rows x 16 values)
    constexpr uint32_t face = TILE16 / 16;  // words of one face of a standard tile
    for (uint32_t t = 0; t < POOL; ++t) {   // (copies inside this core's L1, done by the NoC)
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_read(
                get_noc_addr(landing + t * TINY + h * (TINY / 2)), raw + t * TILE16 + h * (TILE16 / 4), TINY / 2);
        }
    }
    for (uint32_t t = 0; t < ROPE; ++t) {
        for (uint32_t h = 0; h < 2; ++h) {
            for (uint32_t i = 0; i < half; ++i) {
                const uint32_t pair = src[(PASS + t) * (TINY / 4) + h * half + i];  // x[2 i] | x[2 i + 1] << 16
                dst[(POOL + t) * (TILE16 / 4) + h * face + i] = (((pair >> 16) ^ 0x8000u) & 0xFFFFu) | (pair << 16);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_raw, POOL + ROPE);
    hy4l::stamp(3);
}
