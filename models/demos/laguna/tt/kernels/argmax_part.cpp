// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Greedy pick, stage 1 (Laguna; greedy_pick.py): each (core, RISC) scans a run of vocabulary tiles of one 32-row
// tile row of this chip's logits shard [1, 1, R, V] (bf16 TILE) and writes, per row, one packed key
// (order-preserving bf16 value << 16 | 0xFFFF - column) of its best column: the largest key is the largest value and,
// among equal values, the smallest column. Unit w = core * 2 + risc covers tile row w / S, tiles (w % S) * per ...
// Output: partials [W, 32] uint32 row-major (page w = unit w's 32 rows).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t grid_x = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);   // vocabulary tiles per tile row
    constexpr uint32_t S = get_compile_time_arg_val(2);    // units per tile row
    constexpr uint32_t per = get_compile_time_arg_val(3);  // tiles per unit
    constexpr uint32_t W = get_compile_time_arg_val(4);    // units in all
    constexpr uint32_t risc = get_compile_time_arg_val(5);
    constexpr auto in_args = TensorAccessorArgs<6>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t w = core * 2 + risc;
    if (w >= W) {
        return;
    }
    const uint32_t tr = w / S, j0 = (w % S) * per;
    const uint32_t j1 = j0 + per < Vt ? j0 + per : Vt;
    const auto in = TensorAccessor(in_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), 128);
    const uint32_t buf = get_write_ptr(risc);  // two tiles (double buffer), then the 32 result words
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf + 4096);
    uint32_t best[32];
    for (uint32_t r = 0; r < 32; ++r) {
        best[r] = 0;
    }
    if (j0 < j1) {
        noc_async_read(in.get_noc_addr(tr * Vt + j0), buf, 2048);
    }
    for (uint32_t j = j0; j < j1; ++j) {
        const uint32_t cur = buf + ((j - j0) & 1) * 2048;
        noc_async_read_barrier();
        if (j + 1 < j1) {
            noc_async_read(in.get_noc_addr(tr * Vt + j + 1), buf + ((j + 1 - j0) & 1) * 2048, 2048);
        }
        invalidate_l1_cache();
        // plain loads (the read has landed, nothing else writes this buffer); two bf16 values per 32-bit word
        const uint32_t* t = reinterpret_cast<const uint32_t*>(cur);
        const uint32_t col0 = 0xFFFFu - j * 32;  // 0xFFFF - column of the tile's first column
        for (uint32_t r = 0; r < 32; ++r) {
            // row r: 16 values (8 words) in face (r / 16) * 2 and 16 in the next face (256 words later)
            const uint32_t* row = t + (r / 16) * 256 + (r % 16) * 8;
            uint32_t b = best[r];
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t* v = row + half * 128;
                const uint32_t cb = col0 - half * 16;
                for (uint32_t c = 0; c < 8; ++c) {
                    const uint32_t w2 = v[c];
                    const uint32_t lo = w2 & 0xFFFFu, hi = w2 >> 16;
                    // order-preserving key: negative values -> ~x, others -> x | 0x8000 (branchless)
                    const uint32_t klo = lo ^ (0x8000u | ((lo >> 15) * 0x7FFFu));
                    const uint32_t khi = hi ^ (0x8000u | ((hi >> 15) * 0x7FFFu));
                    const uint32_t plo = (klo << 16) | (cb - 2 * c);
                    const uint32_t phi = (khi << 16) | (cb - 2 * c - 1);
                    b = plo > b ? plo : b;
                    b = phi > b ? phi : b;
                }
            }
            best[r] = b;
        }
    }
    for (uint32_t r = 0; r < 32; ++r) {
        res[r] = best[r];
    }
    noc_async_write(buf + 4096, out.get_noc_addr(w), 128);
    noc_async_write_barrier();
}
