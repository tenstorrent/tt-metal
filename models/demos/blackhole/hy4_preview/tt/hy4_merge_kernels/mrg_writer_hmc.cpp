// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 decode merge writer for the COMPACT (verify rows, S >= 2) A1 merge, HEAD-MAJOR output (HY4_ATTN_HM=1): the merge
// tile (tile row t, column j) holds user s = 4t+b at rows 8b..8b+7 (heads 0..7). Row 8b+h of CB tile j -> row s of out
// tile (h*16 + j) of out [1,8,S,512] TILE (padded [1,8,32,512], persistent, zero-initialised), face 0/1 (s < 16) or 2/3
// (s >= 16), 32 B per face: 2 NoC writes of 32 B per (user, head). Rows of one out tile are written by different merge
// cores (no read-modify-write: sub-line NoC writes).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t TPC = get_compile_time_arg_val(0);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);
    constexpr uint32_t NU = get_compile_time_arg_val(2);  // real users (verify rows)
    constexpr auto o_args = TensorAccessorArgs<3, 1>();
    constexpr uint32_t TILE_B = 2048;
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t t = get_arg_val<uint32_t>(0);
    const uint32_t j0 = get_arg_val<uint32_t>(1);
    const auto oa = TensorAccessor(o_args, o_addr, TILE_B);
    CircularBuffer o(CB_OUT);
    o.wait_front(TPC);
    const uint32_t p = o.get_read_ptr();
    for (uint32_t j = 0; j < TPC; ++j) {
        for (uint32_t b = 0; b < 4; ++b) {
            const uint32_t s = 4 * t + b;
            if (s >= NU) {
                break;
            }
            const uint32_t dface = (s < 16) ? 0 : 1024;
            const uint32_t drow = (s & 15) * 32;
            for (uint32_t h = 0; h < 8; ++h) {
                const uint32_t sr = 8 * b + h;  // source row (< 32)
                const uint32_t sface = (sr < 16) ? 0 : 1024;
                const uint32_t srow = (sr & 15) * 32;
                const uint32_t tid = h * 16 + j0 + j;
                noc_async_write(p + j * TILE_B + sface + srow, oa.get_noc_addr(tid, dface + drow), 32);
                noc_async_write(p + j * TILE_B + sface + 512 + srow, oa.get_noc_addr(tid, dface + 512 + drow), 32);
            }
        }
    }
    noc_async_write_barrier();
    o.pop_front(TPC);
}
