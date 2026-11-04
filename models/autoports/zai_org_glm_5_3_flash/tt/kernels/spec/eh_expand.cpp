// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 17 MTP input (tt/spec_mtp.py EhExpand; doc/ttnn/17-mtp-spec-decode/design.md §3), one owner core (BRISC):
// x0 = eh_proj(...) [1, 1, T, 4096] bf16 TILE (one row of tiles) -> the MTP layer's stacked residual [T, 1, 4, 4096]
// fp32: tile (t, c0 + j) rows 0..3 = row t of x0's column tile c0 + j (the 4 identical streams of the identity mHC,
// design.md §3; MkEmbed's layout), every other row 0. bf16 -> fp32 is exact (bits << 16).
// Compile-time args: TensorAccessorArgs of x0, res. Runtime args: x0, res addresses, c0, T (<= 4).
// CB 8 (kernels/head/head_common.hpp EMB_CB): 2 KB bf16 staging + one fp32 tile (+ 64-B alignment).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/head/head_common.hpp"

using namespace head;

void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto r_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto res = TensorAccessor(r_args, get_arg_val<uint32_t>(1));
    const uint32_t c0 = get_arg_val<uint32_t>(2);
    const uint32_t T = get_arg_val<uint32_t>(3);

    const uint32_t stage = (get_write_ptr(EMB_CB) + 63u) & ~63u;  // 2 KB: one bf16 tile of x0
    const uint32_t tile = stage + 2048;                           // one fp32 tile
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
    for (uint32_t j = 0; j < COLS; ++j) {
        noc_async_read_page(c0 + j, xs, stage);
        for (uint32_t off = 0; off < FP32_TILE_BYTES; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, tile + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t c = 0; c < TILE; ++c) {
                const uint32_t v = static_cast<uint32_t>(s[tile_off(t, c)]) << 16;
                for (uint32_t st = 0; st < N_STREAMS; ++st) {
                    d[tile_off(st, c)] = v;
                }
            }
            noc_async_write_page(t * COL_TILES + c0 + j, res, tile);
            noc_async_write_barrier();  // the tile is rebuilt for the next row
        }
    }
}
