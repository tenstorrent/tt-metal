// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the fused V4.1 mHC projection (mhc_proj_compute.cpp): streams the KT tiles of each tile row r of x
// [T, KT tiles], BK per block; the writer (mhc_proj_writer.cpp) streams the matching weight blocks. Each core starts
// its rows at a different block (staggered by row) so the cores do not read the same DRAM bank or weight tile in
// lockstep. cb_x and cb_xsq alias one L1 buffer (the host gives them one CB descriptor): each x block is read from
// DRAM once and published under both, as the matmul operand (cb_x) and as the fp32 unpack-to-DST input of the
// squares (cb_xsq; an unpack-to-dest CB cannot feed the matmul). Before the rows the reader builds the constant
// tile E in cb_const (ones in column MIX_COL).
//
// compile_time_args = [cb_x, cb_xsq, cb_const, KT, BK, MIX_COL, TensorAccessorArgs(x)]
// runtime args      = [x_addr, row_start, row_count]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t row_count = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xsq = get_compile_time_arg_val(1);
    constexpr uint32_t cb_const = get_compile_time_arg_val(2);
    constexpr uint32_t KT = get_compile_time_arg_val(3);
    constexpr uint32_t BK = get_compile_time_arg_val(4);
    constexpr uint32_t MIX_COL = get_compile_time_arg_val(5);
    constexpr uint32_t NB = KT / BK;
    static_assert(NB * BK == KT, "BK must divide KT");
    constexpr auto x_args = TensorAccessorArgs<6>();

    if (row_count == 0) {
        return;
    }

    const auto xs = TensorAccessor(x_args, x_addr);
    Noc noc;
    DataflowBuffer x(cb_x);
    DataflowBuffer xsq(cb_xsq);
    DataflowBuffer consts(cb_const);
    const uint32_t page = get_local_cb_interface(cb_x).fifo_page_size;  // fp32 tile
    constexpr uint32_t TILE_U32 = 32 * 32;
    constexpr uint32_t FACE_U32 = 16 * 16;
    constexpr uint32_t ONE = 0x3F800000u;

    consts.reserve_back(1);
    tt_l1_ptr uint32_t* c = reinterpret_cast<tt_l1_ptr uint32_t*>(consts.get_write_ptr());
    for (uint32_t i = 0; i < TILE_U32; ++i) {
        c[i] = 0;
    }
    for (uint32_t r = 0; r < 32; ++r) {
        c[(r >> 4) * 2 * FACE_U32 + (r & 15u) * 16 + (MIX_COL >> 4) * FACE_U32 + (MIX_COL & 15u)] =
            ONE;  // E[r, MIX_COL]
    }
    asm volatile("" ::: "memory");  // the plain stores above complete before the tile is published
    consts.push_back(1);

    for (uint32_t r = row_start; r < row_start + row_count; ++r) {
        const uint32_t b0 = r % NB;
        for (uint32_t i = 0; i < NB; ++i) {
            const uint32_t k0 = ((b0 + i) % NB) * BK;
            x.reserve_back(BK);
            xsq.reserve_back(BK);
            for (uint32_t b = 0; b < BK; ++b) {
                noc.async_read(xs, x, page, {.page_id = r * KT + k0 + b}, {.offset_bytes = b * page});
            }
            noc.async_read_barrier();
            x.push_back(BK);
            xsq.push_back(BK);
        }
    }
}
