// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Final mixer stage 3a reader: this core's T column tiles (first..first+T-1 of the W-tile-wide partial row) of the
// D gathered fp32 partial rows [D, 1, 1, 32 W] (pages d * W + first + t) -> T stacked fp32 tiles in CB 0: tile t row d
// = partial d's row 0 of column tile t (columns 0-15 = face 0 row d, columns 16-31 = face 1 row d), rows D..31 = 0.0
// (the transpose's zero padding).  NoC only: each stacked tile is a copy of the zero tile, then the D x 2 face rows (64
// bytes each, DRAM-aligned at tile offsets 0 and 1024) land at their row offsets.  Then the reduce scaler tile (1.0 in
// the CB's fp32 format, the chain's prepare_reduce_scaler for REDUCE_COL) into CB 1. Compile-time args: 0 T, 1 D, 2 W,
// then TensorAccessorArgs(partials), (zero tile).  Runtime args: 0 partials addr, 1 zero-tile addr, 2 first (this
// core's first column tile).  One core takes T = W, first = 0; the merged low_rank_gate program runs it on its row
// cores with T = W / cores.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "../../kernels/zones.h"

constexpr uint32_t T = get_compile_time_arg_val(0);
constexpr uint32_t D = get_compile_time_arg_val(1);
constexpr uint32_t W = get_compile_time_arg_val(2);
constexpr uint32_t TILE_BYTES = 4096;  // fp32 32x32
constexpr uint32_t FACE_BYTES = 1024;
constexpr uint32_t ROW_BYTES = 64;  // 16 fp32 = one face row
constexpr uint32_t c_st = 0, c_scaler = 1;

void kernel_main() {
    constexpr auto a_p = TensorAccessorArgs<3>();
    constexpr auto a_z = TensorAccessorArgs<a_p.next_compile_time_args_offset()>();
    const auto partials = TensorAccessor(a_p, get_arg_val<uint32_t>(0));
    const auto zero = TensorAccessor(a_z, get_arg_val<uint32_t>(1));
    const uint32_t first = get_arg_val<uint32_t>(2);
    Noc noc;
    DataflowBuffer st(c_st);
    {
        FUSED_ZONE("fz_fm_lr_r_zero");
        st.reserve_back(T);
        for (uint32_t t = 0; t < T; ++t) {
            noc.async_read(zero, st, TILE_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = t * TILE_BYTES});
        }
        noc.async_read_barrier();  // the zero copies land before the rows they must not overwrite
    }
    {
        FUSED_ZONE("fz_fm_lr_r_rows");
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t d = 0; d < D; ++d) {
                const uint32_t tile = t * TILE_BYTES + d * ROW_BYTES;  // row d of face 0; face 1 is +FACE_BYTES
                const uint32_t page = d * W + first + t;
                noc.async_read(partials, st, ROW_BYTES, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = tile});
                noc.async_read(
                    partials,
                    st,
                    ROW_BYTES,
                    {.page_id = page, .offset_bytes = FACE_BYTES},
                    {.offset_bytes = tile + FACE_BYTES});
            }
        }
        noc.async_read_barrier();
        st.push_back(T);
    }
    dataflow_kernel_lib::prepare_reduce_scaler<c_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(1.0f);
}
