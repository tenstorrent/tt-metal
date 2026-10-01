// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One 32x32 row tile per CB page (four 16x16 faces: rows 0-15 in faces 0,1; rows 16-31 in faces 2,3) -> the first
// `rows` rows of the tile as ROW_MAJOR output rows, for the tile columns [start, start + count).
// Compile-time args: 0 element bytes, 1 rows (1..32), 2.. TensorAccessorArgs(output).
// Runtime args: 0 output address, 1 tile count, 2 first tile column.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);

    constexpr uint32_t elem_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t face = 16;
    constexpr uint32_t face_bytes = face * face * elem_bytes;
    constexpr uint32_t half_row_bytes = face * elem_bytes;
    constexpr uint32_t tile_row_bytes = 2 * half_row_bytes;

    const auto dst = TensorAccessor(dst_args, dst_addr);
    Noc noc;
    DataflowBuffer dfb(cb_id);

    for (uint32_t tile = start; tile < start + count; ++tile) {
        FUSED_ZONE("fz_ur_w_tile");
        dfb.wait_front(1);
        const uint32_t col_bytes = tile * tile_row_bytes;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t left_face = (r >> 4) * 2;
            const uint32_t in_face = (r & (face - 1)) * half_row_bytes;
            noc.async_write(
                dfb,
                dst,
                half_row_bytes,
                {.offset_bytes = left_face * face_bytes + in_face},
                {.page_id = r, .offset_bytes = col_bytes});
            noc.async_write(
                dfb,
                dst,
                half_row_bytes,
                {.offset_bytes = (left_face + 1) * face_bytes + in_face},
                {.page_id = r, .offset_bytes = col_bytes + half_row_bytes});
        }
        noc.async_write_barrier();
        dfb.pop_front(1);
    }
}
