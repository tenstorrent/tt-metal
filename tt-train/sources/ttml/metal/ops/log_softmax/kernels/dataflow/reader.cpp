// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tile_constants.h"

void kernel_main() {
    constexpr uint32_t wt = get_compile_time_arg_val(0), block = get_compile_time_arg_val(1);
    constexpr uint32_t th = get_compile_time_arg_val(2), tw = get_compile_time_arg_val(3);
    const uint32_t x = get_arg_val<uint32_t>(0), dy = get_arg_val<uint32_t>(1), state = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3), start = get_arg_val<uint32_t>(4);
    constexpr auto xa = TensorAccessorArgs<4>();
    const auto input = TensorAccessor(xa, x);
#ifdef BACKWARD
    constexpr auto da = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    const auto grad = TensorAccessor(da, dy);
    const auto stats = TensorAccessor(sa, state);
    constexpr uint32_t passes = 3;
#else
    constexpr uint32_t passes = 4;
#endif
    const Noc noc;
    DataflowBuffer xb(0), db(1), columns(2), sb(9);
    const uint32_t bytes = get_tile_size(0), grad_bytes = get_tile_size(1), stat_bytes = get_tile_size(9);
    columns.reserve_back(1);
    auto ptr = reinterpret_cast<volatile tt_l1_ptr float *>(columns.get_write_ptr());
    constexpr uint32_t fh = tt::constants::FACE_HEIGHT, fw = tt::constants::FACE_WIDTH;
    for (uint32_t f = 0; f < 4; ++f)
        for (uint32_t r = 0; r < fh; ++r)
            for (uint32_t c = 0; c < fw; ++c) ptr[f * fh * fw + r * fw + c] = static_cast<float>((f % 2) * fw + c);
    columns.push_back(1);
    for (uint32_t row = start; row < start + count; ++row) {
#ifdef BACKWARD
        sb.reserve_back(3);
        for (uint32_t i = 0; i < 3; ++i)
            noc.async_read(stats, sb, stat_bytes, {.page_id = row * 3 + i}, {.offset_bytes = i * stat_bytes});
        noc.async_read_barrier();
        sb.push_back(3);
#endif
        for (uint32_t p = 0; p < passes; ++p)
            for (uint32_t col = 0; col < wt; col += block) {
                xb.reserve_back(block);
#ifdef BACKWARD
                db.reserve_back(block);
#endif
                for (uint32_t i = 0; i < block; ++i) {
                    noc.async_read(input, xb, bytes, {.page_id = row * wt + col + i}, {.offset_bytes = i * bytes});
#ifdef BACKWARD
                    noc.async_read(
                        grad, db, grad_bytes, {.page_id = row * wt + col + i}, {.offset_bytes = i * grad_bytes});
#endif
                }
                noc.async_read_barrier();
                xb.push_back(block);
#ifdef BACKWARD
                db.push_back(block);
#endif
            }
    }
}
