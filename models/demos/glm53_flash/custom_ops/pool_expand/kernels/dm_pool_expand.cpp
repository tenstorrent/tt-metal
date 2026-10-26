// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The DSA prefill's index rows from the ranked pool ids in one data-movement pass (tt-blaze G53F-0210):
//   out[r, P*i + j] = P * top[r, i] + j   (i < K, j < P; a SENTINEL pool id gives P SENTINELs)
//   out[r, P*K + t] = tail[r, t]           (t < TAILN: the query's incomplete-pool token ids or SENTINELs)
//   out[r, P*K + TAILN ..] = SENTINEL      (padding to the row width W)
// which is what TtIndexer._expand_pools builds with typecast / repeat_interleave / multiply / add / typecast / concat
// (all exact on integers below 2^24). This core's rows: an even split of the T rows over NCORES cores, row-major core
// index from the logical position (no per-core runtime args, G53F-0206); a row's top ids and tail ids are read, the
// row is built in L1 and written while the next row is read (two output buffers).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

constexpr uint32_t T = get_compile_time_arg_val(0);
constexpr uint32_t NCORES = get_compile_time_arg_val(1);
constexpr uint32_t GRID_X = get_compile_time_arg_val(2);
constexpr uint32_t K = get_compile_time_arg_val(3);
constexpr uint32_t P = get_compile_time_arg_val(4);
constexpr uint32_t TAILN = get_compile_time_arg_val(5);
constexpr uint32_t W = get_compile_time_arg_val(6);
constexpr uint32_t cb_scr_id = get_compile_time_arg_val(7);
constexpr uint32_t top_page = get_compile_time_arg_val(8);
constexpr uint32_t tail_page = get_compile_time_arg_val(9);
constexpr uint32_t out_page = get_compile_time_arg_val(10);
constexpr auto top_args = TensorAccessorArgs<11>();
constexpr auto tail_args = TensorAccessorArgs<top_args.next_compile_time_args_offset()>();
constexpr auto out_args = TensorAccessorArgs<tail_args.next_compile_time_args_offset()>();
constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
static_assert(P * K + TAILN <= W, "the row holds the expanded pools and the tail");

void kernel_main() {
    const uint32_t top_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tail_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t core_idx = get_absolute_logical_y() * GRID_X + get_absolute_logical_x();
    const uint32_t base = T / NCORES, extra = T % NCORES;
    const uint32_t n = base + (core_idx < extra ? 1u : 0u);
    if (n == 0) {
        return;
    }
    const uint32_t r0 = core_idx * base + (core_idx < extra ? core_idx : extra);

    const auto s_top = TensorAccessor(top_args, top_addr, top_page);
    const auto s_tail = TensorAccessor(tail_args, tail_addr, tail_page);
    const auto s_out = TensorAccessor(out_args, out_addr, out_page);
    Noc noc;
    CircularBuffer cb_scr(cb_scr_id);
    cb_scr.reserve_back(1);  // private scratch, reserved once and never pushed
    const uint32_t top_l1 = get_write_ptr(cb_scr_id);
    const uint32_t tail_l1 = top_l1 + top_page;
    const uint32_t out_l1[2] = {tail_l1 + tail_page, tail_l1 + tail_page + out_page};
    const CoreLocalMem<volatile uint32_t> top(top_l1), tail(tail_l1);

    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t r = r0 + k;
        noc.async_read(s_top, CoreLocalMem<uint32_t>(top_l1), top_page, {.page_id = r}, {});
        noc.async_read(s_tail, CoreLocalMem<uint32_t>(tail_l1), tail_page, {.page_id = r}, {});
        noc.async_read_barrier();
        // the buffer written two rows ago must have left L1 (the previous row's write may still be in flight)
        noc.async_write_barrier();
        const CoreLocalMem<volatile uint32_t> out(out_l1[k & 1]);
        for (uint32_t i = 0; i < K; ++i) {
            const uint32_t v = top[i];
            const uint32_t b = v * P;
            for (uint32_t j = 0; j < P; ++j) {
                out[P * i + j] = v == SENTINEL ? SENTINEL : b + j;
            }
        }
        for (uint32_t t = 0; t < TAILN; ++t) {
            out[P * K + t] = tail[t];
        }
        for (uint32_t c = P * K + TAILN; c < W; ++c) {
            out[c] = SENTINEL;
        }
        noc.async_write(CoreLocalMem<uint32_t>(out_l1[k & 1]), s_out, W * 4u, {}, {.page_id = r});
    }
    noc.async_write_barrier();
}
