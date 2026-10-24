// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// qwen_width_norm writer: per tile row, send this core's partial to slot cx of half (r % 2) of CB_GATH on every
// core of the row group (itself included), then bump their SEM_RECV; then write the row's output tiles.
//
// CB_GATH is declared on every core of the program, so it has one L1 address on all of them; its base is read
// here before any use (this kernel never reserves or pushes it).
//
// Ordering: a peer can only send row r + 2 after it has received this core's row r + 1 partial, which this
// compute produces after it consumed (popped) row r's gathered partials, so a half is never overwritten while it
// is read.
//
// Compile-time args: wnorm_common.hpp prefix, accessor out, then the NOC x of the gx grid columns and the NOC y of
// the gy grid rows. Common runtime args: 0 out_addr.

#include <stdint.h>
#include <utility>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "wnorm_common.hpp"

using namespace wnorm;

template <uint32_t OFF, uint32_t... I>
FORCE_INLINE void fill_ct(uint32_t* dst, std::integer_sequence<uint32_t, I...>) {
    ((dst[I] = get_compile_time_arg_val(OFF + I)), ...);
}

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t gx = get_compile_time_arg_val(1);
    constexpr uint32_t gy = get_compile_time_arg_val(2);
    constexpr uint32_t Rt = get_compile_time_arg_val(3);
    constexpr auto o_args = TensorAccessorArgs<CT_NEXT>();
    constexpr uint32_t NOC_OFF = o_args.next_compile_time_args_offset();  // gx NOC x, then gy NOC y

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t cx = get_absolute_logical_x();
    const uint32_t cy = get_absolute_logical_y();
    const Work wk = core_work(cx, cy, Wt, gx, Rt, gy);
    const uint32_t row0 = wk.row0, num_rows = wk.n, c0 = wk.c0, w = wk.w;
    uint32_t px[gx];
    fill_ct<NOC_OFF>(px, std::make_integer_sequence<uint32_t, gx>{});
    uint32_t pys[gy];
    fill_ct<NOC_OFF + gx>(pys, std::make_integer_sequence<uint32_t, gy>{});
    const uint32_t py = pys[cy];
    const auto out = TensorAccessor(o_args, out_addr);

    constexpr uint32_t part_bytes = get_tile_size(CB_PART);
    static_assert(get_tile_size(CB_PART) == get_tile_size(CB_GATH), "CB_PART and CB_GATH tiles must match");
    constexpr uint32_t out_bytes = get_tile_size(CB_OUT);
    const uint32_t gath_base = get_write_ptr(CB_GATH);
    const uint32_t sem_addr = get_semaphore(SEM_RECV);

    for (uint32_t r = 0; r < num_rows; ++r) {
        cb_wait_front(CB_PART, 1);
        const uint32_t src = get_read_ptr(CB_PART);
        const uint32_t slot = gath_base + ((r & 1) * gx + cx) * part_bytes;
        for (uint32_t k = 0; k < gx; ++k) {
            noc_async_write(src, get_noc_addr(px[k], py, slot), part_bytes);
        }
        noc_async_write_barrier();
        for (uint32_t k = 0; k < gx; ++k) {
            noc_semaphore_inc(get_noc_addr(px[k], py, sem_addr), 1);
        }
        cb_pop_front(CB_PART, 1);

        uint32_t page = (row0 + r) * Wt + c0;
        for (uint32_t t = 0; t < w; t += DB) {
            cb_wait_front(CB_OUT, DB);
            uint32_t rd = get_read_ptr(CB_OUT);
            for (uint32_t i = 0; i < DB; ++i) {
                noc_async_write_page(page + i, out, rd);
                rd += out_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(CB_OUT, DB);
            page += DB;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
