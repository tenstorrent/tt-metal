// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the fused sLSTM loop. Every step and item (d-tile j of unit i / 4): once compute has packed the
// item's y, bump the local semaphore of the unit, copy the bf16 y tile into the same y-buffer slot of every
// other core touching the unit and bump theirs, then store the fp32 y tile (and c, n, m after the last step,
// or after every step with record) to DRAM.
//
// ys [T, 1, B, 512] page (t * Bt + row) * 16 + h * 4 + j; states likewise with t = 0 unless record.
//
// Compile-time args: TensorAccessorArgs(ys), TensorAccessorArgs(states)
// Runtime args: ys_addr, c_addr, n_addr, m_addr, num_items, first_item, T, record, Bt, ablate,
//   then per item: num_dests, 3 x (noc_x, noc_y, dest unit slot)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t ys_addr = get_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_addr = get_arg_val<uint32_t>(2);
    const uint32_t m_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_items = get_arg_val<uint32_t>(4);
    const uint32_t first_item = get_arg_val<uint32_t>(5);
    const uint32_t T = get_arg_val<uint32_t>(6);
    const bool record = get_arg_val<uint32_t>(7) != 0;
    const uint32_t Bt = get_arg_val<uint32_t>(8);
    // Timing ablation only: bit 8 skips the DRAM y write, bit 9 the peer y writes + incs (pair with bit 7).
    const uint32_t ablate = get_arg_val<uint32_t>(9);
    const bool no_dram_y = ablate & 256, no_peer = ablate & 512;
    constexpr uint32_t dest_args = 10, dest_stride = 10;

    constexpr auto ys_args = TensorAccessorArgs<0>();
    constexpr auto s_args = TensorAccessorArgs<ys_args.next_compile_time_args_offset()>();
    constexpr uint32_t yout_cb_id = tt::CBIndex::c_16;
    constexpr uint32_t y_cb_id = tt::CBIndex::c_2;

    Noc noc;
    CircularBuffer cb_yout(yout_cb_id);
    CircularBuffer cb_s[3] = {CircularBuffer(tt::CBIndex::c_17), CircularBuffer(tt::CBIndex::c_18), CircularBuffer(tt::CBIndex::c_19)};
    const uint32_t bytes = get_tile_size(yout_cb_id);
    const uint32_t y_bytes = get_tile_size(y_cb_id);
    const uint32_t y_base = get_write_ptr(y_cb_id);  // same address on every core
    const auto ys = TensorAccessor(ys_args, ys_addr);
    const auto sc = TensorAccessor(s_args, c_addr);
    const auto sn = TensorAccessor(s_args, n_addr);
    const auto sm = TensorAccessor(s_args, m_addr);
    const uint32_t u0 = first_item / 4;

    for (uint32_t t = 0; t < T; ++t) {
        const bool emit = record || t + 1 == T;
        const uint32_t ts = record ? t : 0;
        for (uint32_t e = 0; e < num_items; ++e) {
            const uint32_t i = first_item + e;
            const uint32_t unit = i / 4;
            const uint32_t j = i % 4;
            const uint32_t row = unit / 4;
            const uint32_t h = unit % 4;
            const uint32_t a = dest_args + e * dest_stride;
            const uint32_t nd = get_arg_val<uint32_t>(a);
            const uint32_t slot = (t % 2) * 4 + j;  // tile within a unit's 8-tile y block
            const uint32_t y_addr = y_base + ((unit - u0) * 8 + slot) * y_bytes;

            cb_yout.wait_front(1);
            noc_semaphore_inc(get_noc_addr(my_x[noc_index], my_y[noc_index], get_semaphore(unit - u0)), 1);
            for (uint32_t d = 0; d < (no_peer ? 0 : nd); ++d) {
                const uint32_t x = get_arg_val<uint32_t>(a + 1 + 3 * d), y = get_arg_val<uint32_t>(a + 2 + 3 * d);
                const uint32_t lu = get_arg_val<uint32_t>(a + 3 + 3 * d);
                noc_async_write(y_addr, get_noc_addr(x, y, y_base + (lu * 8 + slot) * y_bytes), y_bytes);
            }
            // One barrier + one inc per (unit, peer) after this core's last own tile of the unit.
            const bool last_of_unit = e + 1 == num_items || (i + 1) / 4 != unit;
            if (nd && last_of_unit && !no_peer) {
                const uint32_t own = i + 1 - (first_item > 4 * unit ? first_item : 4 * unit);  // own tiles of the unit here
                noc_async_write_barrier();
                for (uint32_t d = 0; d < nd; ++d) {
                    const uint32_t x = get_arg_val<uint32_t>(a + 1 + 3 * d), y = get_arg_val<uint32_t>(a + 2 + 3 * d);
                    const uint32_t lu = get_arg_val<uint32_t>(a + 3 + 3 * d);
                    noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(lu)), own);
                }
            }
            if (!no_dram_y) noc.async_write(cb_yout, ys, bytes, {.offset_bytes = 0}, {.page_id = (t * Bt + row) * 16 + h * 4 + j});
            if (emit) {
                const uint32_t page = (ts * Bt + row) * 16 + h * 4 + j;
                cb_s[0].wait_front(1);
                cb_s[1].wait_front(1);
                cb_s[2].wait_front(1);
                noc.async_write(cb_s[0], sc, bytes, {.offset_bytes = 0}, {.page_id = page});
                noc.async_write(cb_s[1], sn, bytes, {.offset_bytes = 0}, {.page_id = page});
                noc.async_write(cb_s[2], sm, bytes, {.offset_bytes = 0}, {.page_id = page});
            }
            noc.async_writes_flushed();
            cb_yout.pop_front(1);
            if (emit) {
                cb_s[0].pop_front(1);
                cb_s[1].pop_front(1);
                cb_s[2].pop_front(1);
            }
        }
    }
    noc.async_write_barrier();
    noc_async_atomic_barrier();
}
