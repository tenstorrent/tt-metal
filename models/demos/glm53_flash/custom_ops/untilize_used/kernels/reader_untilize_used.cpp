// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the used-rows untilize (tt-blaze G53F-0203): the regions MoE's expert output [R, H] (bf8 TILE) holds the
// local regions in its first ``run`` rows (the regions are packed from row 0 in expert order), run = offsets[E-1] +
// align32(counts[E-1]); the rows past it are never read by the combine (except the dummy row R-1, the writer's). This
// core's share of the used tile rows: an even split of run / 32 over the NCORES cores, so every core works however few
// rows are used. The tile-row count goes to the compute kernel through a one-page CB; then each tile row's Wt tiles are
// read into the input CB, all in flight, one barrier per row.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

constexpr uint32_t E = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t NCORES = get_compile_time_arg_val(2);
constexpr uint32_t cb_in_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_cnt_id = get_compile_time_arg_val(4);
constexpr uint32_t cb_scr_id = get_compile_time_arg_val(5);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
constexpr uint32_t table_bytes = get_compile_time_arg_val(7);
constexpr auto in_args = TensorAccessorArgs<8>();
constexpr auto counts_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
constexpr auto offsets_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
// the grid width: a core's index is its logical position (row-major), so no per-core runtime args (G53F-0206)
constexpr uint32_t GRID_X = get_compile_time_arg_val(offsets_args.next_compile_time_args_offset());

void kernel_main() {
    // the buffer addresses are common runtime args (one list for every core), the core's index from its logical
    // position (G53F-0206)
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t offsets_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t core_idx = get_absolute_logical_y() * GRID_X + get_absolute_logical_x();

    const auto s_in = TensorAccessor(in_args, in_addr, tile_bytes);
    const auto s_counts = TensorAccessor(counts_args, counts_addr, table_bytes);
    const auto s_offsets = TensorAccessor(offsets_args, offsets_addr, table_bytes);

    Noc noc;
    CircularBuffer cb_in(cb_in_id), cb_cnt(cb_cnt_id), cb_scr(cb_scr_id);
    cb_scr.reserve_back(1);  // private scratch, reserved once and never pushed
    const uint32_t counts_l1 = get_write_ptr(cb_scr_id), offsets_l1 = counts_l1 + table_bytes;
    noc.async_read(s_counts, CoreLocalMem<uint32_t>(counts_l1), table_bytes, {.page_id = 0}, {});
    noc.async_read(s_offsets, CoreLocalMem<uint32_t>(offsets_l1), table_bytes, {.page_id = 0}, {});
    noc.async_read_barrier();
    const CoreLocalMem<volatile uint32_t> counts(counts_l1), offsets(offsets_l1);
    const uint32_t run = offsets[E - 1] + ((counts[E - 1] + 31u) & ~31u);
    const uint32_t used_tr = run / 32u;
    const uint32_t base = used_tr / NCORES, extra = used_tr % NCORES;
    const uint32_t n = base + (core_idx < extra ? 1u : 0u);
    const uint32_t b0 = core_idx * base + (core_idx < extra ? core_idx : extra);

    cb_cnt.reserve_back(1);
    CoreLocalMem<volatile uint32_t>(get_write_ptr(cb_cnt_id))[0] = n;
    cb_cnt.push_back(1);

    for (uint32_t tr = b0; tr < b0 + n; ++tr) {
        cb_in.reserve_back(Wt);
        const uint32_t l1 = get_write_ptr(cb_in_id);
        for (uint32_t c = 0; c < Wt; ++c) {
            noc.async_read(s_in, CoreLocalMem<uint32_t>(l1 + c * tile_bytes), tile_bytes, {.page_id = tr * Wt + c}, {});
        }
        noc.async_read_barrier();
        cb_in.push_back(Wt);
    }
}
