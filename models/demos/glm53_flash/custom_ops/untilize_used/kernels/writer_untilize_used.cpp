// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the used-rows untilize (tt-blaze G53F-0203): the same split of the used tile rows as the reader (from the
// same counts / offsets rows), each untilized block's 32 rows written as whole H-wide row pages; core 0 also writes
// the dummy row (R - 1) as zeros, the one row past ``run`` the combine reads (a token's pick 0 at the dummy slot, with
// weight 0: ttnn.untilize copied whatever the expert kernel's never-written buffer row held there).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

constexpr uint32_t E = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t NCORES = get_compile_time_arg_val(2);
constexpr uint32_t cb_out_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_scr_id = get_compile_time_arg_val(4);
constexpr uint32_t row_bytes = get_compile_time_arg_val(5);
constexpr uint32_t table_bytes = get_compile_time_arg_val(6);
constexpr uint32_t dummy_row = get_compile_time_arg_val(7);
constexpr auto out_args = TensorAccessorArgs<8>();
constexpr auto counts_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
constexpr auto offsets_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
// the grid width: a core's index is its logical position (row-major), so no per-core runtime args (G53F-0206)
constexpr uint32_t GRID_X = get_compile_time_arg_val(offsets_args.next_compile_time_args_offset());

void kernel_main() {
    // the buffer addresses are common runtime args (one list for every core), the core's index from its logical
    // position (G53F-0206)
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t offsets_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t core_idx = get_absolute_logical_y() * GRID_X + get_absolute_logical_x();

    const auto s_out = TensorAccessor(out_args, out_addr, row_bytes);
    const auto s_counts = TensorAccessor(counts_args, counts_addr, table_bytes);
    const auto s_offsets = TensorAccessor(offsets_args, offsets_addr, table_bytes);

    Noc noc;
    CircularBuffer cb_out(cb_out_id), cb_scr(cb_scr_id);
    cb_scr.reserve_back(1);  // private scratch, reserved once and never pushed
    const uint32_t counts_l1 = get_write_ptr(cb_scr_id), offsets_l1 = counts_l1 + table_bytes;
    const uint32_t zero_l1 = offsets_l1 + table_bytes;
    noc.async_read(s_counts, CoreLocalMem<uint32_t>(counts_l1), table_bytes, {.page_id = 0}, {});
    noc.async_read(s_offsets, CoreLocalMem<uint32_t>(offsets_l1), table_bytes, {.page_id = 0}, {});
    noc.async_read_barrier();
    const CoreLocalMem<volatile uint32_t> counts(counts_l1), offsets(offsets_l1);
    const uint32_t run = offsets[E - 1] + ((counts[E - 1] + 31u) & ~31u);
    const uint32_t used_tr = run / 32u;
    const uint32_t base = used_tr / NCORES, extra = used_tr % NCORES;
    const uint32_t n = base + (core_idx < extra ? 1u : 0u);
    const uint32_t b0 = core_idx * base + (core_idx < extra ? core_idx : extra);

    if (core_idx == 0) {
        const CoreLocalMem<volatile uint32_t> zero(zero_l1);
        for (uint32_t i = 0; i < row_bytes / 4u; ++i) {
            zero[i] = 0u;
        }
        noc.async_write(CoreLocalMem<uint32_t>(zero_l1), s_out, row_bytes, {}, {.page_id = dummy_row});
    }
    for (uint32_t tr = b0; tr < b0 + n; ++tr) {
        cb_out.wait_front(Wt);
        const uint32_t l1 = get_read_ptr(cb_out_id);
        for (uint32_t j = 0; j < 32u; ++j) {
            noc.async_write(
                CoreLocalMem<uint32_t>(l1 + j * row_bytes), s_out, row_bytes, {}, {.page_id = tr * 32u + j});
        }
        noc.async_write_barrier();
        cb_out.pop_front(Wt);
    }
    noc.async_write_barrier();
}
