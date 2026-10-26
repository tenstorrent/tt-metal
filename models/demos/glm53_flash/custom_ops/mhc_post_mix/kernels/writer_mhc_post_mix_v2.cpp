// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per position p = r * Ct + c: writes the n output tiles out[r, j, c] (page (r*n + j)*Ct + c), or
// the single collapsed tile out[r, c] (page r*Ct + c).
void kernel_main() {
    // the output address is a common runtime arg, the work split from the core's position (G53F-0206)
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);
    constexpr uint32_t Ct = get_compile_time_arg_val(2);
    constexpr bool collapse = get_compile_time_arg_val(3) != 0;
    constexpr auto out_args = TensorAccessorArgs<4>();
    constexpr uint32_t TOTAL = get_compile_time_arg_val(out_args.next_compile_time_args_offset());
    constexpr uint32_t NCORES = get_compile_time_arg_val(out_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t GRID_Y = get_compile_time_arg_val(out_args.next_compile_time_args_offset() + 2);
    // the core's span of the TOTAL positions: split_work's deal (column-major, the first TOTAL % NCORES cores one more)
    // from the core's logical position -- no per-core runtime args (G53F-0206)
    const uint32_t core_i = get_absolute_logical_x() * GRID_Y + get_absolute_logical_y();
    const uint32_t base_ = TOTAL / NCORES, rem_ = TOTAL % NCORES;
    const uint32_t num_pos = base_ + (core_i < rem_ ? 1u : 0u);
    const uint32_t start_pos = core_i * base_ + (core_i < rem_ ? core_i : rem_);
    constexpr uint32_t tiles_per_pos = collapse ? 1 : n;

    const uint32_t out_page = get_local_cb_interface(cb_out).fifo_page_size;
    const auto s_out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer cbo(cb_out);

    for (uint32_t p = start_pos; p < start_pos + num_pos; ++p) {
        const uint32_t r = p / Ct;
        const uint32_t c = p - r * Ct;
        cbo.wait_front(tiles_per_pos);
        for (uint32_t j = 0; j < tiles_per_pos; ++j) {
            const uint32_t page = collapse ? (r * Ct + c) : ((r * n + j) * Ct + c);
            noc.async_write(cbo, s_out, out_page, {.offset_bytes = j * out_page}, {.page_id = page});
        }
        noc.async_write_barrier();
        cbo.pop_front(tiles_per_pos);
    }
}
