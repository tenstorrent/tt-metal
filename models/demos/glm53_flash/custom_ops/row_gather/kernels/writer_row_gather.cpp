// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// row_gather writer: drains the reader's batches into the output rows [row0, row0 + nrows) (one page per row).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row0 = get_arg_val<uint32_t>(1);
    const uint32_t nrows = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_rows = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    const auto s_out = TensorAccessor(out_args, out_addr, row_bytes);
    Noc noc;
    CircularBuffer cbr(cb_rows);
    for (uint32_t done = 0; done < nrows; done += batch) {
        const uint32_t nb = (nrows - done) < batch ? (nrows - done) : batch;
        cbr.wait_front(batch);
        for (uint32_t j = 0; j < nb; j++) {
            noc.async_write(cbr, s_out, row_bytes, {.offset_bytes = j * row_bytes}, {.page_id = row0 + done + j});
        }
        noc.async_write_barrier();
        cbr.pop_front(batch);
    }
}
