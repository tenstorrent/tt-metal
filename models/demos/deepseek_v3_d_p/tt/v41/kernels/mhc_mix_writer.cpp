// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the fused V4.1 mHC stream mix (mhc_mix_compute.cpp): the J output tiles of unit (r, c) go to
// (r, j * CT + c) of the [T, J * C] output (fp32 or bf16: the page size is cb_out's).
//
// compile_time_args = [cb_out, CT, J, BLOCK, TensorAccessorArgs(out)...]  (BLOCK units per write flush)
// runtime args      = [out_addr, unit_start, unit_count]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr uint32_t J = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();

    const auto out = TensorAccessor(out_args, out_addr);
    Noc noc;
    DataflowBuffer dfb(cb_out);
    const uint32_t page = get_local_cb_interface(cb_out).fifo_page_size;
    const uint32_t unit_end = unit_start + unit_count;
    for (uint32_t u = unit_start; u < unit_end;) {
        constexpr uint32_t units = BLOCK;  // the host aligns every core's range to whole blocks
        dfb.wait_front(units * J);
        for (uint32_t b = 0; b < units; ++b) {
            const uint32_t r = (u + b) / CT;
            const uint32_t c = (u + b) - r * CT;
            for (uint32_t j = 0; j < J; ++j) {
                noc.async_write(
                    dfb, out, page, {.offset_bytes = (b * J + j) * page}, {.page_id = r * (J * CT) + j * CT + c});
            }
        }
        noc.async_writes_flushed();
        dfb.pop_front(units * J);
        u += units;
    }
    noc.async_write_barrier();
}
