// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Packed-layout mHC combine, writer: output tile (r, stream j, column tile c) -> page r*NOUT*CT + j*CT + c.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr uint32_t NOUT = get_compile_time_arg_val(2);
    constexpr uint32_t OTILE = get_compile_time_arg_val(3);
    constexpr uint32_t BLK = get_compile_time_arg_val(4);
    constexpr auto o_args = TensorAccessorArgs<5>();
    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(1);
    const uint32_t ncols = get_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);

    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, OTILE);
    experimental::CB out(cb_out);
    for (uint32_t b = 0; b < ncols; b += BLK) {
        for (uint32_t ci = 0; ci < BLK; ++ci) {
            out.wait_front(NOUT);
            for (uint32_t j = 0; j < NOUT; ++j) {
                noc.async_write(
                    out,
                    o_acc,
                    OTILE,
                    {.offset_bytes = j * OTILE},
                    {.page_id = r * NOUT * CT + j * CT + c0 + b + ci, .offset_bytes = 0});
            }
            noc.async_writes_flushed();
            out.pop_front(NOUT);
        }
    }
    noc.async_write_barrier();
}
