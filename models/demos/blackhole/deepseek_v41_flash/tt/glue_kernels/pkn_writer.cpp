// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// T19 packed-mHC norm apply, writer: bf16 tile (r, c0 + j) -> page r*CT + c0 + j.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(1);
    const uint32_t cn = get_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
    experimental::CB out(cb_out);
    for (uint32_t j = 0; j < cn; ++j) {
        out.wait_front(1);
        noc.async_write(out, o_acc, 2048, {.offset_bytes = 0}, {.page_id = r * CT + c0 + j, .offset_bytes = 0});
        noc.async_writes_flushed();
        out.pop_front(1);
    }
    noc.async_write_barrier();
}
