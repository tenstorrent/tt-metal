// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t XS = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    const uint32_t w = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t t = w >> 1, j = w & 1;
    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
    experimental::CB out(cb_out);
    out.wait_front(1);
    noc.async_write(out, o_acc, 2048, {.offset_bytes = 0}, {.page_id = t * XS + 14 + j, .offset_bytes = 0});
    noc.async_write_barrier();
    out.pop_front(1);
}
