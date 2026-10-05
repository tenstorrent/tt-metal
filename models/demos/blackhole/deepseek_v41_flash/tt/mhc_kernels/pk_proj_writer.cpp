// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// partial tile (r, k) -> page r*S + k of the [R, S, 32, 32] partial tensor

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_p = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr auto p_args = TensorAccessorArgs<3>();
    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);
    Noc noc;
    const auto p_acc = TensorAccessor(p_args, p_addr, 4096);
    experimental::CB p(cb_p);
    for (uint32_t r = 0; r < R; ++r) {
        p.wait_front(1);
        noc.async_write(p, p_acc, 4096, {.offset_bytes = 0}, {.page_id = r * S + k, .offset_bytes = 0});
        noc.async_writes_flushed();
        p.pop_front(1);
    }
    noc.async_write_barrier();
}
