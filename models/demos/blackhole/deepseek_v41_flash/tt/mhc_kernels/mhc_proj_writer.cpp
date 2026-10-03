// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_p = get_compile_time_arg_val(0);
    constexpr auto p_args = TensorAccessorArgs<1>();
    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);
    Noc noc;
    const auto p_acc = TensorAccessor(p_args, p_addr, 4096);
    experimental::CB p(cb_p);
    p.wait_front(1);
    noc.async_write(p, p_acc, 4096, {.offset_bytes = 0}, {.page_id = k, .offset_bytes = 0});
    noc.async_write_barrier();
    p.pop_front(1);
}
