// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_hw = get_compile_time_arg_val(0);
    constexpr uint32_t cb_p = get_compile_time_arg_val(1);
    constexpr auto h_args = TensorAccessorArgs<2>();
    constexpr auto p_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();

    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t j1 = get_arg_val<uint32_t>(1);
    const uint32_t k = get_arg_val<uint32_t>(2);
    const uint32_t h_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto h_acc = TensorAccessor(h_args, h_addr, 2048);
    const auto p_acc = TensorAccessor(p_args, p_addr, 4096);
    experimental::CB hw(cb_hw), p(cb_p);
    const uint32_t ng = j1 - j0;
    hw.wait_front(ng);
    for (uint32_t g = 0; g < ng; ++g) {
        noc.async_write(hw, h_acc, 2048, {.offset_bytes = g * 2048}, {.page_id = j0 + g, .offset_bytes = 0});
    }
    p.wait_front(1);
    noc.async_write(p, p_acc, 4096, {.offset_bytes = 0}, {.page_id = k, .offset_bytes = 0});
    noc.async_write_barrier();
    hw.pop_front(ng);
    p.pop_front(1);
}
