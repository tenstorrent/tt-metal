// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t TPC = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    Noc noc;
    const auto oA = TensorAccessor(o_args, o_addr, 2048);
    experimental::CB co(cb_o);
    co.wait_front(TPC);
    for (uint32_t j = 0; j < TPC; ++j) {
        noc.async_write(co, oA, 2048, {.offset_bytes = j * 2048}, {.page_id = j0 + j, .offset_bytes = 0});
    }
    noc.async_write_barrier();
    co.pop_front(TPC);
}
