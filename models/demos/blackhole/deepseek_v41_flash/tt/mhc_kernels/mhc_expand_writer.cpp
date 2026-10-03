// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t T = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    constexpr uint32_t TILE = 4096;

    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t j1 = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);

    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, TILE);
    experimental::CB out(cb_out);
    const uint32_t nb = (j1 - j0) * T;
    out.wait_front(nb);
    uint32_t k = 0;
    for (uint32_t j = j0; j < j1; ++j) {
        for (uint32_t t = 0; t < T; ++t, ++k) {
            noc.async_write(out, o_acc, TILE, {.offset_bytes = k * TILE}, {.page_id = t * NT + j, .offset_bytes = 0});
        }
    }
    noc.async_write_barrier();
    out.pop_front(nb);
}
