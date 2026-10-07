// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t blk0 = get_arg_val<uint32_t>(0);
    const uint32_t nblk = get_arg_val<uint32_t>(1);
    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
    experimental::CB out(cb_out);
    for (uint32_t b = 0; b < nblk; ++b) {
        out.wait_front(NB);
        for (uint32_t j = 0; j < NB; ++j) {
            noc.async_write(
                out, o_acc, 2048, {.offset_bytes = j * 2048}, {.page_id = (blk0 + b) * NB + j, .offset_bytes = 0});
        }
        noc.async_write_barrier();
        out.pop_front(NB);
    }
}
