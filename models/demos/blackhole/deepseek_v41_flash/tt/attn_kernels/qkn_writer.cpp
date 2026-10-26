// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// qk_norm, writer (BRISC): writes the gpc normalised bf16 tiles of this core to the q output (is_kv = 0) or the kv
// output, pages p0 ...

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto q_args = TensorAccessorArgs<1>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    const uint32_t is_kv = get_arg_val<uint32_t>(0);
    const uint32_t p0 = get_arg_val<uint32_t>(1);
    const uint32_t gpc = get_arg_val<uint32_t>(2);
    Noc noc;
    const auto q_acc = TensorAccessor(q_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto k_acc = TensorAccessor(k_args, get_common_arg_val<uint32_t>(1), 2048);
    experimental::CB out(cb_out);
    out.wait_front(gpc);
    for (uint32_t g = 0; g < gpc; ++g) {
        if (is_kv) {
            noc.async_write(out, k_acc, 2048, {.offset_bytes = g * 2048}, {.page_id = p0 + g, .offset_bytes = 0});
        } else {
            noc.async_write(out, q_acc, 2048, {.offset_bytes = g * 2048}, {.page_id = p0 + g, .offset_bytes = 0});
        }
    }
    noc.async_write_barrier();
    out.pop_front(gpc);
}
