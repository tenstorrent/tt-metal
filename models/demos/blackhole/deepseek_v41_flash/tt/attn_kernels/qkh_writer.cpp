// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// qk_heads writer of the RoPE cores (n = 14, 15): the rotated tiles -> out pages t*16+n.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    Noc noc;
    const auto o_acc = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2), 2048);
    experimental::CB out(cb_o);
    for (uint32_t i = 0; i < nt; ++i) {
        out.wait_front(i + 1);
        noc.async_write(
            out, o_acc, 2048, {.offset_bytes = i * 2048}, {.page_id = (t0 + i) * 16 + n, .offset_bytes = 0});
    }
    noc.async_write_barrier();
    out.pop_front(nt);
}
