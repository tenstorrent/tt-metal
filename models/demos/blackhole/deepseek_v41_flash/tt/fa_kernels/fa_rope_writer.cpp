// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused prefill attention glue, writer: the finished block (16 tiles, or 32 row-major rows of 1 KiB after the in-kernel
// untilize) goes to
//   OUT tile mode: page u*OUT_U + h*OUT_H + tr_l*OUT_TR + OUT_COL0 + j of a tile tensor
//   OUT rm mode  : page h*C + tr_l*32 + r of the user's own [1,32,C,512] row-major buffer.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t OUT_RM = get_compile_time_arg_val(1);
    constexpr uint32_t NH = get_compile_time_arg_val(2);
    constexpr uint32_t CB32 = get_compile_time_arg_val(3);
    constexpr uint32_t C = get_compile_time_arg_val(4);
    constexpr uint32_t OUT_U = get_compile_time_arg_val(5);
    constexpr uint32_t OUT_H = get_compile_time_arg_val(6);
    constexpr uint32_t OUT_TR = get_compile_time_arg_val(7);
    constexpr uint32_t OUT_COL0 = get_compile_time_arg_val(8);
    constexpr uint32_t NT = get_compile_time_arg_val(9);
    constexpr auto o_args = TensorAccessorArgs<10>();

    const uint32_t b0 = get_arg_val<uint32_t>(0);
    const uint32_t nb = get_arg_val<uint32_t>(1);
    Noc noc;
    experimental::CB out(cb_out);
    for (uint32_t b = b0; b < b0 + nb; ++b) {
        const uint32_t tr_g = b / NH;
        const uint32_t h = b - tr_g * NH;
        const uint32_t u = tr_g / CB32;
        const uint32_t tr_l = tr_g - u * CB32;
        out.wait_front(NT);
        if constexpr (OUT_RM) {
            const uint32_t o_addr = get_common_arg_val<uint32_t>(u);
            const auto o_acc = TensorAccessor(o_args, o_addr, 1024);
            const uint32_t row0 = h * C + tr_l * 32;
            for (uint32_t r = 0; r < 32; ++r) {
                noc.async_write(out, o_acc, 1024, {.offset_bytes = r * 1024}, {.page_id = row0 + r, .offset_bytes = 0});
            }
        } else {
            const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
            const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
            const uint32_t base = u * OUT_U + h * OUT_H + tr_l * OUT_TR + OUT_COL0;
            for (uint32_t j = 0; j < NT; ++j) {
                noc.async_write(out, o_acc, 2048, {.offset_bytes = j * 2048}, {.page_id = base + j, .offset_bytes = 0});
            }
        }
        noc.async_write_barrier();
        out.pop_front(NT);
    }
}
