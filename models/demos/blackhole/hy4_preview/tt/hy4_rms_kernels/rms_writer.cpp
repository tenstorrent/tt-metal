// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B47c fused MTP RMSNorm, writer (BRISC): writes the bf16 normalised tiles (page OOFF + j) and,
// if OUT_X, the fp32 sum x + residual (page j).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xo = get_compile_time_arg_val(1);
    constexpr uint32_t unused_ = get_compile_time_arg_val(2);
    constexpr uint32_t GPC = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_X = get_compile_time_arg_val(4);
    constexpr uint32_t OOFF = get_compile_time_arg_val(5);
    constexpr auto o_args = TensorAccessorArgs<6>();
    constexpr auto x_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t j0 = k * GPC;
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t xo_addr = get_common_arg_val<uint32_t>(1);
    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
    const auto x_acc = TensorAccessor(x_args, xo_addr, 4096);
    experimental::CB out(cb_out), xo(cb_xo);

    if constexpr (OUT_X) {
        xo.wait_front(GPC);
        for (uint32_t g = 0; g < GPC; ++g) {
            noc.async_write(xo, x_acc, 4096, {.offset_bytes = g * 4096}, {.page_id = j0 + g, .offset_bytes = 0});
        }
    }
    out.wait_front(GPC);
    for (uint32_t g = 0; g < GPC; ++g) {
        noc.async_write(out, o_acc, 2048, {.offset_bytes = g * 2048}, {.page_id = OOFF + j0 + g, .offset_bytes = 0});
    }
    noc.async_write_barrier();
    out.pop_front(GPC);
    if constexpr (OUT_X) {
        xo.pop_front(GPC);
    }
}
