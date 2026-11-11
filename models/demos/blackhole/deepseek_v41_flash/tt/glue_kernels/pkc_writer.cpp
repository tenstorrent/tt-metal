// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// T19 packed-mHC collapse (+ sum of squares), writer: bf16 output tile (r, c) -> page r*CT + c; partial tile -> page
// r*NRC + p.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_p = get_compile_time_arg_val(1);
    constexpr uint32_t CT = get_compile_time_arg_val(2);
    constexpr uint32_t NRC = get_compile_time_arg_val(3);
    constexpr uint32_t BLK = get_compile_time_arg_val(4);
    constexpr auto o_args = TensorAccessorArgs<5>();
    constexpr auto p_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(1);
    const uint32_t ncols = get_arg_val<uint32_t>(2);
    const uint32_t p = get_arg_val<uint32_t>(3);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
    const auto p_acc = TensorAccessor(p_args, p_addr, 4096);
    experimental::CB out(cb_out), pp(cb_p);
    for (uint32_t b = 0; b < ncols; b += BLK) {
        for (uint32_t ci = 0; ci < BLK; ++ci) {
            out.wait_front(1);
            noc.async_write(
                out, o_acc, 2048, {.offset_bytes = 0}, {.page_id = r * CT + c0 + b + ci, .offset_bytes = 0});
            noc.async_writes_flushed();
            out.pop_front(1);
        }
    }
    pp.wait_front(1);
    noc.async_write(pp, p_acc, 4096, {.offset_bytes = 0}, {.page_id = r * NRC + p, .offset_bytes = 0});
    noc.async_writes_flushed();
    pp.pop_front(1);
    noc.async_write_barrier();
}
