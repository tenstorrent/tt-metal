// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batched decode dense MoE gate|up + bias + SwiGLU + routing weights, writer (NCRISC) (dense_gu_reader.cpp): this
// core's act tiles n in order from cb_out to page n of the act tensor (width-sharded where the down matmul reads it).
//
// common runtime args: [act]; compile-time args: [cb_out, NA, NC, GX, accessor (act)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t act_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t NA = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t GX = get_compile_time_arg_val(3);
    constexpr auto act_args = TensorAccessorArgs<4>();
    constexpr uint32_t tile = 2048;  // BF16
    const auto s_act = TensorAccessor(act_args, act_addr, tile);

    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    for (uint32_t n = c * NA / NC; n < (c + 1) * NA / NC; ++n) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), s_act.get_noc_addr(n), tile);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
