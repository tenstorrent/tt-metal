// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash drafter MLP, writer (BRISC) (draft_silu_mul_reader.cpp): this core's NH act tiles into the L1-
// interleaved act [rows, NT tiles] (core (j, m): tiles (m, j NH + t)), the down matmul's input.
// common runtime args: [act_addr]; compile-time args: [cb_act, NH, NT, accessor (act)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_act = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr auto act_args = TensorAccessorArgs<3>();
    constexpr uint32_t tile = 2048;
    const auto s_act = TensorAccessor(act_args, get_common_arg_val<uint32_t>(0), tile);
    const uint32_t j = get_absolute_logical_x(), m = get_absolute_logical_y();
    for (uint32_t t = 0; t < NH; ++t) {
        cb_wait_front(cb_act, 1);
        noc_async_write(get_read_ptr(cb_act), s_act.get_noc_addr(m * NT + j * NH + t), tile);
        noc_async_writes_flushed();
        cb_pop_front(cb_act, 1);
    }
    noc_async_write_barrier();
}
