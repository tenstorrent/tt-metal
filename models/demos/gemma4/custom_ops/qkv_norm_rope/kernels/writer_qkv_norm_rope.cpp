// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer for QkvNormRope: the outputs are tensor-backed CBs written by the packer; wait for all three so the
// program ends only once the shards are complete.
// compile-time args: [cb_qo, cb_ko, cb_vo, cols_t]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_qo = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ko = get_compile_time_arg_val(1);
    constexpr uint32_t cb_vo = get_compile_time_arg_val(2);
    constexpr uint32_t cols_t = get_compile_time_arg_val(3);
    for (uint32_t cb : {cb_qo, cb_ko, cb_vo}) {
        CircularBuffer c(cb);
        c.wait_front(cols_t);
        c.pop_front(cols_t);
    }
}
