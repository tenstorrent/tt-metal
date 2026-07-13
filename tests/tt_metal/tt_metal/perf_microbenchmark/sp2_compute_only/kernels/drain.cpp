// SPDX-License-Identifier: Apache-2.0
// SP2 compute-only drain: pop the single output block the compute kernel produces (once).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t out_blk = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = 2;
    cb_wait_front(out_cb, out_blk);
    cb_pop_front(out_cb, out_blk);
}
