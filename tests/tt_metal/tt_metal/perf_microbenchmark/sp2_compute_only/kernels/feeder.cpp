// SPDX-License-Identifier: Apache-2.0
// SP2 compute-only feeder: push in0/in1 blocks WITHOUT reading DRAM (CB memory is garbage; matmul
// FPU throughput is data-independent). cb_reserve_back self-throttles to the compute consumer, so
// the feeder never bottlenecks a compute-bound kernel. Single output block => no in0 reuse.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in0_blk = get_compile_time_arg_val(0);
    constexpr uint32_t in1_blk = get_compile_time_arg_val(1);
    constexpr uint32_t k_num = get_compile_time_arg_val(2);
    constexpr uint32_t in0_cb = 0, in1_cb = 1;
    for (uint32_t k = 0; k < k_num; ++k) {
        cb_reserve_back(in0_cb, in0_blk);
        cb_push_back(in0_cb, in0_blk);
        cb_reserve_back(in1_cb, in1_blk);
        cb_push_back(in1_cb, in1_blk);
    }
}
