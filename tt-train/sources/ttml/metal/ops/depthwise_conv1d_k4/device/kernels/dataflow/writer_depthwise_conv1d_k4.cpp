// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_output = tt::CBIndex::c_4;

constexpr uint32_t block_ct = get_compile_time_arg_val(0);
constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
constexpr uint32_t Ct = get_compile_time_arg_val(2);

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto output_args = TensorAccessorArgs<3>();
    const auto output = TensorAccessor(output_args, output_addr);
    const uint32_t tile_bytes = get_tile_size(cb_output);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t mt = work / num_blocks;
        const uint32_t ct_start = (work % num_blocks) * block_ct;
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            cb_wait_front(cb_output, 1);
            noc_async_write_page(mt * Ct + ct_start + ct, output, get_read_ptr(cb_output));
            noc_async_write_barrier();
            cb_pop_front(cb_output, 1);
        }
    }
}
