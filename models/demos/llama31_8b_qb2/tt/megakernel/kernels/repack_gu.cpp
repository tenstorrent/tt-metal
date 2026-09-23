// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
constexpr auto source_args = TensorAccessorArgs<0>();
constexpr auto target_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
void kernel_main() {
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(1), 576);
    const auto target = TensorAccessor(target_args, get_arg_val<uint32_t>(2), 576);
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t k = 0; k < 128; k += 16) {
        // Source bank rows have28 tiles; copy their raw BFP4 headers and data.
        noc_async_read<16 * 28 * 576>(source.get_noc_addr(k * 224 + bank * 28), scratch, 16 * 28 * 576);
        noc_async_read_barrier();
        for (uint32_t row = 0; row < 16; ++row) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t tile = (half * 128 + k + row) * 112 + bank * 14;
                noc_async_write(scratch + (row * 28 + half * 14) * 576, target.get_noc_addr(tile), 14 * 576);
            }
        }
        noc_async_write_barrier();
    }
}
