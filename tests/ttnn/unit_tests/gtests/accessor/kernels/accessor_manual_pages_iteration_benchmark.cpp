// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    constexpr uint32_t base_idx_cta = 0;
    constexpr uint32_t base_idx_crta = 0;

    auto args = TensorAccessorArgs<base_idx_cta, base_idx_crta>();
    auto tensor_accessor = TensorAccessor(args, 0);

#if INTERLEAVED_LAYOUT
    // For interleaved layout, get page count from compile-time args (passed from host)
    uint32_t num_pages = get_compile_time_arg_val(args.next_compile_time_args_offset());
#else
    // For sharded layout, get page count from dspec
    auto num_pages = tensor_accessor.dspec().tensor_volume();
#endif

    constexpr size_t benchmark_iterations = 125;
    for (size_t iteration = 0; iteration < benchmark_iterations; ++iteration) {
        {
            DeviceZoneScopedN(ACCESSOR_CONFIG_NAME);
            // Manual iteration over all pages
            for (uint32_t page_id = 0; page_id < num_pages; ++page_id) {
                volatile auto _ = tensor_accessor.get_noc_addr(page_id);
            }
        }
    }
}
