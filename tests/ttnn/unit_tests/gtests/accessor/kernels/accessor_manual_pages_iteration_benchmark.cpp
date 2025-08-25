// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/tensor_accessor.h"

void kernel_main() {
    constexpr uint32_t base_idx_cta = 0;
    constexpr uint32_t base_idx_crta = 0;

    auto args = TensorAccessorArgs<base_idx_cta, base_idx_crta>();
    auto tensor_accessor = TensorAccessor(args, 0, 1024);

    auto tensor_volume = tensor_accessor.dspec().tensor_volume();

    constexpr size_t benchmark_iterations = 125;
    for (size_t iteration = 0; iteration < benchmark_iterations; ++iteration) {
        {
            DeviceZoneScopedN(ACCESSOR_CONFIG_NAME);
            // Manual iteration over all pages
            for (uint32_t page_id = 0; page_id < tensor_volume; ++page_id) {
                volatile auto _ = tensor_accessor.get_noc_addr(page_id);
            }
        }
    }
}
