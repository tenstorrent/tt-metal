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
    /* Benchmark get_noc_addr for both accessors
     * - get_noc_addr is a good proxy for page lookup logic
     * - Use volatile to prevent compiler from optimizing away the calls
     * - Some notes on tracy profiler:
     *   - Max number of markers is 125 so we use a hard-coded loop count to make sure tests are consistent for
     * larger shapes
     *   - Cycles are fairly consistent across runs without much variation
     *   - There is a small overhead for tracy (~18 cycles?) for each marker
     *     * You can verify by inserting a dummy marker or removing the volatile since compiler will optimize out the
     * calls
     */
    constexpr size_t loop_count = 125;
    for (size_t i = 0; i < loop_count; ++i) {
        auto page_id = i % tensor_accessor.dspec().tensor_volume();
        {
            DeviceZoneScopedN(ACCESSOR_CONFIG_NAME);
            volatile auto _ = tensor_accessor.get_noc_addr(i);
        }
    }
}
