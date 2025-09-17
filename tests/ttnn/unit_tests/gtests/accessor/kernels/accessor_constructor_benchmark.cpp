// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/tensor_accessor.h"

#ifndef N_REPEAT
#define N_REPEAT 100
#endif

void kernel_main() {
    constexpr uint32_t base_idx_cta = 0;
    constexpr uint32_t base_idx_crta = 0;

    constexpr size_t max_tracy_zones = 125;
    constexpr size_t n_repeat = N_REPEAT;
    for (size_t i = 0; i < max_tracy_zones; ++i) {
        {
            DeviceZoneScopedN(ACCESSOR_CONFIG_NAME);
            for (size_t j = 0; j < n_repeat; ++j) {
                auto args = TensorAccessorArgs<base_idx_cta, base_idx_crta>();
                volatile auto tensor_accessor = TensorAccessor(args, 0, 1024);
                (void)tensor_accessor;
            }
        }
    }
}
