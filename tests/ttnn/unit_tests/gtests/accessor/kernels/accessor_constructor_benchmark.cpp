// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/tensor_accessor.h"

void kernel_main() {
    constexpr uint32_t base_idx_cta = 0;
    constexpr uint32_t base_idx_crta = 0;

    constexpr size_t loop_count = 125;
    for (size_t i = 0; i < loop_count; ++i) {
        {
            DeviceZoneScopedN(ACCESSOR_CONFIG_NAME);
            auto args = make_tensor_accessor_args<base_idx_cta, base_idx_crta>();
            volatile auto tensor_accessor = make_tensor_accessor_from_args(args, 0, 1024);
            (void)tensor_accessor;
        }
    }
}
