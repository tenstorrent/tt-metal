// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/sharded_accessor.h"

void kernel_main() {
    constexpr uint32_t base_idx_cta = 0;
    constexpr uint32_t base_idx_crta = 0;

    constexpr size_t loop_count = 125;
    for (size_t i = 0; i < loop_count; ++i) {
        {
            DeviceZoneScopedN(ACCESSOR_CONFIG_NAME);
            auto args = nd_sharding::make_args<base_idx_cta, base_idx_crta>();
            volatile auto sharded_accessor = nd_sharding::make_sharded_accessor_from_args(args, 0, 1024);
            (void)sharded_accessor;
        }
    }
}
