// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/tensor_accessor.h"

void kernel_main() {
    constexpr uint32_t base_idx_cta = 0;
    constexpr uint32_t base_idx_crta = 0;

    auto args = TensorAccessorArgs<base_idx_cta, base_idx_crta>();
    auto sharded_accessor = TensorAccessor(args, 0, 1024);
    auto tensor_shape = sharded_accessor.dspec().tensor_shape();
    auto rank = sharded_accessor.dspec().rank();
    auto tensor_w = tensor_shape[rank - 1];
    auto tensor_h = tensor_shape[rank - 2];
    uint32_t page_coord[tensor_accessor::MAX_RANK] = {0};

    size_t loop_count = 125;
    for (size_t h = 0; h < tensor_h; ++h) {
        page_coord[rank - 2] = h;
        for (size_t w = 0; w < tensor_w; ++w) {
            page_coord[rank - 1] = w;
            {
                DeviceZoneScopedN(ACCESSOR_CONFIG_NAME);
                volatile auto _ = sharded_accessor.get_noc_addr(page_coord);
            }
            if (--loop_count == 0) {
                break;
            }
        }
    }
}
