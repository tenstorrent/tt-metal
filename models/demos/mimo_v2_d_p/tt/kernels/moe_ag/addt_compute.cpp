// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Row add with a tiled output, compute: per block, the 32 row segments of a + b (element-wise, layout agnostic) into
// c_24 (32 rows x 1024 row major), then tilized into c_16 (32 tiles).
// RT: 0 n (blocks on this core)
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0);
    constexpr uint32_t DST = 4;
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_24);
    for (uint32_t blk = 0; blk < n; ++blk) {
        add_tiles_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
        pack_reconfig_data_format(tt::CBIndex::c_24);
        cb_wait_front(tt::CBIndex::c_0, 32);
        cb_wait_front(tt::CBIndex::c_1, 32);
        cb_reserve_back(tt::CBIndex::c_24, 32);
        for (uint32_t j0 = 0; j0 < 32; j0 += DST) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < DST; ++j) {
                add_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, j0 + j, j0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < DST; ++j) {
                pack_tile<true>(j, tt::CBIndex::c_24, j0 + j);
            }
            tile_regs_release();
        }
        cb_push_back(tt::CBIndex::c_24, 32);
        cb_pop_front(tt::CBIndex::c_0, 32);
        cb_pop_front(tt::CBIndex::c_1, 32);
        compute_kernel_lib::tilize<32, tt::CBIndex::c_24, tt::CBIndex::c_16>(1);
    }
}
