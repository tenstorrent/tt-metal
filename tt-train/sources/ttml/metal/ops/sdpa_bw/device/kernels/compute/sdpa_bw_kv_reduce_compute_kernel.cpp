// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KV-backward head reduce, compute: out = sum of the G fp32 partial tiles of the block (fp32 DEST), packed bf16.

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t cb_partials = tt::CBIndex::c_0;
constexpr uint32_t cb_out = tt::CBIndex::c_1;
constexpr uint32_t G = get_compile_time_arg_val(0);

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_partials, cb_out);
    for (uint32_t item = 0; item < count; ++item) {
        cb_wait_front(cb_partials, G);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_partials);
        copy_init(cb_partials);
        copy_tile(cb_partials, 0, 0);
        add_binary_tile_init();
        for (uint32_t h = 1; h < G; ++h) {
            copy_tile(cb_partials, h, 1);
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        pack_and_push(0, cb_out);
        cb_pop_front(cb_partials, G);
    }
}
