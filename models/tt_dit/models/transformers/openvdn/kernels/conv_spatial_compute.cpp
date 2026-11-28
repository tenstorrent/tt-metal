// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Depthwise spatial conv (see conv_spatial_reader.cpp): per output tile, sum over the taps of (shifted bf16 input tile)
// x (the tap's per-channel fp32 weights), in fp32 on the SFPU. The bf16 input enters DST exactly; each product is one
// fp32 multiply, summed in tap order like the composite's multiply-add chain.
//
// Compile-time args: taps. Runtime args: number of units.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_w = 0;      // fp32, unpacked to DST
constexpr uint32_t cb_shift = 2;  // bf16
constexpr uint32_t cb_out = 16;   // fp32

FORCE_INLINE void copy_to(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t taps = get_compile_time_arg_val(0);
    const uint32_t count = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_shift, cb_out);
    pack_reconfig_data_format(cb_out);
    cb_wait_front(cb_w, taps);

    for (uint32_t u = 0; u < count; ++u) {
        tile_regs_acquire();
        for (uint32_t t = 0; t < taps; ++t) {
            cb_wait_front(cb_shift, 1);
            copy_to(cb_shift, 0, 1);
            copy_to(cb_w, t, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, t == 0 ? 0 : 1);
            if (t > 0) {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            cb_pop_front(cb_shift, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_w, taps);
}
