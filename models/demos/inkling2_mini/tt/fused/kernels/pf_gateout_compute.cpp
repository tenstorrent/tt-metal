// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill attention output gate, compute: out = bf16(attn * sigmoid(gate)), fp32 SFPU math (as the op chain:
// typecast both to fp32, sigmoid, multiply, typecast to bf16).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_a = 0, cb_g = 1, cb_out = 16;
    const uint32_t n_units = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_a, cb_out);
    for (uint32_t u = 0; u < n_units; ++u) {
        cb_wait_front(cb_a, 4);
        cb_wait_front(cb_g, 4);
        for (uint32_t c = 0; c < 4; ++c) {
            tile_regs_acquire();
            load(cb_a, c, 0);
            load(cb_g, c, 1);
            sigmoid_tile_init();
            sigmoid_tile(1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
            typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_a, 4);
        cb_pop_front(cb_g, 4);
    }
}
