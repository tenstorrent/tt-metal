// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// glu_split compute (TRISC): h = silu(min(g, limit)) * clamp(u, -limit, limit) per tile pair, the same fused SFPU
// op the expert stream kernels apply in DST (api/compute/clamped_silu_glu.h, limit = 10 compile-time, GLM-5.3's
// swiglu_limit): gate -> DST 0, up -> DST 1, clamped_silu_glu -> DST 2, packed to the output CB.

#include <cstdint>

#include "api/compute/clamped_silu_glu.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t CB_G = get_compile_time_arg_val(0);
constexpr uint32_t CB_U = get_compile_time_arg_val(1);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(2);
}  // namespace

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_G, CB_OUT);
    CircularBuffer cbg(CB_G), cbu(CB_U), cbo(CB_OUT);
    for (uint32_t i = 0; i < count; ++i) {
        cbg.wait_front(1);
        cbu.wait_front(1);
        tile_regs_acquire();
        copy_init(CB_G);
        copy_tile(CB_G, 0, 0);
        copy_init(CB_U);
        copy_tile(CB_U, 0, 1);
        clamped_silu_glu_tile_init();
        clamped_silu_glu_tile(0, 1, 2);
        tile_regs_commit();
        cbo.reserve_back(1);
        tile_regs_wait();
        pack_tile(2, CB_OUT);
        tile_regs_release();
        cbo.push_back(1);
        cbg.pop_front(1);
        cbu.pop_front(1);
    }
}
