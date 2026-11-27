// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// swiglu compute: per column tile, in fp32 DST on the SFPU: g = SiLU(min(gate, limit)), u = clamp(up, -limit, limit),
// m = g * u, packed to bf16 (the op chain: clamped_silu_glu = multiply with the gate's MINIMUM + SILU and the up's
// CLAMP input activations).
// Compile-time args (PH_CT0 + i): NT, LIMIT_BITS, NEG_LIMIT_BITS (fp32 bit patterns).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/swiglu/swiglu_common.hpp"

void kernel_main() {
    using namespace swiglu;
    constexpr uint32_t NT = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t LIMIT_BITS = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t NEG_LIMIT_BITS = get_compile_time_arg_val(PH_CT0 + 2);
    compute_kernel_hw_startup(CB_G, CB_U, CB_M);
    cb_wait_front(CB_G, NT);
    cb_wait_front(CB_U, NT);
    for (uint32_t j = 0; j < NT; ++j) {
        cb_reserve_back(CB_M, 1);
        tile_regs_acquire();
        copy_init(CB_G);
        copy_tile(CB_G, j, 0);
        copy_init(CB_U);
        copy_tile(CB_U, j, 1);
        unary_min_tile_init();
        unary_min_tile(0, LIMIT_BITS);
        silu_tile_init();
        silu_tile(0);
        clamp_tile_init();
        clamp_tile(1, NEG_LIMIT_BITS, LIMIT_BITS);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_M);
        tile_regs_release();
        cb_push_back(CB_M, 1);
    }
    cb_pop_front(CB_G, NT);
    cb_pop_front(CB_U, NT);
}
