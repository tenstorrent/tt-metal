// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0

// Bank-local indexed expert matmul, compute: per slot, out[1 x NW tiles] = in0[slot or 0][1 x Kt] @ w[Kt x NW].
// Runtime arg 0: 1 = worker, 0 = idle core.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t NW = get_compile_time_arg_val(1);
    constexpr uint32_t SLOTS = get_compile_time_arg_val(2);
    constexpr uint32_t IN0_PER_SLOT = get_compile_time_arg_val(3);
    constexpr uint32_t cb_in0 = 0, cb_in1 = 1, cb_out = 16;
    if (get_arg_val<uint32_t>(0) == 0) {
        return;  // idle core of the multicast rectangle
    }

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_block_init(cb_in0, cb_in1, false, NW, 1, 1);
    cb_wait_front(cb_in0, IN0_PER_SLOT ? SLOTS * Kt : Kt);  // multicast once, all slots
    for (uint32_t s = 0; s < SLOTS; ++s) {
        const uint32_t a0 = IN0_PER_SLOT ? s * Kt : 0;
        cb_wait_front(cb_in1, Kt * NW);
        tile_regs_acquire();
        for (uint32_t k = 0; k < Kt; ++k) {
            matmul_block(cb_in0, cb_in1, a0 + k, k * NW, 0, false, NW, 1, 1);
        }
        tile_regs_commit();
        cb_pop_front(cb_in1, Kt * NW);
        cb_reserve_back(cb_out, NW);
        tile_regs_wait();
        for (uint32_t n = 0; n < NW; ++n) {
            pack_tile(n, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, NW);
    }
}
