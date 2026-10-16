// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// residual + bf16(block * coefficient) per tile, the GR write's two binary_ng programs in one 16-bit-dest kernel:
// ttnn.multiply (fast_and_approximate_mode False -> SFPU: copy lhs, broadcast the coefficient column, mul_binary_tile
// = fp32 product, software RNE, 0 * x = 0, pack bf16) then ttnn.add (fast_and_approximate_mode True, the binding's
// default -> FPU add_tiles(residual, update) from the two bf16 CBs, pack).  Operands sit in the chain's slots/order.
// Named compile-time args: cb_block, cb_res, cb_coef, cb_upd, cb_out.  Runtime args: 0 units.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "../../kernels/zones.h"

using namespace ckernel;

void kernel_main() {
    const uint32_t units = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_block = get_named_compile_time_arg_val("cb_block");
    constexpr uint32_t cb_res = get_named_compile_time_arg_val("cb_res");
    constexpr uint32_t cb_coef = get_named_compile_time_arg_val("cb_coef");
    constexpr uint32_t cb_upd = get_named_compile_time_arg_val("cb_upd");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t dst_upd = 0, dst_coef = 1;

    compute_kernel_hw_startup(cb_block, cb_coef, cb_out);
    mul_binary_tile_init();
    for (uint32_t u = 0; u < units; ++u) {
        FUSED_ZONE("fz_gw_c_unit");
        cb_wait_front(cb_block, 1);
        cb_wait_front(cb_coef, 1);
        cb_reserve_back(cb_upd, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_block);
        copy_init(cb_block);
        copy_tile(cb_block, 0, dst_upd);
        reconfig_data_format_srca(cb_coef);
        unary_bcast_init<BroadcastType::COL>(cb_coef);
        unary_bcast<BroadcastType::COL>(cb_coef, 0, dst_coef);
        mul_binary_tile(dst_upd, dst_coef, dst_upd);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_upd, cb_upd);
        tile_regs_release();
        cb_push_back(cb_upd, 1);
        cb_pop_front(cb_block, 1);
        cb_pop_front(cb_coef, 1);

        cb_wait_front(cb_res, 1);
        cb_wait_front(cb_upd, 1);
        cb_reserve_back(cb_out, 1);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_res, cb_upd);
        tile_regs_acquire();
        add_tiles(cb_res, cb_upd, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_res, 1);
        cb_pop_front(cb_upd, 1);
    }
}
