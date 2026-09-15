// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#include "experiments/sdpa-l2/single-core-resident-v1/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "../exp_grid7.hpp"

void kernel_main() {
    constexpr uint32_t scale = get_compile_time_arg_val(0);
    constexpr uint32_t batch = get_compile_time_arg_val(1);
    static_assert(!DST_ACCUM_MODE, "Grid7 probe requires BF16 DST");
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 16);
    copy_init(0);
    MATH((ckernel::math::_configure_src_zero_flag_(false)));
    lofi_grid7_exp_packthread_tile_init<true, scale, InputClamping::None, false>();
    PACK((llk_pack_relu_config(ReluConfig::zero())));
    for (uint32_t i = 0; i < count; i += batch) {
        cb_wait_front(0, batch);
        cb_reserve_back(16, batch);
        tile_regs_acquire();
        for (uint32_t j = 0; j < batch; ++j) {
            copy_tile(0, j, j);
        }
        tile_regs_commit();
        cb_pop_front(0, batch);
        tile_regs_wait();
        for (uint32_t j = 0; j < batch; ++j) {
            exp_packthread_tile<true, false, InputClamping::None, 32, false>(j, VectorMode::None);
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_tile(j, 16);
        }
        tile_regs_release();
        cb_push_back(16, batch);
    }
    PACK((llk_pack_relu_config(ReluConfig::none())));
}
