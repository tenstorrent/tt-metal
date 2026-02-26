// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#endif

namespace ckernel {
constexpr uint32_t SFPU_FPU = semaphore::UNPACK_MATH_DONE;
}

ALWI void deepseek_compute_kernel_init() {
    MATH(ckernel::t6_semaphore_init(ckernel::semaphore::FPU_SFPU, 0, 1));
    PACK(ckernel::t6_semaphore_init(ckernel::SFPU_FPU, 0, 1));
    compute_kernel_hw_startup(0, 0, 0);
}

/**
 * Hardware startup for DeepSeek compute kernel.
 * Call once at kernel start. Same as compute_kernel_hw_startup() but with configurable fp32_dest_acc_en.
 */
template <bool fp32_dest_acc_en = false>
ALWI void deepseek_compute_kernel_hw_startup(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    UNPACK((llk_unpack_hw_configure<fp32_dest_acc_en>(icb0, icb1)));

    MATH((llk_math_pack_sync_init<fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<fp32_dest_acc_en>(icb0, icb1)));

    PACK((llk_pack_init<false, false, false>(ocb)));
    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(ocb)));
    PACK((llk_pack_dest_init<fp32_dest_acc_en, false>(ocb)));
}
