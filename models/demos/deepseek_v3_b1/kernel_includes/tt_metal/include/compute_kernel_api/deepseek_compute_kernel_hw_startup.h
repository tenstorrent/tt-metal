// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#ifdef ARCH_BLACKHOLE
    MATH((llk_math_reconfig_remap(true)));
#endif
}
