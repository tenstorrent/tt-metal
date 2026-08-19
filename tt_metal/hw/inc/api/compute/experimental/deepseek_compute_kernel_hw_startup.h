// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <cstdint>

#include "api/compute/common.h"

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#endif

namespace ckernel {
constexpr std::uint32_t SFPU_FPU = semaphore::UNPACK_MATH_DONE;
}

/**
 * @brief Initialize the DeepSeek compute kernel: set up the FPU<->SFPU handshake semaphores, then run the
 *        default hardware startup. Call once at kernel start, before the compute loop.
 * @note MATH: init the FPU_SFPU semaphore. PACK: init the SFPU_FPU (= UNPACK_MATH_DONE) semaphore. The two
 *       gate the cross-thread FPU<->SFPU handshake the DeepSeek kernels rely on; the call then runs
 *       compute_kernel_hw_startup(0, 0, 0). Use @ref deepseek_compute_kernel_hw_startup instead when the
 *       kernel needs configurable fp32 dest accumulation or non-default CB indices.
 * @tparam enable_math_reconfig_remap On Blackhole, enable math srcA/srcB register remap on reconfig after
 *         startup (no-op on other archs). The deepseek_v3_b1 / tt-blaze kernel family runs with it enabled
 *         (the default); the ttnn moe-gate lineage predates it and opts out to preserve its behavior.
 */
template <bool enable_math_reconfig_remap = true>
ALWI void deepseek_compute_kernel_init() {
    MATH(ckernel::t6_semaphore_init(ckernel::semaphore::FPU_SFPU, 0, 1));
    PACK(ckernel::t6_semaphore_init(ckernel::SFPU_FPU, 0, 1));
    compute_kernel_hw_startup(0, 0, 0);
    if constexpr (enable_math_reconfig_remap) {
        MATH((llk_math_reconfig_remap(true)));
    }
}

// Self-guarding + self-seeding compute HW init. chlkc unpack_src_format is a per-core
// constexpr table (255 = CB absent). Seed HW startup from the first present CB — an
// identity conversion the LLK allowlist always accepts — or skip if no CB is configured.
#if defined(COMPILE_FOR_TRISC)
constexpr std::uint32_t DEEPSEEK_NO_PRESENT_CB = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t deepseek_first_present_cb() {
    constexpr std::uint32_t n = (std::uint32_t)(sizeof(unpack_src_format) / sizeof(unpack_src_format[0]));
    for (std::uint32_t i = 0; i < n; ++i) {
        if (unpack_src_format[i] != 255) {
            return i;
        }
    }
    return DEEPSEEK_NO_PRESENT_CB;
}

ALWI void deepseek_compute_kernel_init_present() {
    constexpr std::uint32_t seed = deepseek_first_present_cb();
    if constexpr (seed != DEEPSEEK_NO_PRESENT_CB) {
        MATH(ckernel::t6_semaphore_init(ckernel::semaphore::FPU_SFPU, 0, 1));
        PACK(ckernel::t6_semaphore_init(ckernel::SFPU_FPU, 0, 1));
        compute_kernel_hw_startup(seed, seed, seed);
        MATH((llk_math_reconfig_remap(true)));
    }
}
#endif

/**
 * Hardware startup for DeepSeek compute kernel.
 * Call once at kernel start. Same as compute_kernel_hw_startup() but with configurable fp32_dest_acc_en.
 */
template <bool fp32_dest_acc_en = false>
ALWI void deepseek_compute_kernel_hw_startup(std::uint32_t icb0, std::uint32_t icb1, std::uint32_t ocb) {
    UNPACK((llk_unpack_hw_configure<fp32_dest_acc_en>(icb0, icb1)));

    MATH((llk_math_pack_sync_init<fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<fp32_dest_acc_en>(icb0, icb1)));

    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(ocb)));
    PACK((llk_pack_init<PackMode::Default>(ocb)));
    PACK((llk_pack_dest_init<fp32_dest_acc_en, PackMode::Default>(ocb)));
}
