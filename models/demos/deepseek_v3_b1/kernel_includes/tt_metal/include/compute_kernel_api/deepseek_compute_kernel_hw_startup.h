
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#endif

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

template <bool fp32_dest_acc_en = false>
ALWI void deepseek_compute_kernel_hw_startup2(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    UNPACK((llk_unpack_hw_configure<fp32_dest_acc_en>(icb1, icb0)));

    MATH((llk_math_pack_sync_init<fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<fp32_dest_acc_en>(icb0, icb1)));

    PACK((llk_pack_init<false, false, false>(ocb)));
    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(ocb)));
    PACK((llk_pack_dest_init<fp32_dest_acc_en, false>(ocb)));
}
