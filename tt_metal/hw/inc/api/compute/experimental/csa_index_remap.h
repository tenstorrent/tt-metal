// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"

#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "sfpu/experimental/ckernel_sfpu_csa_index_remap.h"
#include "llk_math_eltwise_sfpu_op.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)
// Call before csa_index_remap, including when switching from another operation.
// Re-establish the common SFPU configuration, address modifiers and counters.
// The init is the shared SFPU init only, so ROW_OFFSET is irrelevant to it.
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void csa_index_remap_init() {
    MATH((SfpuUnaryFn<sfpu::_csa_index_remap_<32, 0>, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

template <std::uint32_t ROW_OFFSET, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void csa_index_remap(std::uint32_t idst) {
    // RC_custom advances over the 32 destination rows that make up one tile.
    MATH((SfpuUnaryFn<sfpu::_csa_index_remap_<32, ROW_OFFSET>, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC_custom)));
}
#endif

}  // namespace ckernel
