// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "../../../ttnn/fused/kernels/source_reciprocal.h"

using namespace ckernel;
#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void qwen38_reciprocal_fused_control() {
    for (int i = 0; i < ITERATIONS; ++i) {
        sfpi::vFloat input = sfpi::dst_reg[0];
        sfpi::vFloat val = sfpi::setexp(sfpi::setsgn(input, 1), 126);
        sfpi::vFloat inverse_ln2 = 1.442695f;
        sfpi::vFloat result = inverse_ln2 * (val * inverse_ln2 + 2.0f);
        for (int j = 0; j < 2; ++j) {
            result = result * (val * result + 2.0f);
        }
        sfpi::vInt exponent = sfpi::exexp(result) - sfpi::exexp(input) + 126;
        sfpi::vFloat out = sfpi::setexp(result, exponent);
        v_if(input < 0.0f) { out = -out; }
        v_endif;
        sfpi::dst_reg[0] = out;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
ALWI void reciprocal_fused_control(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        qwen38_reciprocal_fused_control,
        (APPROX, DST_ACCUM_MODE),
        idst,
        VectorMode::RC));
}

void kernel_main() {
    const uint32_t tiles = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 1);
    copy_init(0);
    qwen38_recip_tile_init();
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        CircularBuffer(0).wait_front(1);
        CircularBuffer(1).reserve_back(1);
        tile_regs_acquire();
        copy_tile(0, 0, 0);
        reciprocal_fused_control(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 1);
        tile_regs_release();
        CircularBuffer(0).pop_front(1);
        CircularBuffer(1).push_back(1);
    }
}
