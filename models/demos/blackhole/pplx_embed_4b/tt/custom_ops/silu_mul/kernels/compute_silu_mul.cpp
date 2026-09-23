// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// out = silu(a) * b, CH tiles per DST session.
//   mode 0: copy a -> DST, silu_tile (SFPU, approximation per math_approx_mode), dst *= b (dest-reuse FPU mul)
//   mode 1: copy a -> DST i and DST CH+i, sigmoid<fast>(DST CH+i), DST i = DST i * DST CH+i (SFPU), dst *= b
//   mode 2: copy a -> DST i, b -> DST CH+i, clamped_silu_glu_tile(i, CH+i, i)  (clamps at 10)
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/clamped_silu_glu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_a = 0, cb_b = 1, cb_out = 16;
constexpr auto D2B = EltwiseBinaryReuseDestType::DEST_TO_SRCB;
}  // namespace

void kernel_main() {
    constexpr uint32_t CH = get_compile_time_arg_val(0);
    constexpr uint32_t mode = get_compile_time_arg_val(1);
    const uint32_t n_units = get_arg_val<uint32_t>(0);
    CircularBuffer ca(cb_a), cbb(cb_b), co(cb_out);
    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    for (uint32_t u = 0; u < n_units; ++u) {
        ca.wait_front(CH);
        cbb.wait_front(CH);
        co.reserve_back(CH);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_a);
        copy_tile_init(cb_a);
        for (uint32_t i = 0; i < CH; ++i) {
            copy_tile(cb_a, i, i);
        }
        if constexpr (mode == 0) {
            silu_tile_init();
            for (uint32_t i = 0; i < CH; ++i) {
                silu_tile(i);
            }
        } else if constexpr (mode == 1) {
            copy_dest_values_init();
            for (uint32_t i = 0; i < CH; ++i) {
                copy_dest_values(i, CH + i);
            }
            sigmoid_tile_init<true>();
            for (uint32_t i = 0; i < CH; ++i) {
                sigmoid_tile<VectorMode::RC, true>(CH + i);
            }
            mul_binary_tile_init();
            for (uint32_t i = 0; i < CH; ++i) {
                mul_binary_tile(i, CH + i, i);
            }
        } else {
            reconfig_data_format_srca(cb_b);
            copy_tile_init(cb_b);
            for (uint32_t i = 0; i < CH; ++i) {
                copy_tile(cb_b, i, CH + i);
            }
            clamped_silu_glu_tile_init();
            for (uint32_t i = 0; i < CH; ++i) {
                clamped_silu_glu_tile(i, CH + i, i);
            }
        }
        if constexpr (mode != 2) {
            reconfig_data_format_srca(cb_b);
            mul_reuse_dest_init<D2B>(cb_b);
            for (uint32_t i = 0; i < CH; ++i) {
                mul_reuse_dest_tiles<D2B>(cb_b, i, i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        for (uint32_t i = 0; i < CH; ++i) {
            pack_tile(i, cb_out, i);
        }
        tile_regs_release();
        co.push_back(CH);
        ca.pop_front(CH);
        cbb.pop_front(CH);
    }
}
