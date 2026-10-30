// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// vp_cross_entropy_bw compute: per logits tile DEST = exp(x - log_normalizer[row]) * scale * g (fp32), packed
// bf16; g is a full tile of the scalar (elementwise multiply); the one-hot term is applied by the writer.

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
#include "tt-train/sources/ttml/metal/common/sdpa_compute_utils_common.hpp"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_logz = 1;
constexpr uint32_t cb_grad = 2;
constexpr uint32_t cb_out = 3;

constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t slices = get_compile_time_arg_val(2);
constexpr uint32_t St = get_compile_time_arg_val(3);
constexpr uint32_t block = get_compile_time_arg_val(4);
constexpr uint32_t scale_bits = get_compile_time_arg_val(5);
constexpr uint32_t one_bits = 0x3F800000U;
constexpr uint32_t grad_reg = block;  // DEST slot of the grad tile (block data tiles before it)

void kernel_main() {
    const uint32_t work_start = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_logz, cb_out);
    cb_wait_front(cb_grad, 1);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t s = work - (work / slices) * slices;
        const uint32_t k0 = s * St;
        const uint32_t nk = (k0 + St <= Wt) ? St : (Wt - k0);
        cb_wait_front(cb_logz, 1);
        for (uint32_t k = 0; k < nk; k += block) {
            const uint32_t n = (k + block <= nk) ? block : (nk - k);
            cb_wait_front(cb_in, block);
            tile_regs_acquire();
            reconfig_data_format(cb_in, cb_logz);
            sub_bcast_cols_init(cb_in, cb_logz);
            for (uint32_t j = 0; j < n; ++j) {
                sub_tiles_bcast_cols(cb_in, cb_logz, j, 0, j);
            }
            sdpa_exp_tile_init();
            for (uint32_t j = 0; j < n; ++j) {
                sdpa_exp_tile(j);
            }
            if constexpr (scale_bits != one_bits) {
                binop_with_scalar_tile_init();
                for (uint32_t j = 0; j < n; ++j) {
                    mul_unary_tile(j, scale_bits);
                }
            }
            reconfig_data_format_srca(cb_grad);
            copy_init(cb_grad);
            copy_tile(cb_grad, 0, grad_reg);
            mul_binary_tile_init();
            for (uint32_t j = 0; j < n; ++j) {
                mul_binary_tile(j, grad_reg, j);
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, block);
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, block);
            cb_pop_front(cb_in, block);
        }
        cb_pop_front(cb_logz, 1);
    }
    cb_pop_front(cb_grad, 1);
}
