// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// lowrank_update compute: per item (image, n, r): DEST0 = a[r] @ w[n] (one K tile, HiFi4, fp32), * scale,
// + acc tile (copied to DEST1 at fp32), packed to bf16. The a column stays in cb::a for the image, the w tile in
// cb::w for the column; the item order (column-major) is the reader's.

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/lowrank_update/device/kernels/lowrank_update_cbs.hpp"

namespace cb = lowrank_update_cb;

constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t Mt = get_compile_time_arg_val(1);
constexpr uint32_t Nt = get_compile_time_arg_val(2);
constexpr uint32_t scale_bits = get_compile_time_arg_val(3);
constexpr uint32_t one_bits = 0x3F800000U;
#ifdef TRANSPOSE_B
constexpr uint32_t transpose_w = 1;
#else
constexpr uint32_t transpose_w = 0;
#endif

void kernel_main() {
    const uint32_t work_start = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb::a, cb::w, cb::out);

    constexpr uint32_t per_image = Mt * Nt;
    uint32_t cur_image = 0xFFFFFFFFU;
    uint32_t cur_col = 0xFFFFFFFFU;
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t image = work / per_image;
        const uint32_t rem = work - image * per_image;
        const uint32_t n = rem / Mt;
        const uint32_t r = rem - n * Mt;
        if (image != cur_image) {
            if (cur_image != 0xFFFFFFFFU) {
                cb_pop_front(cb::a, Mt);
            }
            if (cur_col != 0xFFFFFFFFU) {
                cb_pop_front(cb::w, 1);  // the reader re-pushes the column's w tile for the new image
            }
            cb_wait_front(cb::a, Mt);
            cur_image = image;
            cur_col = 0xFFFFFFFFU;
        }
        if (n != cur_col) {
            if (cur_col != 0xFFFFFFFFU) {
                cb_pop_front(cb::w, 1);
            }
            cb_wait_front(cb::w, 1);
            cur_col = n;
        }
        cb_wait_front(cb::acc, 1);

        tile_regs_acquire();
        reconfig_data_format(cb::a, cb::w);
        matmul_init(cb::a, cb::w, transpose_w);
        matmul_tiles(cb::a, cb::w, r, 0, 0);
        if constexpr (scale_bits != one_bits) {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, scale_bits);
        }
        reconfig_data_format_srca(cb::acc);
        copy_init(cb::acc);
        copy_tile(cb::acc, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        pack_and_push(0, cb::out);
        cb_pop_front(cb::acc, 1);
    }
    if (cur_col != 0xFFFFFFFFU) {
        cb_pop_front(cb::w, 1);
    }
    if (cur_image != 0xFFFFFFFFU) {
        cb_pop_front(cb::a, Mt);
    }
}
