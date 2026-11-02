// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// skinny_matmul phase B: item (row r, n): out[r, n] = sum_s partial[r, s*Nt + n], packed to bf16.

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/skinny_matmul/device/kernels/skinny_matmul_cbs.hpp"

namespace cb = skinny_matmul_cb;

constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(5);

void kernel_main() {
    compute_kernel_hw_startup(cb::partials, cb::out);
    for (uint32_t item = 0; item < work_count; ++item) {
        cb_wait_front(cb::partials, S);
        tile_regs_acquire();
        reconfig_data_format_srca(cb::partials);
        copy_init(cb::partials);
        copy_tile(cb::partials, 0, 0);
        add_binary_tile_init();
        for (uint32_t s = 1; s < S; ++s) {
            copy_tile(cb::partials, s, 1);
            add_binary_tile(0, 1, 0);
        }
#if defined(TRANSPOSE_OUT) && !defined(X_TRANSPOSED)
        transpose_dest_init</*is_32bit*/ true>(cb::partials);  // out tile (n, r) = (sum)^T, exact in fp32 DEST
        transpose_dest</*is_32bit*/ true>(0);  // (a transposed x's partials are the transposed tiles already)
#endif
        tile_regs_commit();
        pack_and_push(0, cb::out);
        cb_pop_front(cb::partials, S);
    }
}
