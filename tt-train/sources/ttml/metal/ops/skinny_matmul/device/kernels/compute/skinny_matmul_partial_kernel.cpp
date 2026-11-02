// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// skinny_matmul phase A: item (row group, K slice); per row r of the group
//   partial[r, n] = sum_{k in slice} x[r, k] @ W[k, n]      (fp32 DEST accumulation over the slice, HiFi4)
// W's slice sits in cb::w for the whole item (index k*Nt + n); the x tiles stream `block` at a time.

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reg_api.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/skinny_matmul/device/kernels/skinny_matmul_cbs.hpp"

namespace cb = skinny_matmul_cb;

constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t rows = get_compile_time_arg_val(1);
constexpr uint32_t Kt = get_compile_time_arg_val(2);
constexpr uint32_t Nt = get_compile_time_arg_val(3);
constexpr uint32_t R = get_compile_time_arg_val(4);
constexpr uint32_t S = get_compile_time_arg_val(5);
constexpr uint32_t St = get_compile_time_arg_val(6);
constexpr uint32_t block = get_compile_time_arg_val(7);
#ifdef TRANSPOSE_B
constexpr uint32_t transpose_w = 1;
#else
constexpr uint32_t transpose_w = 0;
#endif

void kernel_main() {
#ifdef X_TRANSPOSED
    // x^T's tiles are the transposes of x's: they can only be the matmul's in1, so the roles swap (iteration 40):
    // DEST[col] += w(col, k) @ x^T(k, r) with w's raw [N, K] tiles (transpose_b) as in0 gives the partial of the
    // TRANSPOSED output tile (col, r) directly; no unpacker transpose anywhere.
    compute_kernel_hw_startup(cb::w, cb::x, cb::partial_out);
    matmul_init(cb::w, cb::x, 0);
#else
    compute_kernel_hw_startup(cb::x, cb::w, cb::partial_out);
    matmul_init(cb::x, cb::w, transpose_w);  // the transpose flag is how ttnn's matmul reads a transposed B
#endif

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work_start = get_arg_val<uint32_t>(0);
        const uint32_t work = work_start + item;
        const uint32_t g = work / S;
        const uint32_t s = work - g * S;
        const uint32_t k0 = s * St;
        const uint32_t nk = (k0 + St <= Kt) ? St : (Kt - k0);
        const uint32_t r0 = g * R;
        const uint32_t nr = (r0 + R <= rows) ? R : (rows - r0);

        // The reader pushes the W slice as St*Nt tiles and x as full blocks (the shorter tail of a slice too), so the
        // CB pointers never straddle the buffer end; only the first nk*Nt / n tiles of each are read here.
        cb_wait_front(cb::w, St * Nt);
        for (uint32_t r = 0; r < nr; ++r) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < nk; k += block) {
                const uint32_t n = (k + block <= nk) ? block : (nk - k);
                cb_wait_front(cb::x, block);
                for (uint32_t j = 0; j < n; ++j) {
                    for (uint32_t col = 0; col < Nt; ++col) {
#ifdef X_TRANSPOSED
                        matmul_tiles(cb::w, cb::x, (k + j) * Nt + col, j, col);
#else
                        matmul_tiles(cb::x, cb::w, j, (k + j) * Nt + col, col);
#endif
                    }
                }
                cb_pop_front(cb::x, block);
            }
            tile_regs_commit();
            cb_reserve_back(cb::partial_out, Nt);
            tile_regs_wait();
            pack_reconfig_data_format(cb::partial_out);
            for (uint32_t col = 0; col < Nt; ++col) {
                pack_tile(col, cb::partial_out);
            }
            tile_regs_release();
            cb_push_back(cb::partial_out, Nt);
        }
        cb_pop_front(cb::w, St * Nt);
    }
}
