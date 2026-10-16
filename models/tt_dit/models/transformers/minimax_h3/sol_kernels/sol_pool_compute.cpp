// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn block-pooling pre-pass, compute. See sol_pool_reader.cpp for the unit decomposition.
// Per unit: O[n] = sum over the group's tile rows k of P[k] @ X[k, n] (DT output tiles accumulated in fp32 DST), with
// P[k] delivered by the reader per row (zero past the valid end); groups without tile rows output zero tiles.
//
// Compile-time args: 0 H, 1 RT, 2 DT, 3 NBPT, 4 R. Per-core runtime args: none (unit range from the core's logical
// coordinates).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_x = 0, cb_ps = 2, cb_zero = 3, cb_out = 16;
constexpr uint32_t BLOCK = 64;
inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }
}  // namespace

void kernel_main() {
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t DT = get_compile_time_arg_val(2);
    constexpr uint32_t NBPT = get_compile_time_arg_val(3);
    static_assert(DT <= 4, "DT output tiles must fit the half-sync fp32 DST (4 tiles)");

    constexpr uint32_t NX = get_compile_time_arg_val(5);
    constexpr uint32_t NY = get_compile_time_arg_val(6);
    constexpr uint32_t UNITS = get_compile_time_arg_val(7);
    // Unit range of this core from its logical coordinates (no per-core runtime args): units [ci*U/N, (ci+1)*U/N).
    constexpr uint32_t NCORES = NX * NY;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    CircularBuffer cbx(cb_x), cbps(cb_ps), cbz(cb_zero), cbo(cb_out);

    compute_kernel_hw_startup(cb_ps, cb_x, cb_out);
    cbz.wait_front(1);

    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t g = u % NBPT;
        const uint32_t k_begin = g * BLOCK;
        const uint32_t k_end = min_u(k_begin + BLOCK, RT);

        tile_regs_acquire();
        if (k_end <= k_begin) {
            copy_init(cb_zero);
            for (uint32_t n = 0; n < DT; ++n) {
                copy_tile(cb_zero, 0, n);
            }
        } else {
            matmul_init(cb_ps, cb_x);
            for (uint32_t k0 = k_begin; k0 < k_end; k0 += 4) {
                const uint32_t nk = min_u(4, k_end - k0);
                cbx.wait_front(4 * DT);  // full batches, see the reader
                for (uint32_t kk = 0; kk < nk; ++kk) {
                    cbps.wait_front(1);
                    for (uint32_t n = 0; n < DT; ++n) {
                        matmul_tiles(cb_ps, cb_x, 0, kk * DT + n, n);
                    }
                    cbps.pop_front(1);
                }
                cbx.pop_front(4 * DT);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        cbo.reserve_back(DT);
        for (uint32_t n = 0; n < DT; ++n) {
            pack_tile<true>(n, cb_out, n);
        }
        cbo.push_back(DT);
        tile_regs_release();
    }
}
