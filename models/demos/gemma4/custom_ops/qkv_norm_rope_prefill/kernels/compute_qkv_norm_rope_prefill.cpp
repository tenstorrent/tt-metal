// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Prefill per-head RMSNorm (+ gamma on q / k) and q / k RoPE, one work unit = one [32, head_dim] tile row of one
// head (32 consecutive positions). Per unit the reader streams the row's column tiles into CB_X and, for q / k,
// the cos / sin tile rows of the same positions into CB_COS / CB_SIN; this kernel squares every column tile
// (FPU), reduces the row sums (FPU reduce with the unit scaler), takes rsqrt(sum / D + eps) on the SFPU and
// scales each column by it (SFPU column broadcast) and by gamma (row broadcast of the gamma tile's row 0), then
// rotates: out_a = n_a cos_a - n_b sin_b-pair, elementwise over the full tile (every row is its own position, so
// no row broadcast as in the decode op). v: norm only (no gamma, no rope). fp32 dest throughout, one bf16 pack.
// The stages are custom_ops/kernels/norm_rope_stages.hpp (shared with the decode forms).
#include <cstdint>
#include "../../kernels/norm_rope_stages.hpp"

constexpr uint32_t CB_X = get_compile_time_arg_val(0);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);
constexpr uint32_t CB_GQ = get_compile_time_arg_val(2);
constexpr uint32_t CB_GK = get_compile_time_arg_val(3);
constexpr uint32_t CB_COS = get_compile_time_arg_val(4);
constexpr uint32_t CB_SIN = get_compile_time_arg_val(5);
constexpr uint32_t CB_SCALER = get_compile_time_arg_val(6);
constexpr uint32_t CB_SQ = get_compile_time_arg_val(7);
constexpr uint32_t CB_INV = get_compile_time_arg_val(8);
constexpr uint32_t CB_N = get_compile_time_arg_val(9);
constexpr uint32_t COLS_T = get_compile_time_arg_val(10);
constexpr uint32_t HALF_T = get_compile_time_arg_val(11);
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(12);
constexpr uint32_t INV_D_BITS = get_compile_time_arg_val(13);

// unit kinds (runtime args): 0 = q (gamma q, rope), 1 = k (gamma k, rope), 2 = v (norm only)
constexpr uint32_t KIND_V = 2;

// the shared stages bound to this kernel's CBs and constants (the RoPE is elementwise: every row its own position)
constexpr auto row_rsqrt = norm_rope::row_rsqrt<CB_SQ, CB_INV, CB_SCALER, COLS_T, INV_D_BITS, EPS_BITS>;
constexpr auto normalize = norm_rope::normalize<CB_INV, COLS_T>;
constexpr auto rope = norm_rope::rope<CB_N, CB_COS, CB_SIN, COLS_T, HALF_T, /*ROW_BCAST=*/false>;

void kernel_main() {
    const uint32_t n_units = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_X, CB_SCALER, CB_OUT);
    CircularBuffer cx(CB_X), ccos(CB_COS), csin(CB_SIN);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t kind = get_arg_val<uint32_t>(1 + u);
        cx.wait_front(COLS_T);
        row_rsqrt(CB_X);
        if (kind == KIND_V) {
            normalize(CB_X, CB_GQ, false, CB_OUT);
        } else {
            normalize(CB_X, kind == 0 ? CB_GQ : CB_GK, true, CB_N);
            rope(CB_OUT);
            ccos.pop_front(COLS_T);  // the unit's cos / sin tile rows are consumed (the reader streams the next)
            csin.pop_front(COLS_T);
        }
        cx.pop_front(COLS_T);
    }
}
