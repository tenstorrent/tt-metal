// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute for QkvHeadsNormRope: the norm + RoPE stages of custom_ops/kernels/norm_rope_stages.hpp (x^2 on the FPU,
// FPU row reduce with the unit scaler, rsqrt on the SFPU, x * inv (SFPU column broadcast) * gamma (SFPU row
// broadcast) into the bf16 CB cb_n, then the FPU row-broadcast RoPE: out_c = n_c cos_c - n_{c+half} sin_c /
// out_c = n_c cos_c + n_{c-half} sin_c with the user's cos / sin row in row 0 of the tile), for ONE tensor per
// core: q or k (gamma + RoPE) or v (norm only), selected by the runtime arg; the output goes to cb_out (the
// writer ships it to the user's shard).
// compile-time args: [cb_x, cb_out, cb_g, cb_cos, cb_sin, cb_scaler, cb_sq, cb_inv, cols_t, half_t, eps_bits,
//                     inv_d_bits, cb_n]
// runtime args: [t0, t_count] (tensors 0 q, 1 k, 2 v of this core's work, in sequence)
#include <cstdint>
#include "../../kernels/norm_rope_stages.hpp"

constexpr uint32_t CB_X = get_compile_time_arg_val(0);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);
constexpr uint32_t CB_G = get_compile_time_arg_val(2);
constexpr uint32_t CB_COS = get_compile_time_arg_val(3);
constexpr uint32_t CB_SIN = get_compile_time_arg_val(4);
constexpr uint32_t CB_SCALER = get_compile_time_arg_val(5);
constexpr uint32_t CB_SQ = get_compile_time_arg_val(6);
constexpr uint32_t CB_INV = get_compile_time_arg_val(7);
constexpr uint32_t COLS_T = get_compile_time_arg_val(8);
constexpr uint32_t HALF_T = get_compile_time_arg_val(9);
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(10);
constexpr uint32_t INV_D_BITS = get_compile_time_arg_val(11);
constexpr uint32_t CB_N = get_compile_time_arg_val(12);

// the shared stages bound to this kernel's CBs and constants
constexpr auto row_rsqrt = norm_rope::row_rsqrt<CB_SQ, CB_INV, CB_SCALER, COLS_T, INV_D_BITS, EPS_BITS>;
constexpr auto normalize = norm_rope::normalize<CB_INV, COLS_T>;
constexpr auto rope = norm_rope::rope<CB_N, CB_COS, CB_SIN, COLS_T, HALF_T, /*ROW_BCAST=*/true>;

void kernel_main() {
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t t_count = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(CB_X, CB_SCALER, CB_OUT);
    CircularBuffer cx(CB_X), cs(CB_SCALER), cg(CB_G), ccos(CB_COS), csin(CB_SIN);
    for (uint32_t t = t0; t < t0 + t_count; ++t) {
        cx.wait_front(COLS_T);
        row_rsqrt(CB_X);
        if (t < 2) {
            normalize(CB_X, CB_G, true, CB_N);
            rope(CB_OUT);
            // the gamma and the cos / sin rows of this tensor are consumed (the reader refills them for the next)
            cg.pop_front(COLS_T);
            ccos.pop_front(COLS_T);
            csin.pop_front(COLS_T);
        } else {
            normalize(CB_X, CB_G, false, CB_OUT);
        }
        cx.pop_front(COLS_T);
    }
    cs.pop_front(1);
}
