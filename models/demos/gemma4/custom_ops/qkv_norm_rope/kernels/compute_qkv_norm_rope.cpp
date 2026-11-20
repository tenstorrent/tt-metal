// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute for QkvNormRope (decode): the norm stage of custom_ops/kernels/norm_rope_stages.hpp (x^2 on the FPU, FPU
// row reduce with the unit scaler, rsqrt on the SFPU, x * inv (SFPU column broadcast) * gamma (SFPU row
// broadcast), one pack per column tile in page order) lands the normalized q / k columns in the bf16 CB cb_n; the
// RoPE stage is its FPU row-broadcast form, one output column per window in page order:
//   c < half:  out_c = n_c * cos_c - n_{c+half} * sin_c      c >= half:  out_c = n_c * cos_c + n_{c-half} * sin_c
// with cos / sin row-broadcast by the FPU (mul_tiles_bcast_rows: the user's row sits in row 0 of the tile) and the
// two products combined on fp32 dest by the binary SFPU. v (no gamma, no RoPE) packs straight to its output.
// compile-time args: [cb_q, cb_k, cb_v, cb_qo, cb_ko, cb_vo, cb_gq, cb_gk, cb_cos, cb_sin, cb_scaler, cb_sq,
//                     cb_inv, cols_t, half_t, eps_bits, inv_d_bits, cb_n]
#include <cstdint>
#include "../../kernels/norm_rope_stages.hpp"

constexpr uint32_t CB_Q = get_compile_time_arg_val(0);
constexpr uint32_t CB_K = get_compile_time_arg_val(1);
constexpr uint32_t CB_V = get_compile_time_arg_val(2);
constexpr uint32_t CB_QO = get_compile_time_arg_val(3);
constexpr uint32_t CB_KO = get_compile_time_arg_val(4);
constexpr uint32_t CB_VO = get_compile_time_arg_val(5);
constexpr uint32_t CB_GQ = get_compile_time_arg_val(6);
constexpr uint32_t CB_GK = get_compile_time_arg_val(7);
constexpr uint32_t CB_COS = get_compile_time_arg_val(8);
constexpr uint32_t CB_SIN = get_compile_time_arg_val(9);
constexpr uint32_t CB_SCALER = get_compile_time_arg_val(10);
constexpr uint32_t CB_SQ = get_compile_time_arg_val(11);
constexpr uint32_t CB_INV = get_compile_time_arg_val(12);
constexpr uint32_t COLS_T = get_compile_time_arg_val(13);
constexpr uint32_t HALF_T = get_compile_time_arg_val(14);
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(15);
constexpr uint32_t INV_D_BITS = get_compile_time_arg_val(16);
constexpr uint32_t CB_N = get_compile_time_arg_val(17);

// the shared stages bound to this kernel's CBs and constants (the decode RoPE: one user's cos / sin row, broadcast)
constexpr auto row_rsqrt = norm_rope::row_rsqrt<CB_SQ, CB_INV, CB_SCALER, COLS_T, INV_D_BITS, EPS_BITS>;
constexpr auto normalize = norm_rope::normalize<CB_INV, COLS_T>;
constexpr auto rope = norm_rope::rope<CB_N, CB_COS, CB_SIN, COLS_T, HALF_T, /*ROW_BCAST=*/true>;

// In-place safe: the output CB of a tensor may be bound to its input shard. q / k are read only by row_rsqrt and
// normalize (which packs to CB_N) before rope packs their output columns; v's normalize reads column c into dest
// before it packs column c and never reads an earlier column again.
void kernel_main() {
    compute_kernel_hw_startup(CB_Q, CB_SCALER, CB_QO);
    CircularBuffer cq(CB_Q), ck(CB_K), cv(CB_V);
    cq.wait_front(COLS_T);
    ck.wait_front(COLS_T);
    cv.wait_front(COLS_T);
    row_rsqrt(CB_Q);
    normalize(CB_Q, CB_GQ, true, CB_N);
    rope(CB_QO);
    row_rsqrt(CB_K);
    normalize(CB_K, CB_GK, true, CB_N);
    rope(CB_KO);
    row_rsqrt(CB_V);
    normalize(CB_V, CB_GQ, false, CB_VO);
    cq.pop_front(COLS_T);
    ck.pop_front(COLS_T);
    cv.pop_front(COLS_T);
}
