// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_gdn_inv_rms_diag_entry.h"
#endif

// The inverse-RMS diagonal of a Gram tile, D = eye * ([scale *] rsqrt(G * eye + eps)), two ways:
//
//   MODE 0: llk_math_gdn_inv_rms_diag, one SFPU pass over the two diagonal faces (the chunk_gdn prep's gather);
//   MODE 1: the five-call chain it replaces, mul_binary(eye), add_unary(eps), rsqrt, [mul_unary(scale)],
//           mul_binary(eye), as full-tile passes.
//
//   c_0  : G tiles, Float32 on the unpack-to-dest path (bit-exact into DST)
//   c_1  : the identity tile, Float32 on the unpack-to-dest path, one per G tile
//   c_16 : D tiles, Float32
//
// Compile-time args: 0 NUM_TILES, 1 MODE, 2 DO_SCALE. Runtime args: 0 eps_bits, 1 scale_bits (fp32 bit patterns).
// Per tile: copy G to DST 0 and the identity to DST 1, run the chain in place on DST 0, pack DST 0. Requires
// fp32_dest_acc_en.
namespace {
constexpr uint32_t NUM_TILES = get_compile_time_arg_val(0);
constexpr uint32_t MODE = get_compile_time_arg_val(1);
constexpr bool DO_SCALE = get_compile_time_arg_val(2) != 0;
constexpr uint32_t DST_G = 0;
constexpr uint32_t DST_EYE = 1;
constexpr uint32_t ONE_TILE = 1;
}  // namespace

void kernel_main() {
    const uint32_t eps_bits = get_arg_val<uint32_t>(0);
    const uint32_t scale_bits = get_arg_val<uint32_t>(1);
    constexpr auto cb_g = tt::CBIndex::c_0;
    constexpr auto cb_eye = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    CircularBuffer g(cb_g);
    CircularBuffer eye(cb_eye);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup(cb_g, cb_out);
    copy_init(cb_g);
    for (uint32_t t = 0; t < NUM_TILES; ++t) {
        g.wait_front(ONE_TILE);
        eye.wait_front(ONE_TILE);
        out.reserve_back(ONE_TILE);
        tile_regs_acquire();
        copy_tile(cb_g, 0 /*in_tile_index*/, DST_G);
        copy_tile(cb_eye, 0 /*in_tile_index*/, DST_EYE);
        if constexpr (MODE == 0) {
            MATH((ckernel::llk_math_gdn_inv_rms_diag_init<APPROX>()));
            MATH((ckernel::llk_math_gdn_inv_rms_diag<APPROX, DO_SCALE>(DST_G, DST_EYE, eps_bits, scale_bits)));
        } else {
            mul_binary_tile_init();
            mul_binary_tile(DST_G, DST_EYE, DST_G);  // diag(G); off-diagonal 0
            binop_with_scalar_tile_init();
            add_unary_tile(DST_G, eps_bits);  // + eps (off-diagonal: eps)
            rsqrt_tile_init();
            rsqrt_tile(DST_G);  // off-diagonal: rsqrt(eps), finite
            if constexpr (DO_SCALE) {
                binop_with_scalar_tile_init();
                mul_unary_tile(DST_G, scale_bits);
            }
            mul_binary_tile_init();
            mul_binary_tile(DST_G, DST_EYE, DST_G);  // off-diagonal -> 0
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(DST_G, cb_out);
        tile_regs_release();
        g.pop_front(ONE_TILE);
        eye.pop_front(ONE_TILE);
        out.push_back(ONE_TILE);
    }
}
