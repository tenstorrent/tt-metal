// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compile-time contract of the custom bcast-col shape guards; never executed on a device.
//
// Calls the real LLK entry points with a shape that is a compile-time constant, so the guard's
// compile-time arm (assert_tensor_shape_sub_bcast_col_custom_) decides whether this kernel builds: an
// unsupported shape must fail the build, a supported one must not.
//
// Compile-time args: 0 = site (0 unpack init, 1 SUB math execute, 2 MUL math execute - Blackhole only),
// 1 = face_r_dim, 2 = num_faces_r_dim, 3 = num_faces_c_dim.

#include <cstdint>
#include "tensor_shape.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#ifdef ARCH_BLACKHOLE
#include "api/compute/experimental/indexer_mul_custom.h"
#endif

void kernel_main() {
    constexpr std::uint32_t site = get_compile_time_arg_val(0);
    constexpr ckernel::TensorShape shape = ckernel::make_tensor_shape(
        get_compile_time_arg_val(1), ckernel::MAX_FACE_C_DIM, get_compile_time_arg_val(2), get_compile_time_arg_val(3));

#if defined(TRISC_UNPACK)
    if constexpr (site == 0) {
        _llk_unpack_AB_sub_bcast_col_init_custom_(shape);
    }
#endif

#if defined(TRISC_MATH)
    if constexpr (site == 1) {
        _llk_math_sub_bcast_cols_reuse_custom_(1 /*ct_dim*/, shape, 0 /*dst_index*/);
    }
#ifdef ARCH_BLACKHOLE
    if constexpr (site == 2) {
        _llk_math_bcast_cols_reuse_custom_<ckernel::EltwiseBinaryType::ELWMUL>(1 /*ct_dim*/, shape, 0 /*dst_index*/);
    }
#endif
#endif
}
