// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "data_format_derive.h"
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

/*************************************************************************
 * LLK hw_configure -- LLKOperand (id-free, compile-time NTTP) overloads
 *
 * Same function names as the CB-id hw_configure APIs, distinguished by taking LLKMemDescriptor NTTPs instead
 * of CB ids. src/dst formats, face geometry and per-tile size all come from the operand descriptors -- no CB
 * arrays, no runtime L1-format inference. The two source-register formats are reconciled to a common
 * exponent-width family via infer_unpack_dst_format_2op: matching formats are unchanged, a Float32
 * operand rebiases to the other's width, and any other cross-family pairing is a hard compile error.
 *
 * This header is deliberately NOT included by the broadly-included *_common_api.h files: it pulls the
 * ckernel::experimental namespace in, which collides with metal's top-level ::experimental (kernel_args.h)
 * under `using namespace ckernel`. It is included ONLY by compute_kernel_hw_startup.h, whose id-free overload
 * is the sole consumer -- and which is never pulled into the legacy kernel_args-based kernels.
 *************************************************************************/

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"

template <
    bool is_fp32_dest_acc_en,
    ckernel::experimental::LLKMemDescriptor DESC_A,
    ckernel::experimental::LLKMemDescriptor DESC_B>
inline void llk_unpack_hw_configure() {
    constexpr DataFormat A = DESC_A.format;
    constexpr DataFormat B = DESC_B.format;
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        static_cast<std::uint32_t>(DESC_A.format),
        static_cast<std::uint32_t>(DESC_B.format),
        static_cast<std::uint32_t>(ckernel::infer_unpack_dst_format_2op<A, B>(is_fp32_dest_acc_en)),
        static_cast<std::uint32_t>(ckernel::infer_unpack_dst_format_2op<B, A>(is_fp32_dest_acc_en)),
        DESC_A.shape.face_r_dim,
        DESC_B.shape.face_r_dim,
        DESC_A.shape.total_num_faces(),
        DESC_B.shape.total_num_faces(),
        ckernel::experimental::tile_stride_words(DESC_A.format, DESC_A.shape),
        ckernel::experimental::tile_stride_words(DESC_B.format, DESC_B.shape));
}
#endif

#ifdef TRISC_MATH
#include "llk_math_common_api.h"

template <
    bool is_fp32_dest_acc_en,
    ckernel::experimental::LLKMemDescriptor DESC_A,
    ckernel::experimental::LLKMemDescriptor DESC_B>
inline void llk_math_hw_configure() {
    constexpr DataFormat A = DESC_A.format;
    constexpr DataFormat B = DESC_B.format;
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(
        static_cast<std::uint32_t>(ckernel::infer_unpack_dst_format_2op<A, B>(is_fp32_dest_acc_en)),
        static_cast<std::uint32_t>(ckernel::infer_unpack_dst_format_2op<B, A>(is_fp32_dest_acc_en)));
}
#endif

#ifdef TRISC_PACK
#include "llk_pack_common_api.h"

template <bool is_fp32_dest_acc_en, ckernel::experimental::LLKMemDescriptor OUT_DESC>
inline void llk_pack_hw_configure() {
    constexpr std::uint8_t pack_src = ckernel::infer_pack_reg_fmt(OUT_DESC.format, is_fp32_dest_acc_en);
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        pack_src,
        static_cast<std::uint32_t>(OUT_DESC.format),
        ckernel::experimental::tile_stride_words(OUT_DESC.format, OUT_DESC.shape),
        OUT_DESC.shape.face_r_dim,
        OUT_DESC.shape.total_col_dim(),
        OUT_DESC.shape.total_num_faces(),
        false /*partial_face*/,
        0 /*relu_config*/);
}
#endif
