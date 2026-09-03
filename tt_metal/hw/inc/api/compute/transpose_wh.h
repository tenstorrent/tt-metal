// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// DEPRECATED HEADER.
//
// The transpose compute API has been renamed to drop the `_wh` suffix and to follow the
// compute_kernel_hw_startup() + <op>_init() programming model. The canonical API now lives in
// "api/compute/transpose.h" (transpose_init / transpose_tile).
//
// Everything below is a thin [[deprecated]] compatibility shim that forwards to the new API. This
// whole file is scheduled for removal (see .github/deprecations.json); migrate to transpose.h:
//   transpose_wh_init(icb, ocb)  ->  compute_kernel_hw_startup(icb, ocb); transpose_init(icb);
//   transpose_wh_init_short(icb) ->  transpose_init(icb);
//   transpose_wh_tile(...)       ->  transpose_tile(...);
//

#include "api/compute/transpose.h"
#include "llk_assert.h"

namespace ckernel {

// clang-format off
/**
 * @deprecated Use compute_kernel_hw_startup(icb, ocb) once at the top of the kernel, then
 * transpose_init(icb) before transpose_tile(). See "api/compute/transpose.h".
 *
 * Performs the full (long) init for transpose_wh: one-time hardware configuration of the
 * unpacker/math/packer plus the transpose-specific reconfiguration. Body kept verbatim for
 * backwards compatibility.
 */
// clang-format on
[[deprecated(
    "Use compute_kernel_hw_startup(icb, ocb) once at the top of the kernel, then transpose_init(icb). See "
    "api/compute/transpose.h.")]] ALWI void
transpose_wh_init(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);

#if defined(TRISC_MATH) || defined(TRISC_UNPACK)
    const std::uint32_t src_format = get_operand_src_format(icb);
    const std::uint32_t dst_format = get_operand_dst_format(icb);

#ifndef ARCH_QUASAR
    const bool is_8bit_int = (src_format & 0xf) == (std::uint32_t)DataFormat::Int8;
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb)));

    if (enable_unpack_to_dest) {
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            true, false, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
        MATH((llk_math_transpose_dest_init<false, true>()));
    } else if (is_8bit_int) {
        // 8-bit integer (Int8/UInt8) transpose needs the int-FPU (ELWADD) A2D reconstruct path,
        // selected here via is_int_fpu_en. Ideally the LLK layer would infer this path from the
        // data format instead of selecting it here in the Compute API layer.
        // TODO: #46832.
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE>(true, true, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<
              DataCopyType::A2D,
              DST_ACCUM_MODE,
              BroadcastType::NONE,
              true /*is_int_fpu_en*/>(icb)));
    } else {
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE>(true, true, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
    }
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb, icb)));
#else
    // Quasar has no unpack-to-dest transpose path (TODO: tt-llk#1559) and no int-FPU 8-bit integer
    // reconstruct path; reject formats that would otherwise silently take the wrong path. UInt32 is
    // treated as unpack-to-dest on WH/BH and Int8/UInt8 (low nibble 0xE) need the int-FPU path.
    const bool is_8bit_int = (src_format & 0xf) == (std::uint32_t)DataFormat::Int8;
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);
    LLK_ASSERT(
        !enable_unpack_to_dest, "32-bit (unpack-to-dest) transpose not supported on Quasar");  // TODO: tt-llk#1559
    LLK_ASSERT(!is_8bit_int, "8-bit integer transpose not supported on Quasar");
    UNPACK((llk_unpack_hw_configure(icb)));
    UNPACK((llk_unpack_A_init<
            BroadcastType::NONE,
            false /*acc_to_dest*/,
            EltwiseBinaryReuseDestType::NONE,
            false /*unpack_to_dest*/>(true /*transpose_of_faces*/, true /*within_face_16x16_transpose*/, icb)));

    MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::A2D, DST_ACCUM_MODE>(icb)));
    MATH((llk_math_pack_sync_init()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb, icb)));
#endif
#endif

#ifndef ARCH_QUASAR
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>()));
#else
    PACK((llk_pack_hw_configure(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init()));
#endif
}

/**
 * @deprecated Use transpose_init(icb). See "api/compute/transpose.h".
 */
[[deprecated("Use transpose_init(icb). See api/compute/transpose.h.")]] ALWI void transpose_wh_init_short(
    uint32_t icb, uint32_t call_line = __builtin_LINE()) {
    transpose_init(icb, call_line);
}

/**
 * @deprecated Use transpose_tile(icb, itile, idst). See "api/compute/transpose.h".
 */
[[deprecated("Use transpose_tile(icb, itile, idst). See api/compute/transpose.h.")]] ALWI void transpose_wh_tile(
    uint32_t icb, uint32_t itile, uint32_t idst) {
    transpose_tile(icb, itile, idst);
}

}  // namespace ckernel
