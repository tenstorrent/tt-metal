// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// main's matmul math init and call (llk_math_matmul.h, llk_math_matmul_api.h and the compute API's dynamic throttle at
// d6b3ab771ce), verbatim in their own namespace: a plain call that takes none of the matmul forms compiles to main's code
#include "llk_math_matmul_api.h"
#include "llk_math_matmul_main_lib.h"
#include "llk_math_matmul_phases.h"

namespace mm_main {

/*************************************************************************
 * LLK MATMUL
 *************************************************************************/

// Unified core, shared by the CB-id API below and the LLKOperand API (experimental/2_0/). Matmul math is
// FORMAT-FREE: it consumes only tile geometry (tile r/c dims + partial_face). The per-source prologue
// (resolving these from a CB id, or from an LLKMemDescriptor) lives in the callers. The matmul math EXECUTE
// (llk_math_matmul below) already takes no operand, so it is reused directly by the id-free API.
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init_impl(
    const std::uint32_t in0_tile_r_dim,
    const std::uint32_t in0_tile_c_dim,
    const std::uint32_t in1_tile_r_dim,
    const std::uint32_t in1_tile_c_dim,
    const bool partial_face,
    const std::uint32_t transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim) {
    _llk_math_matmul_init_<math_fidelity, THROTTLE_LEVEL>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    const std::uint32_t in0_id = get_operand_id(operandA);
    const std::uint32_t in1_id = get_operand_id(operandB);

    const std::uint32_t in0_tile_r_dim = get_operand_tile_r_dim(in0_id);
    const std::uint32_t in0_tile_c_dim = get_operand_tile_c_dim(in0_id);
    const std::uint32_t in1_tile_r_dim = get_operand_tile_r_dim(in1_id);
    const std::uint32_t in1_tile_c_dim = get_operand_tile_c_dim(in1_id);

    const bool partial_face = (in0_tile_r_dim < FACE_R_DIM);

    // In0/operandA -> srcB, In1/operandB -> srcA
    SAN_HOOK(init<OperationFpuMatmul>(
        StateVal<OperationFpuMatmul::MathFidelity>(to_underlying(math_fidelity)),
        StateVal<OperationFpuMatmul::ThrottleLevel>(THROTTLE_LEVEL),
        StateVal<OperationFpuMatmul::CtDim>(ct_dim),
        StateVal<OperationFpuMatmul::RtDim>(rt_dim),
        StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[in1_id]),
        StateDiscard<std::uint32_t>(transpose),
        StateDiscard<std::uint32_t>(in0_tile_r_dim),
        StateDiscard<std::uint32_t>(in0_tile_c_dim),
        StateDiscard<std::uint32_t>(in1_tile_r_dim),
        StateDiscard<std::uint32_t>(in1_tile_c_dim),
        StateDiscard<bool>(partial_face)));

    llk_math_matmul_init_impl<math_fidelity, THROTTLE_LEVEL>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0, std::uint32_t num_faces = 4 /*not used*/>
inline void llk_math_matmul(
    const std::uint32_t dst_index, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    static_assert(num_faces == 4, "num_faces other than 4 is not supported in llk_math_matmul");

    // DPRINT("llk_math_matmul: calculated dest tiles = {}, max dest tiles = {} (dst_index={}, ct_dim={},
    // rt_dim={})\n",
    //     ckernel::math::get_dest_max_matmul_tiles(dst_index, ct_dim, rt_dim),
    //     get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>(),
    //     dst_index,
    //     ct_dim,
    //     rt_dim);

    LLK_ASSERT(
        (ckernel::math::get_dest_max_matmul_tiles(dst_index, ct_dim, rt_dim) <
         get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "llk_math_matmul: computed matmul dest tile range exceeds available dest register "
        "capacity. Uncomment the DPRINT block above and enable DPRINT support to inspect "
        "the calculated and max dest tile values.");

    SAN_HOOK(execute<OperationFpuMatmul>(
        StateVal<OperationFpuMatmul::MathFidelity>(to_underlying(math_fidelity)),
        StateVal<OperationFpuMatmul::ThrottleLevel>(THROTTLE_LEVEL),
        StateVal<OperationFpuMatmul::CtDim>(ct_dim),
        StateVal<OperationFpuMatmul::RtDim>(rt_dim),
        StateDiscard<uint>(dst_index)));

    _llk_math_matmul_<math_fidelity, THROTTLE_LEVEL>(dst_index, ct_dim, rt_dim);
}

ALWI void matmul_block_math_dynamic_throttle(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t idst,
    const std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    // Dynamic throttling is only available on Blackhole architecture
    // Check firmware-controlled throttle enable flag (even = no throttle, odd = throttle)
    volatile std::uint32_t mm_throttle_en = *(throttle_ptr) % 2;
    if (mm_throttle_en) {
        if (throttled_mop_status != 1) {
            MATH((
                llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE_MAX>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 1;
        }
        MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE_MAX>(idst, ct_dim, rt_dim)));
    } else {
        if (throttled_mop_status != 0) {
            MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            if constexpr (matmul_kernel_phase_elidable() && MM_THROTTLE == 0) {
                MATH((matmul_cb_phases(to_underlying(MATH_FIDELITY), in0_cb_id, in1_cb_id, ct_dim, rt_dim)));
            }
            throttled_mop_status = 0;
        }
        MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
    }
#endif
}

}  // namespace mm_main
