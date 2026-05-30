// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_rmsnorm_bcast_scalar_dest_reuse_api.h"
#endif
#ifdef TRISC_UNPACK
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_unpack_A_rmsnorm_api.h"
#endif

// `MATH_FIDELITY` is emitted by jit_build/genfiles into chlkc_descriptors.h
// gated on `#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK)` (see
// `emit_compute_descriptors` in `tt_metal/jit_build/genfiles.cpp`), so it is
// visible to the MATH and PACK TRISC builds but NOT to UNPACK. The helpers
// below take a `MathFidelity math_fidelity = MATH_FIDELITY` template default;
// that default is parsed by all three TRISC compilations (UNPACK/MATH/PACK),
// even though the value is only ever read inside `MATH(...)` macros (which
// expand to nothing in UNPACK/PACK). Provide a stub for the UNPACK build so
// the parse succeeds; the value is unused. The stub MUST be excluded for
// both MATH and PACK to avoid redefining the chlkc_descriptors.h symbol.
#if !defined(UCK_CHLKC_MATH) && !defined(UCK_CHLKC_PACK)
[[maybe_unused]] static constexpr ckernel::MathFidelity MATH_FIDELITY = ckernel::MathFidelity::HiFi4;
#endif

namespace ckernel {

// "rmsnorm-style" eltwise binary with SCALAR broadcast on SrcB and
// DEST_TO_SRCB destination reuse. Originally written for RMSNorm's
// `x * (1/RMS)` (see deepseek_b1_ops::RMSNorm in `rmsnorm.hpp`), this is the
// hang-safe shape for "binary op with the result of a prior SFPU op", because
// it keeps the unpacker/math DVALID handshake aligned across SFPU drains:
//
//   * UNPACK side (`llk_unpack_A_rmsnorm_init`):
//       SrcB DVALID is asserted exactly ONCE per MOP run via the MOP's
//       start_op (`UNPACR_NOP(SrcB SET_DVALID)`); the body streams only SrcA
//       UNPACRs, regardless of num_tiles/num_faces or the optional 32x32
//       transpose. This is the property that survives an SFPU op such as
//       `recip_tile` -- a per-face SrcB DVALID cadence (the shape of stock
//       `llk_unpack_A_init<NONE, true, DEST_TO_SRCB>`) races with the
//       FPU-drain stall and deadlocks TRISC0 inside
//       `_llk_unpack_A_init_::mop_sync`.
//
//   * MATH side (`_llk_math_rmsnorm_bcast_scalar_dest_reuse_`):
//       Init asserts `CLR_DVALID_SrcA_Disable_ADDR32`, so SrcA DVALID is NOT
//       auto-cleared per face -- matching one-shot SrcB DVALID. Each call
//       does a single `STALLWAIT(WAIT_SFPU | SRCB_VLD)` then `MOVD2B(DST ->
//       SrcB)`, runs the elwise MOP, and finally issues
//       `SETRWC(CLR_B, …, SET_D)` to drop SrcB. So MATH consumes exactly one
//       SrcB DVALID per call -- matching the one the unpacker sets.
//
// Use this helper (or the `rmsnorm_mul_*` aliases below) for any eltwise op
// of the form "DST[d] = DST[s] op (in_cb tile)" where the DST value to reuse
// was produced by a reduce-then-SFPU sequence. Setting
// `unpack_full_transpose=true` additionally folds a 32x32 tile transpose
// into the SrcA stream (within-face 16x16 transpose via Haloize_mode +
// face-level (0,2,1,3) face-read order via the unpack MOP). Currently the
// transpose path supports the (num_tiles==1, num_faces==4) shape only; see
// `_llk_unpack_A_rmsnorm_mop_config_` in
// `tt_llk_blackhole/llk_lib/llk_unpack_A_rmsnorm.h`.
template <
    EltwiseBinaryType eltwise_binary_type = ELWADD,
    uint32_t num_tiles,
    MathFidelity math_fidelity = MATH_FIDELITY,
    bool unpack_full_transpose = false>
ALWI void rmsnorm_bcast_scalar_reuse_tiles_init(uint32_t icb0) {
    UNPACK((llk_unpack_A_rmsnorm_init<num_tiles, BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
        unpack_full_transpose /*transpose_of_faces*/, unpack_full_transpose /*within_face_16x16_transpose*/, icb0)));
    MATH((llk_math_rmsnorm_bcast_scalar_dest_reuse_init_with_operands<eltwise_binary_type, num_tiles, math_fidelity>(
        icb0, icb0, false /*acc_to_dest*/)));
}

template <
    EltwiseBinaryType eltwise_binary_type = ELWADD,
    uint32_t num_tiles,
    MathFidelity math_fidelity = MATH_FIDELITY,
    bool clear_dest = false>
ALWI void rmsnorm_bcast_scalar_reuse_tiles(
    uint32_t in_cb_id, uint32_t in_tile_index, uint32_t src_tile_index, uint32_t dst_tile_index) {
    // Transpose configuration (Haloize_mode + transposed face-read MOP) is
    // established at init; the runtime side just runs the programmed MOP.
    UNPACK(
        (llk_unpack_A<BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(in_cb_id, in_tile_index)));
    MATH((llk_math_rmsnorm_bcast_scalar_dest_reuse<
          eltwise_binary_type,
          num_tiles,
          DST_ACCUM_MODE,
          math_fidelity,
          clear_dest>(src_tile_index, dst_tile_index)));
}

template <uint32_t num_tiles>
ALWI void rmsnorm_mul_bcast_scalar_reuse_tiles_init(uint32_t icb0) {
    rmsnorm_bcast_scalar_reuse_tiles_init<EltwiseBinaryType::ELWMUL, num_tiles>(icb0);
}

template <uint32_t num_tiles, bool clear_dest = false>
ALWI void rmsnorm_mul_bcast_scalar_reuse_tiles(
    uint32_t in_cb_id, uint32_t in_tile_index, uint32_t src_tile_index, uint32_t dst_tile_index) {
    rmsnorm_bcast_scalar_reuse_tiles<ELWMUL, num_tiles, MATH_FIDELITY, clear_dest>(
        in_cb_id, in_tile_index, src_tile_index, dst_tile_index);
}
}  // namespace ckernel
