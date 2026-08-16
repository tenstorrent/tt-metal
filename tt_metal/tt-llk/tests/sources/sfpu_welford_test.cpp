// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Silicon validation driver for the stateful Welford SFPU path.  It deliberately
// keeps the four code-generation variants in one source so their ELFs differ only
// in the recurrence body:
//   0 HANDWRITTEN_DIRECT, 1 HANDWRITTEN_REPLAY, 2 VFLOAT_DIRECT,
//   3 VFLOAT_RESCUE, 4 VFLOAT_MANUAL_EARLY_FOLD.

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params) {
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}
#endif

#ifdef LLK_TRISC_MATH
#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_welfords_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"

using namespace ckernel;

namespace {
constexpr std::array<std::uint32_t, 0> no_recip_lut{};

template <std::uint32_t I, std::uint32_t J>
sfpi_inline void handwritten_direct_block(std::uint32_t sample) {
    _welfords_load_block_<I, J>();
    _load_recip_of_idx_<0>(sample + 0, no_recip_lut);
    _compute_welfords_row_<ckernel::p_sfpu::LREG0>();
    _load_recip_of_idx_<0>(sample + 1, no_recip_lut);
    _compute_welfords_row_<ckernel::p_sfpu::LREG1>();
    _load_recip_of_idx_<0>(sample + 2, no_recip_lut);
    _compute_welfords_row_<ckernel::p_sfpu::LREG2>();
    _load_recip_of_idx_<0>(sample + 3, no_recip_lut);
    _compute_welfords_row_<ckernel::p_sfpu::LREG3>();
}

sfpi_inline void handwritten_direct() {
    handwritten_direct_block<0, 0>(0); handwritten_direct_block<0, 1>(4);
    handwritten_direct_block<0, 2>(8); handwritten_direct_block<0, 3>(12);
    handwritten_direct_block<1, 0>(16); handwritten_direct_block<1, 1>(20);
    handwritten_direct_block<1, 2>(24); handwritten_direct_block<1, 3>(28);
}

template <std::uint32_t N, sfpi::LRegs Input, bool EarlyFold>
sfpi_inline void vfloat_step() {
    sfpi::vFloat x    = sfpi::l_reg[Input];
    sfpi::vFloat mean = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vFloat m2   = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vFloat delta = x - mean;
    if constexpr (EarlyFold) {
        mean += delta * (1.0f / static_cast<float>(N));
        m2 += delta * (x - mean);
    } else {
        sfpi::vFloat next_mean = mean + delta * (1.0f / static_cast<float>(N));
        sfpi::vFloat next_m2 = m2 + delta * (x - next_mean);
        mean = next_mean;
        m2 = next_m2;
    }
    sfpi::l_reg[sfpi::LRegs::LReg4] = mean;
    sfpi::l_reg[sfpi::LRegs::LReg5] = m2;
}

template <std::uint32_t Base, std::uint32_t I, std::uint32_t J, bool EarlyFold>
sfpi_inline void vfloat_block() {
    _welfords_load_block_<I, J>();
    vfloat_step<Base + 1, sfpi::LRegs::LReg0, EarlyFold>();
    vfloat_step<Base + 2, sfpi::LRegs::LReg1, EarlyFold>();
    vfloat_step<Base + 3, sfpi::LRegs::LReg2, EarlyFold>();
    vfloat_step<Base + 4, sfpi::LRegs::LReg3, EarlyFold>();
}

template <bool EarlyFold>
sfpi_inline void vfloat_welford() {
    vfloat_block<0, 0, 0, EarlyFold>(); vfloat_block<4, 0, 1, EarlyFold>();
    vfloat_block<8, 0, 2, EarlyFold>(); vfloat_block<12, 0, 3, EarlyFold>();
    vfloat_block<16, 1, 0, EarlyFold>(); vfloat_block<20, 1, 1, EarlyFold>();
    vfloat_block<24, 1, 2, EarlyFold>(); vfloat_block<28, 1, 3, EarlyFold>();
}
}  // namespace

void run_kernel(RUNTIME_PARAMETERS params) {
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(0, formats.math, formats.math);

    _llk_math_welfords_sfpu_init_();
    ckernel::sfpu::_clear_previous_mean_and_m2_();
    if constexpr (WELFORD_IMPL == 0) {
        _llk_math_welfords_sfpu_params_(handwritten_direct, 0);
    } else if constexpr (WELFORD_IMPL == 1) {
        _llk_math_welfords_sfpu_params_(ckernel::sfpu::_calculate_welfords_tile_<0>, 0, 0, no_recip_lut);
    } else if constexpr (WELFORD_IMPL == 2) {
        _llk_math_welfords_sfpu_params_(vfloat_welford<false>, 0);
    } else if constexpr (WELFORD_IMPL == 3) {
        // Keep the state values live across an explicit local scope: this is the
        // pressure/rescue shape exercised by the solver-enabled compiler.
        { _llk_math_welfords_sfpu_params_(vfloat_welford<false>, 0); }
    } else {
        _llk_math_welfords_sfpu_params_(vfloat_welford<true>, 0);
    }
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_store_mean_m2_to_dst_, 1);
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif

#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params) {
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(1, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(2, L1_ADDRESS(params.buffer_Res[1]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif
