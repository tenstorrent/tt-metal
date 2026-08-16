// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// BH Welford ABI probe.  Each invocation performs the first three rows with
// the canonical raw sequence, then observes one prefix of row four.  The nine
// result tiles are LREG0..LREG7 and LREG11, respectively.  This deliberately
// avoids stores between the observed instructions, so a snapshot cannot hide
// a dependency hazard in the prefix being measured.

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

template <std::uint32_t Stage>
sfpi_inline void raw_prefix() {
    // Canonical Welford row-four sequence, stopped after Stage (1..6).
    TTI_SFPMAD(p_sfpu::LREG11, p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG6, 0);
    if constexpr (Stage >= 2) TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG7, p_sfpu::LREG4, p_sfpu::LREG6, 0);
    if constexpr (Stage >= 3) TTI_SFPMAD(p_sfpu::LREG11, p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG4, 0);
    if constexpr (Stage >= 4) TTI_SFPMAD(p_sfpu::LREG11, p_sfpu::LREG6, p_sfpu::LREG3, p_sfpu::LREG3, 0);
    if constexpr (Stage >= 5) TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG5, 0);
    if constexpr (Stage >= 6) TTI_SFPMOV(0, p_sfpu::LREG6, p_sfpu::LREG4, 0);
}

template <std::uint32_t Stage>
sfpi_inline void vfloat_prefix() {
    // Same ABI and recurrence, but through vFloat lowering.  LREG7 is loaded
    // with raw IEEE FP32 1/4 below rather than using a compile-time scalar.
    sfpi::vFloat x     = sfpi::l_reg[sfpi::LRegs::LReg3];
    sfpi::vFloat mean  = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vFloat m2    = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vFloat recip = sfpi::l_reg[sfpi::LRegs::LReg7];
    sfpi::vFloat delta = x - mean;
    if constexpr (Stage >= 2) {
        sfpi::vFloat next_mean = mean + delta * recip;
        if constexpr (Stage >= 3) {
            sfpi::vFloat alpha = x - mean;
            if constexpr (Stage >= 4) {
                sfpi::vFloat beta = x - next_mean;
                if constexpr (Stage >= 5) {
                    sfpi::vFloat next_m2 = m2 + alpha * beta;
                    if constexpr (Stage >= 6) mean = next_mean;
                    if constexpr (Stage >= 6) m2 = next_m2;
                }
            }
        }
    }
    if constexpr (Stage >= 6) {
        sfpi::l_reg[sfpi::LRegs::LReg4] = mean;
        sfpi::l_reg[sfpi::LRegs::LReg5] = m2;
    }
}

sfpi_inline void dump_abi() {
    constexpr auto mode = sfpi::SFPSTORE_MOD0_FMT_SRCB;
    TTI_SFPSTORE(p_sfpu::LREG0, mode, ADDR_MOD_7,   0);
    TTI_SFPSTORE(p_sfpu::LREG1, mode, ADDR_MOD_7,  64);
    TTI_SFPSTORE(p_sfpu::LREG2, mode, ADDR_MOD_7, 128);
    TTI_SFPSTORE(p_sfpu::LREG3, mode, ADDR_MOD_7, 192);
    TTI_SFPSTORE(p_sfpu::LREG4, mode, ADDR_MOD_7, 256);
    TTI_SFPSTORE(p_sfpu::LREG5, mode, ADDR_MOD_7, 320);
    TTI_SFPSTORE(p_sfpu::LREG6, mode, ADDR_MOD_7, 384);
    TTI_SFPSTORE(p_sfpu::LREG7, mode, ADDR_MOD_7, 448);
    TTI_SFPSTORE(p_sfpu::LREG11, mode, ADDR_MOD_7, 512);
}

template <std::uint32_t Impl, std::uint32_t Stage>
sfpi_inline void probe() {
    _welfords_load_block_<0, 0>();
    sfpu::_clear_previous_mean_and_m2_();
    _load_recip_of_idx_<0>(0, no_recip_lut); _compute_welfords_row_<p_sfpu::LREG0>();
    _load_recip_of_idx_<0>(1, no_recip_lut); _compute_welfords_row_<p_sfpu::LREG1>();
    _load_recip_of_idx_<0>(2, no_recip_lut); _compute_welfords_row_<p_sfpu::LREG2>();
    // This is exact raw FP32 1/(3 + 1), placed in the required ABI LREG7.
    _load_recip_of_idx_<0>(3, no_recip_lut);
    if constexpr (Impl == 0) raw_prefix<Stage>(); else vfloat_prefix<Stage>();
    dump_abi();
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
    _llk_math_welfords_sfpu_params_(probe<PROBE_IMPL, PROBE_STAGE>, 0);
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
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(9, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif
