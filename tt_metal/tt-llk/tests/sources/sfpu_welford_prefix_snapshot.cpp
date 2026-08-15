// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// One-snapshot full-body Welford diagnostic.  TRACE_N selects exactly one
// post-row state.  No SFPSTORE occurs before that row, so the observed prefix
// is not perturbed by tracing.  The remaining rows intentionally still exist
// in the linked program to keep code-generation context full-tile.

#include <array>
#include <cstdint>
#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context = 0;
std::uint32_t pack_sync_tile_dst_ptr = 0;
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

template <std::uint32_t N>
sfpi_inline void maybe_capture() {
    if constexpr (N == TRACE_N) {
        // Dedicated Dst locations, all distinct from input tile 0.  This is
        // emitted only after the selected state has been produced.
        TTI_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7,  64);
        TTI_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 128);
        TTI_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 192);
        TTI_SFPSTORE(p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 256);
        TTI_SFPSTORE(p_sfpu::LREG4, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 320);
        TTI_SFPSTORE(p_sfpu::LREG5, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 384);
        TTI_SFPSTORE(p_sfpu::LREG6, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 448);
        TTI_SFPSTORE(p_sfpu::LREG7, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 512);
        TTI_SFPSTORE(p_sfpu::LREG11, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 576);
    }
}

template <std::uint32_t N, std::uint32_t Input>
sfpi_inline void raw_step() {
    _load_recip_of_idx_<0>(N - 1, no_recip_lut);
    _compute_welfords_row_<Input>();
    maybe_capture<N>();
}

template <std::uint32_t N, sfpi::LRegs Input, std::uint32_t Impl>
sfpi_inline void vfloat_step() {
    sfpi::vFloat x = sfpi::l_reg[Input];
    sfpi::vFloat mean = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vFloat m2 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vFloat delta = x - mean;
    constexpr float recip = 1.0f / static_cast<float>(N);
    if constexpr (Impl == 2) { // VFLOAT_DIRECT: normal source shape/literals.
        sfpi::vFloat next_mean = mean + delta * recip;
        sfpi::vFloat next_m2 = m2 + delta * (x - next_mean);
        mean = next_mean;
        m2 = next_m2;
    } else if constexpr (Impl == 3) { // VFLOAT_RESCUE.
        mean += delta * recip;
        sfpi::vFloat delta2 = x - mean;
        m2 += delta * delta2;
    } else { // VFLOAT_MANUAL_EARLY_FOLD.
        sfpi::vFloat delta2 = x - (mean + delta * recip);
        mean += delta * recip;
        m2 += delta * delta2;
    }
    sfpi::l_reg[sfpi::LRegs::LReg4] = mean;
    sfpi::l_reg[sfpi::LRegs::LReg5] = m2;
    maybe_capture<N>();
}

template <std::uint32_t Impl, std::uint32_t Base, std::uint32_t I, std::uint32_t J>
sfpi_inline void block() {
    _welfords_load_block_<I, J>();
    if constexpr (Impl == 0) { // HANDWRITTEN_DIRECT.
        raw_step<Base+1, p_sfpu::LREG0>(); raw_step<Base+2, p_sfpu::LREG1>();
        raw_step<Base+3, p_sfpu::LREG2>(); raw_step<Base+4, p_sfpu::LREG3>();
    } else if constexpr (Impl == 1) { // HANDWRITTEN_REPLAY.
        _load_recip_of_idx_<0>(Base, no_recip_lut); _execute_welfords_row_replay_buffer_<p_sfpu::LREG0>(); maybe_capture<Base+1>();
        _load_recip_of_idx_<0>(Base+1, no_recip_lut); _execute_welfords_row_replay_buffer_<p_sfpu::LREG1>(); maybe_capture<Base+2>();
        _load_recip_of_idx_<0>(Base+2, no_recip_lut); _execute_welfords_row_replay_buffer_<p_sfpu::LREG2>(); maybe_capture<Base+3>();
        _load_recip_of_idx_<0>(Base+3, no_recip_lut); _execute_welfords_row_replay_buffer_<p_sfpu::LREG3>(); maybe_capture<Base+4>();
    } else {
        vfloat_step<Base+1, sfpi::LRegs::LReg0, Impl>(); vfloat_step<Base+2, sfpi::LRegs::LReg1, Impl>();
        vfloat_step<Base+3, sfpi::LRegs::LReg2, Impl>(); vfloat_step<Base+4, sfpi::LRegs::LReg3, Impl>();
    }
}

template <std::uint32_t Impl>
sfpi_inline void full_body() {
    block<Impl,0,0,0>(); block<Impl,4,0,1>(); block<Impl,8,0,2>(); block<Impl,12,0,3>();
    block<Impl,16,1,0>(); block<Impl,20,1,1>(); block<Impl,24,1,2>(); block<Impl,28,1,3>();
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
    _llk_math_welfords_sfpu_init_(); ckernel::sfpu::_clear_previous_mean_and_m2_();
    {
        START_PERF_MEASURE("WELFORD_BODY")
        _llk_math_welfords_sfpu_params_(full_body<TRACE_IMPL>, 0);
    }
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
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>(); _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(1, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(2, L1_ADDRESS(params.buffer_Res[1]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(3, L1_ADDRESS(params.buffer_Res[2]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(4, L1_ADDRESS(params.buffer_Res[3]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(5, L1_ADDRESS(params.buffer_Res[4]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(6, L1_ADDRESS(params.buffer_Res[5]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(7, L1_ADDRESS(params.buffer_Res[6]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(8, L1_ADDRESS(params.buffer_Res[7]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(9, L1_ADDRESS(params.buffer_Res[8]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif
