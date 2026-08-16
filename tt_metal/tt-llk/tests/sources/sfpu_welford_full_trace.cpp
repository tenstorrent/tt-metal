// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Full-tile BH Welford differential trace.  Tile 0 remains the input until
// the computation is complete.  Per-row mean/M2 snapshots use only tiles 1-4
// at non-overlapping four-row destinations; final normal output then uses
// tile 0.  Thus trace storage cannot overwrite an input or a prior snapshot.

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

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
sfpi_inline void trace_state() {
    static_assert(N >= 1 && N <= 32);
    constexpr std::uint32_t slot = ((N - 1) & 15) << 2;
    constexpr std::uint32_t bank = ((N - 1) >> 4) << 6;
    constexpr auto mode = sfpi::SFPSTORE_MOD0_FMT_SRCB;
    // mean occupies tiles 1/2; M2 occupies tiles 3/4.
    TTI_SFPSTORE(p_sfpu::LREG4, mode, ADDR_MOD_7,  64 + bank + slot);
    TTI_SFPSTORE(p_sfpu::LREG5, mode, ADDR_MOD_7, 192 + bank + slot);
}

template <std::uint32_t N, std::uint32_t Input>
sfpi_inline void raw_step_trace() {
    _load_recip_of_idx_<0>(N - 1, no_recip_lut);
    _compute_welfords_row_<Input>();
    trace_state<N>();
}

template <std::uint32_t N, sfpi::LRegs Input>
sfpi_inline void vfloat_step_trace() {
    // This is intentionally the normal VFLOAT_DIRECT source shape: literals
    // are compiler-generated, not normalized through LREG7.
    sfpi::vFloat x = sfpi::l_reg[Input];
    sfpi::vFloat mean = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vFloat m2 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vFloat delta = x - mean;
    sfpi::vFloat next_mean = mean + delta * (1.0f / static_cast<float>(N));
    sfpi::vFloat next_m2 = m2 + delta * (x - next_mean);
    mean = next_mean;
    m2 = next_m2;
    sfpi::l_reg[sfpi::LRegs::LReg4] = mean;
    sfpi::l_reg[sfpi::LRegs::LReg5] = m2;
    trace_state<N>();
}

template <std::uint32_t Base, std::uint32_t I, std::uint32_t J>
sfpi_inline void raw_block() {
    _welfords_load_block_<I, J>();
    raw_step_trace<Base + 1, p_sfpu::LREG0>(); raw_step_trace<Base + 2, p_sfpu::LREG1>();
    raw_step_trace<Base + 3, p_sfpu::LREG2>(); raw_step_trace<Base + 4, p_sfpu::LREG3>();
}

template <std::uint32_t Base, std::uint32_t I, std::uint32_t J>
sfpi_inline void vfloat_block() {
    _welfords_load_block_<I, J>();
    vfloat_step_trace<Base + 1, sfpi::LRegs::LReg0>(); vfloat_step_trace<Base + 2, sfpi::LRegs::LReg1>();
    vfloat_step_trace<Base + 3, sfpi::LRegs::LReg2>(); vfloat_step_trace<Base + 4, sfpi::LRegs::LReg3>();
}

template <bool VFloat>
sfpi_inline void trace_tile() {
    if constexpr (VFloat) {
        vfloat_block<0, 0, 0>(); vfloat_block<4, 0, 1>(); vfloat_block<8, 0, 2>(); vfloat_block<12, 0, 3>();
        vfloat_block<16, 1, 0>(); vfloat_block<20, 1, 1>(); vfloat_block<24, 1, 2>(); vfloat_block<28, 1, 3>();
    } else {
        raw_block<0, 0, 0>(); raw_block<4, 0, 1>(); raw_block<8, 0, 2>(); raw_block<12, 0, 3>();
        raw_block<16, 1, 0>(); raw_block<20, 1, 1>(); raw_block<24, 1, 2>(); raw_block<28, 1, 3>();
    }
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
    if constexpr (TRACE_IMPL == 0) _llk_math_welfords_sfpu_params_(trace_tile<false>, 0);
    else _llk_math_welfords_sfpu_params_(trace_tile<true>, 0);
    // Final state is written after all input reads to two dedicated tiles.
    // This has the same format/instruction as the normal final store, without
    // colliding with trace tiles 1-4.
    TTI_SFPSTORE(p_sfpu::LREG4, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 320);
    TTI_SFPSTORE(p_sfpu::LREG5, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 384);
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
    // The LLK pack argument is the Dst tile index, not a count.  Pack each
    // dedicated trace tile to its own host result buffer.
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(1, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(2, L1_ADDRESS(params.buffer_Res[1]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(3, L1_ADDRESS(params.buffer_Res[2]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(4, L1_ADDRESS(params.buffer_Res[3]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(5, L1_ADDRESS(params.buffer_Res[4]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(6, L1_ADDRESS(params.buffer_Res[5]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(7, L1_ADDRESS(params.buffer_Res[6]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif
