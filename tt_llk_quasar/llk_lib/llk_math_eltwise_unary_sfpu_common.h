// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "cmath_common.h"
#include "llk_defs.h"
using namespace ckernel::math;

/**
 * @brief Programs SFPU addrmods
 */
inline void _eltwise_unary_sfpu_configure_addrmod_()
{
    // TODO (RT): Ask if addrmods are still shared between fpu and sfpu?
    // NOTE: this kernel is typically used in conjunction with
    // A2D, which is using ADDR_MOD_0 and ADDR_MOD_1, so use one
    // that doesn't conflict!

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7, csr_read<CSR::TRISC_ID>());
}

/**
 * @brief Sets up starting index of SFPU, Stalls till all FPU operations are done
 * @param tile_index: Use to index to a tile in Destination register
 */
inline void _llk_math_eltwise_unary_sfpu_start_(const uint tile_index)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_index);
    TTI_STALLWAIT(p_stall::STALL_SFPU, 0, 0, p_stall::MATH);
}

/**
 * @brief Resets dest counter to 0
 */
inline void _llk_math_eltwise_unary_sfpu_done_()
{
    _reset_counters_<p_setrwc::SET_D>();
}

/**
 * @brief Clear SrcS valids
 * @tparam SRCS_RD_DONE: Whether the source reg S read is done
 * @tparam SRCS_WR_DONE: Whether the source reg S write is done
 */
template <bool SRCS_RD_DONE, bool SRCS_WR_DONE>
inline void _llk_math_eltwise_unary_sfpu_srcs_clear_vlds_()
{
    TTI_SFPNOP(SRCS_WR_DONE, SRCS_RD_DONE, 0);
}

/**
 * @brief Increments dest counter by a face (16x16 default)
 */
inline void _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()
{
    _inc_dst_addr_<16>();
}

/**
 * @brief Initialization for SFPU operations
 */
inline void _llk_math_eltwise_unary_sfpu_init_()
{
    _init_sfpu_config_reg_();
    _eltwise_unary_sfpu_configure_addrmod_();
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Runs SFPU operation for a tile (default 32x32)
 * @tparam APPROXIMATE: Some sfpu functions have 2 implementations
 * either less accurate and more performant mode, or more accurate but less performant mode
 * APPROXIMATE flag is set for more performant but less accurate mode
 * @param: sfpu_func: function pointer to the sfpu functions to run, can look at list of functions here: common/inc/sfpu/cmath_sfpu*
 * @param: dst_tile_index: Starting tile index in the destination register, values = 0 - 15
 * @param: args: variable number of args can be passed into this function, that will be passed
 * to the SFPU function pointer
 */
template <bool APPROXIMATE, class F, class... ARGS>
inline void _llk_math_eltwise_unary_sfpu_params_(F&& sfpu_func, uint dst_tile_index, ARGS&&... args)
{
    _llk_math_eltwise_unary_sfpu_start_(dst_tile_index);

    for (uint face = 0; face < NUM_FACES; face++)
    {
        sfpu_func(static_cast<ARGS&&>(args)...);

        // Move to the next face
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
    }

    _llk_math_eltwise_unary_sfpu_done_();
}

/**
 * @brief Determines the stochround conversion type based on source and cast data formats
 */
template <DataFormat SRC_FMT, DataFormat CAST_FMT>
inline constexpr uint _sfpu_stochround_conversion_()
{
    if constexpr (SRC_FMT == DataFormat::Float32 && CAST_FMT == DataFormat::Float16)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::FP32_TO_FP16A;
    }
    else if constexpr (SRC_FMT == DataFormat::Float32 && CAST_FMT == DataFormat::Float16_b)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::FP32_TO_FP16B;
    }
    else if constexpr (SRC_FMT == DataFormat::Float32 && CAST_FMT == DataFormat::Uint8)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::FP32_TO_UINT8;
    }
    else if constexpr (SRC_FMT == DataFormat::Float32 && CAST_FMT == DataFormat::Int8)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::FP32_TO_INT8;
    }
    else if constexpr (SRC_FMT == DataFormat::Int32 && CAST_FMT == DataFormat::Uint8)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::INT32_TO_UINT8;
    }
    else if constexpr (SRC_FMT == DataFormat::Int32 && CAST_FMT == DataFormat::Int8)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::INT32_TO_INT8;
    }
    else if constexpr (SRC_FMT == DataFormat::Float32 && CAST_FMT == DataFormat::Int16)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::FP32_TO_INT16;
    }
    else
    {
        static_assert(
            !std::is_same_v<decltype(SRC_FMT), DataFormat>,
            "Unsupported DataFormats for stochround conversion"); // need the condition to depend on the template parameter... compiler things
    }
}

/**
 * @brief Determines the sfpmem type parameter based on data format
 */
template <DataFormat FMT>
inline constexpr uint _sfpu_sfpmem_type_()
{
    if constexpr (FMT == DataFormat::Float16)
    {
        return ckernel::p_sfpu::sfpmem::FP16A;
    }
    else if constexpr (FMT == DataFormat::Float16_b)
    {
        return ckernel::p_sfpu::sfpmem::FP16B;
    }
    else if constexpr (FMT == DataFormat::Float32)
    {
        return ckernel::p_sfpu::sfpmem::FP32;
    }
    else if constexpr (FMT == DataFormat::Int32)
    {
        return ckernel::p_sfpu::sfpmem::INT32;
    }
    else
    {
        static_assert(
            !std::is_same_v<decltype(FMT), DataFormat>,
            "Unsupported DataFormat for sfpmem type determination"); // need the condition to depend on the template parameter... compiler things
    }
}
