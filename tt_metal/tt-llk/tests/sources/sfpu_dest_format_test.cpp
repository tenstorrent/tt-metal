// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_debug.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_MATH
#include "experimental/llk_math_hw_cleanup.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace ckernel;

extern "C" [[gnu::noinline, gnu::flatten]] void configure_math_format_for_test()
{
    static_assert(FORMAT_WRITER <= 5);
    constexpr std::uint32_t format = RAW_SRCB;
    if constexpr (FORMAT_WRITER == 0)
    {
        _llk_math_hw_configure_<false>(to_underlying(DEST_FORMAT), format);
    }
    else if constexpr (FORMAT_WRITER == 1)
    {
        _llk_math_reconfig_data_format_<false>(to_underlying(DEST_FORMAT), format);
    }
    else if constexpr (FORMAT_WRITER == 2)
    {
        _llk_math_reconfig_data_format_<false, true>(to_underlying(DEST_FORMAT), format);
    }
    else if constexpr (FORMAT_WRITER == 3)
    {
        _llk_math_reconfig_data_format_srcb_<false>(format);
    }
    else if constexpr (FORMAT_WRITER == 4)
    {
        _llk_math_reconfig_data_format_srcb_<false, true>(format);
    }
    else
    {
        _llk_math_reconfig_data_format_srca_<false>(format);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    auto* out = reinterpret_cast<std::uint32_t*>(params.buffer_Res[0]);
    if constexpr (SWITCH_BANKS)
    {
        flip_cfg_state_id();
        _llk_math_hw_configure_<true>(to_underlying(DataFormat::Float16_b), to_underlying(DataFormat::Float16_b));
        _llk_math_hw_configure_<false>(to_underlying(DataFormat::Float16_b), to_underlying(DataFormat::Float16_b));
        flip_cfg_state_id();
    }
    constexpr auto initial_format = (WAIT_PROBE || SWITCH_BANKS || POISON_CACHE || INITIAL_SRCB != 255)
                                        ? DEST_FORMAT
                                        : (DEST_FORMAT == DataFormat::Float16 ? DataFormat::Float16_b : DataFormat::Float16);
    _llk_math_hw_configure_<false>(to_underlying(initial_format), INITIAL_SRCB == 255 ? to_underlying(initial_format) : INITIAL_SRCB);
    if constexpr (POISON_CACHE)
    {
        _llk_math_hw_cleanup_configure_current_bank_<false>();
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_RMW>(to_underlying(DataFormat::Float16));
        _llk_math_hw_cleanup_configure_current_bank_<false>();
        tensix_sync();
        out[1024] = (cfg_read(ALU_FORMAT_SPEC_REG1_SrcB_ADDR32) & ALU_FORMAT_SPEC_REG1_SrcB_MASK) >> ALU_FORMAT_SPEC_REG1_SrcB_SHAMT;
        return;
    }
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::abs>();
    {
        if constexpr (DEST_FORMAT == DataFormat::Float16)
        {
            sfpi::dst_reg[0].template mode<sfpi::DataLayout::F16a>() = 1.5f;
        }
        else
        {
            sfpi::dst_reg[0].template mode<sfpi::DataLayout::F16b>() = 1.5f;
        }
    }
    math::reset_counters(p_setrwc::SET_ABD_F);
    tensix_sync();

    if constexpr (SWITCH_BANKS)
    {
        flip_cfg_state_id();
    }
    if constexpr (WAIT_PROBE)
    {
        sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::dst_reg[0]   = value + 1.0f;
        configure_math_format_for_test();
    }
    else
    {
        // Pass the raw CB format, bypassing the harness's FP8-to-Float16 ALU rewrite.
        configure_math_format_for_test();
        tensix_sync();
        out[1024]          = (cfg_read(ALU_FORMAT_SPEC_REG1_SrcB_ADDR32) & ALU_FORMAT_SPEC_REG1_SrcB_MASK) >> ALU_FORMAT_SPEC_REG1_SrcB_SHAMT;
        sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::dst_reg[0]   = value + 1.0f;
    }
    math::reset_counters(p_setrwc::SET_ABD_F);
    {
        sfpi::vFloat value;
        if constexpr (DEST_FORMAT == DataFormat::Float16)
        {
            value = sfpi::dst_reg[0].template mode<sfpi::DataLayout::F16a>();
        }
        else
        {
            value = sfpi::dst_reg[0].template mode<sfpi::DataLayout::F16b>();
        }
        sfpi::dst_reg[0].template mode<sfpi::DataLayout::F32>() = value;
    }
    dbg_copy_dest_tile<DbgDestTileOp::Read>(DataFormat::Float32, 0, out);
    out[1024] = (cfg_read(ALU_FORMAT_SPEC_REG1_SrcB_ADDR32) & ALU_FORMAT_SPEC_REG1_SrcB_MASK) >> ALU_FORMAT_SPEC_REG1_SrcB_SHAMT;
    if constexpr (WAIT_PROBE)
    {
        // Export reference encodings from the architecture headers for the ELF ordering check.
        out[1026] = TT_OP_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        out[1027] = TT_OP_RMWCIB2((ALU_FORMAT_SPEC_REG1_SrcB_MASK >> 16) & 0xff, 0, ALU_FORMAT_SPEC_REG1_SrcB_ADDR32);
        out[1028] = TT_OP_RMWCIB3((ALU_FORMAT_SPEC_REG1_SrcB_MASK >> 24) & 0xff, 0, ALU_FORMAT_SPEC_REG1_SrcB_ADDR32);
        out[1029] = ~TT_OP_RMWCIB2(0, 0xff, 0) | TT_OP_RMWCIB2(0, 0, 0);
    }
    if constexpr (SWITCH_BANKS)
    {
        flip_cfg_state_id();
    }
}
#else
void run_kernel(RUNTIME_PARAMETERS)
{
}
#endif
