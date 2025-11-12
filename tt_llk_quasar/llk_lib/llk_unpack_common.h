// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cunpack_common.h"
using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @brief Programs unpacker l1 info & source register format
 * @tparam UNP_SEL: Sets unpacker to configure. values = p_unpacr::UNP_A/UNP_B/UNP_S
 * @param tdma_desc_src: Contains L1 buffer descriptor information & source reg format for Src Reg
 */
template <uint32_t UNP_SEL>
inline void _llk_unpack_hw_configure_(const tdma_descriptor_t& tdma_desc_src)
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_S) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_S/UNP_DEST");

    // Populate the buffer descriptor table
    _configure_buf_desc_table_(tdma_desc_src.buf_desc_id, tdma_desc_src.buf_desc);

    // RT: make defines to aggregate the source format address, to make the below a single function
    // Program src formats
    if constexpr (UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_DEST)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_OUT_DATA_FORMAT_RMW, static_cast<uint8_t>(tdma_desc_src.reg_data_format));
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B)
    {
        cfg_rmw(THCON_UNPACKER1_REG0_OUT_DATA_FORMAT_RMW, static_cast<uint8_t>(tdma_desc_src.reg_data_format));
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_S)
    {
        cfg_rmw(THCON_UNPACKER2_REG0_OUT_DATA_FORMAT_RMW, static_cast<uint8_t>(tdma_desc_src.reg_data_format));
    }
}

// RT: make defines to aggregate _llk_unpack_hw_configure_ calls into one
/**
 * @brief Programs unpacker l1 info & source register format for unary operation
 * @tparam UNP_SEL: Sets unpacker to configure. values = p_unpacr::UNP_A/UNP_B/UNP_S
 * @param tdma_desc_src: Contains L1 buffer descriptor information & source reg format for Src Reg
 */
template <uint32_t UNP_SEL>
inline void _llk_unpack_configure_unary_(const tdma_descriptor_t& tdma_desc_src)
{
    _llk_unpack_hw_configure_<UNP_SEL>(tdma_desc_src);
}

/**
 * @brief Programs unpacker l1 info & source register format for binary operation
 * @tparam UNP_SEL0/1: Sets unpacker to configure. values = p_unpacr::UNP_A/UNP_B/UNP_S
 * @param tdma_desc_src0/1: Contains L1 buffer descriptor information & source reg format for Src Reg
 */
template <uint32_t UNP_SEL_0, uint32_t UNP_SEL_1>
inline void _llk_unpack_configure_binary_(const tdma_descriptor_t& tdma_desc_src0, const tdma_descriptor_t& tdma_desc_src1)
{
    _llk_unpack_hw_configure_<UNP_SEL_0>(tdma_desc_src0);
    _llk_unpack_hw_configure_<UNP_SEL_1>(tdma_desc_src1);
}

// template <bool IS_FP32_MATH_DEST_EN>
inline void _llk_unpack_dest_dvalid_section_done_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::WAIT_SFPU, p_stall::UNPACK0);
    TTI_CLEARDVALID(0, 0, 0, 0, p_cleardvalid::UNPACK_TO_DEST, 0);
}
