// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Test-only adaptation of ckernel_sfpu_topk.h::_bitonic_topk_merge.
// Preserve the production loop, instruction words, formats and addressing.
// Only the load8 -> swap -> store8 region gets explicit C++ value lifetimes.
// Raw issue deliberately bypasses effect annotations: this is an independent
// alternative, not threaded values accidentally protected by the effect pass.
// TOPK_IMPL=2 injects no stress temporaries into the measured workload.
// TOPK_IMPL=3 adds a typed load/store identity roundtrip solely for correctness
// stress; it is not a production workload or a performance comparison arm.
// Include after llk_sfpu/ckernel_sfpu_topk.h.
namespace ckernel::sfpu
{
namespace topk_threaded_merge
{
template <bool is_fp32_dest_acc_en, bool top_min, bool STABLE_SORT>
__attribute__((always_inline)) inline void merge8(std::uint32_t offset, std::uint32_t dist)
{
    constexpr std::uint32_t dst_indices_offset = 128;
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    constexpr InstrModLoadStore instr_mod_value = TOPK_UINT16_IN_FP32_DEST ? InstrModLoadStore::INT32 : InstrModLoadStore::DEFAULT;
    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset = (offset & 0xF) + face_offset * 32;

    // These are the four production TT_SFPLOAD words, with each value captured
    // immediately; delaying all captures until after load8 hides earlier lives.
    instrn_buffer[0] = TT_OP_SFPLOAD(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, ld_offset);
    auto value0 = __builtin_rvtt_sfpreadlreg(0);
    if constexpr (TOPK_IMPL == 3)
    {
        // Preserve value0 across an allocator-owned vector temporary. The
        // roundtrip uses the OTHER compare operand and nonincrementing
        // address mode, so accidental L0 reuse changes value0. It must leave
        // the input unchanged in the BF16/FP16
        // finite-value test domain. Parentheses bypass SFPI arity macros.
        auto gap = (__builtin_rvtt_sfpload)(instrn_buffer, ld_offset + dist, 0, 0, static_cast<unsigned>(instr_mod_value), ADDR_MOD_7);
        (__builtin_rvtt_sfpstore)(instrn_buffer, gap, ld_offset + dist, 0, 0, static_cast<unsigned>(instr_mod_value), ADDR_MOD_7);
    }
    instrn_buffer[0] = TT_OP_SFPLOAD(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, ld_offset + dist);
    auto value1 = __builtin_rvtt_sfpreadlreg(1);
    instrn_buffer[0] = TT_OP_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset);
    auto index0 = __builtin_rvtt_sfpreadlreg(4);
    instrn_buffer[0] = TT_OP_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset + dist);
    auto index1 = __builtin_rvtt_sfpreadlreg(5);

    __builtin_rvtt_sfpwritelreg(value0, 0);
    __builtin_rvtt_sfpwritelreg(value1, 1);
    __builtin_rvtt_sfpwritelreg(index0, 4);
    __builtin_rvtt_sfpwritelreg(index1, 5);
    INSTRUCTION_WORD(TT_OP_SFPSWAP(0, top_min ? p_sfpu::LREG1 : p_sfpu::LREG0, top_min ? p_sfpu::LREG0 : p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX));
    if constexpr (STABLE_SORT)
    {
        // Same second swap/stall as production. No C++ operation intervenes.
        INSTRUCTION_WORD(TT_OP_SFPSWAP(0, top_min ? p_sfpu::LREG1 : p_sfpu::LREG0, top_min ? p_sfpu::LREG0 : p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX));
    }

    // Index tracking updates L4/L5 together with L0/L1. Capture all four NEW
    // results; restoring the pre-swap inputs here would undo the raw operation.
    auto result0 = __builtin_rvtt_sfpreadlreg(0);
    auto result1 = __builtin_rvtt_sfpreadlreg(1);
    auto result_index0 = __builtin_rvtt_sfpreadlreg(4);
    auto result_index1 = __builtin_rvtt_sfpreadlreg(5);
    __builtin_rvtt_sfpwritelreg(result0, 0);
    instrn_buffer[0] = TT_OP_SFPSTORE(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, ld_offset);
    __builtin_rvtt_sfpwritelreg(result1, 1);
    instrn_buffer[0] = TT_OP_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, ld_offset + dist);
    __builtin_rvtt_sfpwritelreg(result_index0, 4);
    instrn_buffer[0] = TT_OP_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset);
    __builtin_rvtt_sfpwritelreg(result_index1, 5);
    instrn_buffer[0] = TT_OP_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset + dist);
}
} // namespace topk_threaded_merge

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min = false, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_merge_threaded(std::uint32_t m_iter_arg, std::uint32_t k_arg)
{
    // The production public wrapper forwards unsigned arguments to an int
    // implementation. Preserve that conversion and its expression types.
    const int m_iter = m_iter_arg;
    const int k = k_arg;
    topk_uint16_clear_value_tiles_high_bits();
    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
            int k_max = k > 32 ? 32 : k;
            std::uint32_t inner_d = k_max >> 2;
            std::uint32_t total_datums_to_compare = ((64 >> m_iter) < 2 * k_max) ? 2 * k_max : (64 >> m_iter);
            std::uint32_t dist = (k_max << m_iter) > 32 ? 32 : (k_max << m_iter);
            std::uint32_t ld_dist = (dist < 16) ? dist : 2 * dist;
            std::uint32_t datums_compared = 0;
            std::uint32_t dst_offset = 0;
            std::uint32_t dst_cr = 0;
            while (datums_compared < total_datums_to_compare)
            {
                for (std::uint32_t ii = 0; ii < inner_d; ii++)
                {
                    topk_threaded_merge::merge8<is_fp32_dest_acc_en, top_min, STABLE_SORT>(dst_offset, ld_dist);
                    datums_compared += 8;
                    if (ii == (inner_d - 1))
                    {
                        dst_cr += 2 * dist;
                        dst_offset = dst_cr;
                    }
                    else
                    {
                        dst_offset += 4;
                    }
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
}
} // namespace ckernel::sfpu
