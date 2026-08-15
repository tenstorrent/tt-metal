// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Test-only typed spellings for the two architectural TopK operations whose
// side effects cover both the value (L0--L3) and index (L4--L7) banks.  The
// multi-result builtins keep the coupled dataflow visible to allocation and
// scheduling while preserving the production LLK as the A/B baseline.
namespace topk_typed_multiresult
{

template <unsigned ValueA, unsigned ValueB, unsigned Mod>
__attribute__((always_inline)) inline void compare_swap()
{
    static_assert(ValueA < 4 && ValueB < 4);
    auto value_a = __builtin_rvtt_sfpreadlreg(ValueA);
    auto value_b = __builtin_rvtt_sfpreadlreg(ValueB);
    auto index_a = __builtin_rvtt_sfpreadlreg(ValueA + 4);
    auto index_b = __builtin_rvtt_sfpreadlreg(ValueB + 4);
    // The ISA assembly spelling is destination,source while TTI_SFPSWAP's
    // C macro parameters are source,destination.  Keep that historical API
    // contract explicit at this single typed boundary.
    auto result = __builtin_rvtt_sfpswap_indexed(value_b, value_a, index_b, index_a, Mod);
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpselect4(result, 0), ValueB);
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpselect4(result, 1), ValueA);
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpselect4(result, 2), ValueB + 4);
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpselect4(result, 3), ValueA + 4);
}

__attribute__((always_inline)) inline void transpose_value_and_index_banks()
{
    auto value0 = __builtin_rvtt_sfpreadlreg(0);
    auto value1 = __builtin_rvtt_sfpreadlreg(1);
    auto value2 = __builtin_rvtt_sfpreadlreg(2);
    auto value3 = __builtin_rvtt_sfpreadlreg(3);
    auto index0 = __builtin_rvtt_sfpreadlreg(4);
    auto index1 = __builtin_rvtt_sfpreadlreg(5);
    auto index2 = __builtin_rvtt_sfpreadlreg(6);
    auto index3 = __builtin_rvtt_sfpreadlreg(7);
    auto result = __builtin_rvtt_sfptransp8(value0, value1, value2, value3, index0, index1, index2, index3);
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpselect4(result, 0), 0);
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpselect4(result, 1), 1);
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpselect4(result, 2), 2);
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpselect4(result, 3), 3);

    // The backend models L4--L7 as additional results in the same PARALLEL.
    // Fixed-LREG reads make those architectural results live at this boundary.
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpreadlreg(4), 4);
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpreadlreg(5), 5);
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpreadlreg(6), 6);
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpreadlreg(7), 7);
}

} // namespace topk_typed_multiresult

#undef TTI_SFPSWAP
#define TTI_SFPSWAP(imm12_math, lreg_src_c, lreg_dest, instr_mod1) \
    topk_typed_multiresult::compare_swap<lreg_src_c, lreg_dest, instr_mod1>()

#undef TTI_SFPTRANSP
#ifdef ARCH_QUASAR
#define TTI_SFPTRANSP topk_typed_multiresult::transpose_value_and_index_banks()
#else
#define TTI_SFPTRANSP(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    topk_typed_multiresult::transpose_value_and_index_banks()
#endif
