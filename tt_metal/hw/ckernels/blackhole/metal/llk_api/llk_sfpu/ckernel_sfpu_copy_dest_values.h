// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "tensix_types.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Generalized copy_dest_value that works with any DataFormat
template <DataFormat DATA_FORMAT, bool APPROXIMATION_MODE, int ITERATIONS = 8>
void copy_dest_value(const uint dst_index_in, const uint dst_index_out, const uint /* unused */) {
    constexpr InstrModLoadStore instr_mod_index = GetSfpLoadStoreInstrMod<DATA_FORMAT>();
    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size = 64;
    // For some reason using __builtin_rvtt_sfp{load,store} here
    // results in test failures.  The compiler unrolls this loop
    // and with the builtin emits assembly directly, rather than
    // synthesize the insn.  Presumably the same problem occurs
    // with using sfpi -- if it was extended to expose the
    // ADDR_MOD PR #41879
    // Four rows by offset per counter step, so the eight words stay in registers.
    const std::uint32_t load = TT_OP_SFPLOAD(p_sfpu::LREG0, instr_mod_index, ADDR_MOD_7, dst_index_in * dst_tile_size);
    const std::uint32_t store = TT_OP_SFPSTORE(p_sfpu::LREG0, instr_mod_index, ADDR_MOD_7, dst_index_out * dst_tile_size);
    static_assert(ITERATIONS % 4 == 0);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 4) {
        TT_INSN(load);
        TT_INSN(store);
        TT_INSN(load + 2);
        TT_INSN(store + 2);
        TT_INSN(load + 4);
        TT_INSN(store + 4);
        TT_INSN(load + 6);
        TT_INSN(store + 6);
        TTI_INCRWC(0, 8, 0, 0);
    }
}

// Deprecated: Use the DataFormat template parameter version instead
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
[[deprecated("Use copy_dest_value<DataFormat, APPROXIMATION_MODE, ITERATIONS> instead")]]
void copy_dest_value(const uint dst_index_in, const uint dst_index_out, const uint /* unused */) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = sfpi::vFloat(sfpi::dst_reg[dst_index_in * dst_tile_size_sfpi]);
        dst_reg++;
    }
}

void copy_dest_value_init() {
    // No initialization required
}

}  // namespace sfpu
}  // namespace ckernel
