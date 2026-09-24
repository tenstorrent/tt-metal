// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "tensix_types.h"
#include "llk_math_eltwise_binary_sfpu_params.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Generalized copy_dest_value that works with any DataFormat
template <DataFormat DATA_FORMAT, bool APPROXIMATION_MODE, int ITERATIONS = 8>
void copy_dest_value(
    const std::uint32_t dst_index_in, const std::uint32_t dst_index_out, const std::uint32_t /* unused */) {
    constexpr InstrModLoadStore instr_mod_index = GetSfpLoadStoreInstrMod<DATA_FORMAT>();
    // size of each tile in Dest is 64 rows
    constexpr std::uint32_t dst_tile_size = 64;
    for (int d = 0; d < ITERATIONS; d++) {
        // For some reason using __builtin_rvtt_sfp{load,store} here
        // results in test failures.  The compiler unrolls this loop
        // and with the builtin emits assembly directly, rather than
        // synthesize the insn.  Presumably the same problem occurs
        // with using sfpi -- if it was extended to expose the
        // ADDR_MOD PR #41879
        TT_SFPLOAD(p_sfpu::LREG0, instr_mod_index, ADDR_MOD_7, dst_index_in * dst_tile_size);
        TT_SFPSTORE(p_sfpu::LREG0, instr_mod_index, ADDR_MOD_7, dst_index_out * dst_tile_size);
        dst_reg++;
    }
}

// Deprecated: Use the DataFormat template parameter version instead
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
[[deprecated("Use copy_dest_value<DataFormat, APPROXIMATION_MODE, ITERATIONS> instead")]]
void copy_dest_value(
    const std::uint32_t dst_index_in, const std::uint32_t dst_index_out, const std::uint32_t /* unused */) {
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] =
            sfpi::vFloat(sfpi::dst_reg[dst_index_in * dst_tile_size_sfpi]);
        dst_reg++;
    }
}

void copy_dest_value_init() {
    // No initialization required
}

// Op class for copying one Dest tile onto another. Only run() needs DATA_FORMAT.
template <bool APPROXIMATION_MODE, DataFormat DATA_FORMAT = DataFormat::Invalid, int ITERATIONS = 8>
struct CopyDestValue : SfpuBinaryOp<CopyDestValue<APPROXIMATION_MODE, DATA_FORMAT, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate(
        const std::uint32_t dst_index_in, const std::uint32_t dst_index_out, const std::uint32_t dst_index_unused) {
        copy_dest_value<DATA_FORMAT, APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out, dst_index_unused);
    }
    static inline __attribute__((always_inline)) void init_op() { copy_dest_value_init(); }
};

}  // namespace sfpu
}  // namespace ckernel
